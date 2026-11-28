"""QH-1/2/3/4 combined exploratory Hadamard comparison."""
import argparse
import json
import math
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file,load_file
from fit_queue import atomic_json
import native_codec as codec
import geometry_codec as geo
import binary_rank_screen as base
import hadamard_scale_core as hs
import hadamard_scale_screen as qh1
import hadamard_combined_core as hc
from binary_qat_coverage import KEYS,validate_bank,read_panel
from rank_representation_screen import panel_score

COMMON=(.2,.4,.4)
WEIGHT50=(.5,.25,.25)


def operator(d,transform):
    def op(x,g):
        x=geo.hadamard(x.bfloat16(),transform); g=g.bfloat16()
        ys=[]; dx=torch.zeros((len(g),d.shape[1]),dtype=torch.float32)
        for i in range(0,len(d),geo.TILE):
            tile=d[i:i+geo.TILE]
            ys.append(F.linear(x,tile))
            dx.add_((g[:,i:i+len(tile)]@tile).float())
        return torch.cat(ys,dim=-1),geo.hadamard(dx,transform).bfloat16()
    return op


def run(w,key,args,stock,trace,split,int8,row,save):
    # Preserve QH-1 verbatim, including legacy/reference parity gates.
    qh1.run(w,key,args,stock,trace,split,int8,row,save)
    row["status"]="running"
    row["scope"]="QH-1 scales; QH-2 codes; QH-3 fixed geometry; QH-4 row precision"
    save()
    m,n=w.shape
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype=torch.float32 if device.type=="cuda" else torch.float64
    fit=[read_panel(stock,key,r,trace,m,n) for r in split["fit"]]
    dev=[read_panel(stock,key,r,trace,m,n) for r in split["development"]]
    native=codec.stock_layer(w,int8)
    dr=[base.refs(w,c,native) for c in dev]
    raw_x=torch.cat([c["x"].float() for c in fit])
    g=torch.cat([c["dy"].double() for c in fit])
    z=geo.hadamard(w.float()).double()
    x=geo.hadamard(raw_x).double()

    def load(name,bits):
        ar=row["candidates"][name]["archive"]
        path=args.out.parent/ar["file"]
        if codec.sha256(path)!=ar["sha256"]: raise ValueError("archive hash mismatch")
        t=load_file(str(path))
        if sum(v.nbytes for v in t.values())!=ar["payload_bytes"]: raise ValueError("archive length mismatch")
        c=codec._unpack(t["codes"],m*n,bits).reshape(m,n).byte()
        return c,t["scales"]

    def archive(name,tensors,meta,expected):
        nb=sum(t.nbytes for t in tensors.values())
        if nb!=expected: raise ValueError("packed size mismatch")
        path=args.out.with_name(args.out.stem+"-"+name+".safetensors")
        if path.exists(): raise ValueError("archive already exists: "+str(path))
        tmp=path.with_suffix(".tmp")
        save_file(tensors,str(tmp),metadata={"key":key,"shape":json.dumps([m,n]),"protocol":"qh-combined-v1",
                                           "codebook":"odd-symmetric",**meta})
        actual=load_file(str(tmp))
        if set(actual)!=set(tensors) or not all(torch.equal(t,actual[k]) for k,t in tensors.items()):
            raise ValueError("archive roundtrip mismatch")
        tmp.replace(path)
        return dict(file=path.name,sha256=codec.sha256(path),payload_bytes=nb,file_bytes=path.stat().st_size)

    def record(name,d,ar,zz,xx,had_size,**extra):
        fit_score=hs.metrics(zz,xx,g,d,COMMON)
        ds=panel_score(operator(d,had_size),dev,dr)
        row["candidates"][name]={"payload_bytes":ar["payload_bytes"],
            "effective_bits":8*ar["payload_bytes"]/(m*n),"development":ds,
            "pooled_fit_fp64":fit_score,"weight_error":math.sqrt(fit_score["weight_relative_squared"]),
            "archive":ar,**extra}
        print(key,name,"dev",ds["objective"],"dx",ds["raw_dx_rms"],"fit",fit_score["objective"],flush=True)
        save()

    def uniform(name,c,s,bits,zz,xx,transform=64,group=64,**extra):
        packed=codec._pack(c,bits)
        if not torch.equal(codec._unpack(packed,m*n,bits).reshape(m,n).byte(),c):
            raise ValueError("code roundtrip mismatch")
        ar=archive(name,{"codes":packed,"scales":s.contiguous()},
                   {"group":str(group),"bits":str(bits),"transform":f"sylvester-h{transform}"},
                   m*(n*bits//8+2*(n//group)))
        record(name,hs.decode(c,s,bits),ar,zz,xx,transform,group=group,transform=transform,**extra)

    def refit(c,s,bits,weights):
        sys=hs.ScaleSystem(z,x,g,c,bits,weights,device=device,dtype=dtype)
        proposal,info=hs.solve(sys,s,maxiter=args.cg_iterations)
        proposal=proposal.cpu()
        del sys
        ss,guard=hs.rounded_guard(z,x,g,c,bits,s,proposal,weights)
        return ss,dict(solver=info,guard=guard)

    c2,s2=load("q2-baseline",2); c3,s3=load("q3-baseline",3)
    for name,source,bits,weights,scales_after in (
        ("q2-task-codes","q2-baseline",2,COMMON,False),
        ("q2-task-alternating","q2-task-scales",2,COMMON,True),
        ("q2-weight50-alternating","q2-weight50-scales",2,WEIGHT50,True),
        ("q3-task-alternating","q3-baseline",3,COMMON,True)):
        t=time.monotonic()
        c,s=load(source,bits)
        stats=[]
        # Two code sweeps. Alternating arms refit after each sweep.
        model=hc.Fit(z,x,g,weights,device=device,dtype=dtype)
        for sweep in range(2):
            before=hs.metrics(z,x,g,hs.decode(c,s,bits),weights)
            cc,info=hc.code_sweep(c,s,bits,model,seed=42+sweep)
            after=hs.metrics(z,x,g,hs.decode(cc,s,bits),weights)
            accepted=after["objective"]<=before["objective"]+1e-12
            if accepted: c=cc
            stage=dict(sweep=sweep+1,code_solver=info,code_accepted=accepted,before=before,proposed=after)
            if scales_after:
                s,stage["scale_refit"]=refit(c,s,bits,weights)
            stats.append(stage)
            print(key,name,"sweep",sweep+1,"code changes",info["net_changed"],"accepted",accepted,flush=True)
        del model
        uniform(name,c,s,bits,z,x,source=source,fit_weights=list(weights),substeps=stats,seconds=time.monotonic()-t)

    # Same group64 payload: only transform width changes. No learned rotations.
    for transform in (1,32,128):
        if n%transform: continue
        zz=geo.hadamard(w.float(),transform).double()
        xx=geo.hadamard(raw_x,transform).double()
        c,s=hs.initial_codes(zz,2)
        uniform(f"q2-h{transform}-g64",c,s,2,zz,xx,transform=transform)
        del zz,xx,c,s

    # Uniform group32 and Q2/Q3 row mix share a 2.5-bit cap.
    c,s=hs.initial_codes(z,2,group=32)
    uniform("q2-h64-g32",c,s,2,z,x,group=32)
    del c,s
    d2,d3=hs.decode(c2,s2,2),hs.decode(c3,s3,3)
    count=hc.mixed_count(m,n)
    if count:
        model=hc.Fit(z,x,g,COMMON,device=device,dtype=dtype)
        priority=hc.row_priority(d2,d3,model)
        del model
        if not torch.isfinite(priority).all(): raise ValueError("invalid row priorities")
        for name,indices in (
            ("mixed-q2q3-random",torch.randperm(m,generator=torch.Generator().manual_seed(42))[:count]),
            ("mixed-q2q3-targeted",torch.argsort(priority,stable=True)[:count])):
            mask=torch.zeros(m,dtype=torch.bool); mask[indices]=True
            tensors={"row_mask":codec._pack(mask,1),"codes2":codec._pack(c2[~mask],2),
                     "codes3":codec._pack(c3[mask],3),"scales":torch.where(mask[:,None],s3,s2).contiguous()}
            nb=(m+7)//8+(m-count)*n//4+count*n*3//8+2*m*(n//64)
            if nb>m*n*5//16: raise ValueError("mixed budget exceeded")
            mm=codec._unpack(tensors["row_mask"],m,1).bool()
            if not torch.equal(mm,mask): raise ValueError("row mask roundtrip mismatch")
            for bits,select,source in ((2,~mm,c2),(3,mm,c3)):
                unpacked=codec._unpack(tensors[f"codes{bits}"],int(select.sum())*n,bits).reshape(-1,n).byte()
                if not torch.equal(unpacked,source[select]): raise ValueError("mixed code roundtrip mismatch")
            ar=archive(name,tensors,{"group":"64","bits":"mixed-2-3","transform":"sylvester-h64",
                                     "q3_rows":str(count)},nb)
            d=torch.where(mask[:,None],d3,d2)
            record(name,d,ar,z,x,64,q3_rows=count,cap_bits=2.5,
                   isolated_delta_sum=float(priority[mask].sum()),group=64,transform=64)
    row["q2_development_winner"]=min((k for k,v in row["candidates"].items() if k.startswith("q2-") and v["effective_bits"]<=2.25),
        key=lambda k:row["candidates"][k]["development"]["objective"])
    row["budget_2p5_winner"]=min((k for k,v in row["candidates"].items() if v["effective_bits"]<=2.5),
        key=lambda k:row["candidates"][k]["development"]["objective"])
    row["status"]="finished"
    save()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw",type=Path,required=True)
    ap.add_argument("--stock",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--cg-iterations",type=int,default=30)
    ap.add_argument("--reference",type=Path)
    ap.add_argument("--threads",type=int,default=8)
    ap.add_argument("--key",choices=KEYS,default=KEYS[1])
    a=ap.parse_args()
    if min(a.cg_iterations,a.threads)<1 or a.out.exists(): ap.error("positive settings and fresh report required")
    torch.set_num_threads(a.threads)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    report={"status":"running","protocol":"qh-combined-v1","torch":torch.__version__,
            "threads":a.threads,"cg_iterations":a.cg_iterations,"projections":[],
            "scope":"offline QH-1/2/3/4 screen on reused H3 fit/development; no confirmation or deployment"}
    def save():
        report["seconds"]=time.monotonic()-start
        atomic_json(a.out,report)
    try:
        trace=json.loads((a.stock/"trace.json").read_text(encoding="utf-8"))
        prep=json.loads((a.stock/"prepare.json").read_text(encoding="utf-8"))
        if trace.get("case")!="stock" or not trace.get("completed_updates") or prep.get("status")!="stock_verified":
            raise ValueError("completed V5 stock required")
        raw_hash=codec.sha256(a.raw)
        if raw_hash!=prep["raw_sha256"]: raise ValueError("raw checkpoint mismatch")
        report["raw_sha256"]=raw_hash
        split=validate_bank(a.stock,trace)
        report["split"]={k:[{"file":r["file"],"step":r["step"],"sha256":codec.sha256(a.stock/r["file"])}
                           for r in records] for k,records in split.items()}
        if a.reference is not None:
            ref=json.loads(a.reference.read_text(encoding="utf-8"))
            if (ref.get("protocol")!="lb-h4-rank-reference-v1" or ref.get("status")!="completed"
                or ref.get("raw_sha256")!=raw_hash or ref.get("split")!=report["split"]):
                raise ValueError("H4 reference raw hash or split mismatch")
        report["fit_partition_used"]=True
        report["confirmation_scored"]=False
        save()
        root=Path(__file__).resolve().parents[3]
        int8=codec.import_file("binary_int8",root/"src/fizgig/modules/int8_train.py")
        from contextlib import ExitStack
        with ExitStack() as stack:
            raw=stack.enter_context(safe_open(str(a.raw),framework="pt",device="cpu"))
            key=a.key
            w=raw.get_tensor(key)
            if w.dtype!=torch.bfloat16 or not torch.isfinite(w).all(): raise ValueError("finite Raw BF16 required")
            row={"key":key,"status":"running"}
            report["projections"].append(row)
            run(w,key,a,a.stock,trace,split,int8,row,save)
            del w
        if codec.sha256(a.raw)!=raw_hash: raise ValueError("raw changed")
        for records in report["split"].values():
            for r in records:
                if codec.sha256(a.stock/r["file"])!=r["sha256"]: raise ValueError("capture changed")
        report["status"]="completed"
    except Exception as e:
        report.update(status="error",error=str(e))
        raise
    finally:
        save()


if __name__=="__main__":
    main()
