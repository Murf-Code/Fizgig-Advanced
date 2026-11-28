"""QH-1: fixed-code Q2 Hadamard scale fitting with matched Q2/Q3 controls."""
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
import mixed_factor_screen as mix
import hadamard_scale_core as hs
from binary_qat_coverage import KEYS,validate_bank,read_panel
from rank_representation_screen import panel_score

COMMON=(.2,.4,.4)


def operator(d):
    # Exact execution convention of the existing direct odd-code control.
    def op(x,g):
        x=geo.hadamard(x.bfloat16()); g=g.bfloat16()
        ys=[]; dx=torch.zeros((len(g),d.shape[1]),dtype=torch.float32)
        for i in range(0,len(d),geo.TILE):
            tile=d[i:i+geo.TILE]
            ys.append(F.linear(x,tile))
            dx.add_((g[:,i:i+len(tile)]@tile).float())
        return torch.cat(ys,dim=-1),geo.hadamard(dx).bfloat16()
    return op


def run(w,key,args,stock,trace,split,int8,row,save):
    m,n=w.shape
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("highest")
    fit=[read_panel(stock,key,r,trace,m,n) for r in split["fit"]]
    dev=[read_panel(stock,key,r,trace,m,n) for r in split["development"]]
    native=codec.stock_layer(w,int8)
    dr=[base.refs(w,c,native) for c in dev]
    z=geo.hadamard(w.float()).double()
    x=geo.hadamard(torch.cat([c["x"].float() for c in fit])).double()
    g=torch.cat([c["dy"].double() for c in fit])
    row.update(shape=[m,n],device=str(device),group=64,fit_count=len(fit),fit_rows=len(x),
               candidates={},confirmation_scored=False,common_fit_weights=list(COMMON))
    def archive(name,c,s,bits):
        tensors={"codes":codec._pack(c,bits),"scales":s.contiguous()}
        nb=sum(t.nbytes for t in tensors.values())
        if nb!=m*(n*bits//8+2*(n//64)): raise ValueError("payload mismatch")
        path=args.out.with_name(args.out.stem+"-"+name+".safetensors")
        if path.exists(): raise ValueError("archive already exists: "+str(path))
        tmp=path.with_suffix(".tmp")
        save_file(tensors,str(tmp),metadata={"key":key,"shape":json.dumps([m,n]),"group":"64",
                                             "bits":str(bits),"protocol":"qh-1-fixed-code-scales-v1",
                                             "codebook":"odd-symmetric","transform":"sylvester-h64"})
        t=load_file(str(tmp))
        cc=codec._unpack(t["codes"],m*n,bits).reshape(m,n).byte()
        if not torch.equal(c,cc) or not torch.equal(s,t["scales"]):
            raise ValueError("archive roundtrip mismatch")
        tmp.replace(path)
        return dict(file=path.name,sha256=codec.sha256(path),payload_bytes=nb,file_bytes=path.stat().st_size)
    def record(name,c,s,bits,**extra):
        d=hs.decode(c,s,bits)
        fit_score=hs.metrics(z,x,g,d,COMMON)
        ds=panel_score(operator(d),dev,dr)
        ar=archive(name,c,s,bits)
        row["candidates"][name]={"payload_bytes":ar["payload_bytes"],
            "effective_bits":8*ar["payload_bytes"]/(m*n),"development":ds,
            "pooled_fit_fp64":fit_score,"weight_error":math.sqrt(fit_score["weight_relative_squared"]),
            "archive":ar,**extra}
        print(key,name,"dev",ds["objective"],"dx",ds["raw_dx_rms"],"fit",fit_score["objective"],flush=True)
        save()
        return ds
    for bits in (2,3):
        c,s=hs.initial_codes(z,bits)
        name=f"q{bits}-baseline"
        # Independently evaluate original H4 implementation; abort on parity failure.
        legacy,nb,_=mix.direct_lowbit(w,bits)
        old=panel_score(legacy,dev,dr)
        new=panel_score(operator(hs.decode(c,s,bits)),dev,dr)
        delta={k:new[k]-old[k] for k in ("objective","raw_forward_rms","raw_dx_rms")}
        if nb!=m*(n*bits//8+2*(n//64)) or max(map(abs,delta.values()))>1e-8:
            raise ValueError("legacy Q2/Q3 score parity failed")
        prior_delta=None
        if args.reference is not None:
            prior=json.loads(args.reference.read_text(encoding="utf-8"))
            previous=next(v for v in prior["projections"] if v["key"]==key)["candidates"][f"direct-q{bits}-odd-had"]["development"]
            prior_delta={k:new[k]-previous[k] for k in delta}
            if max(map(abs,prior_delta.values()))>1e-5: raise ValueError("H4 reference score parity failed")
        del legacy
        record(name,c,s,bits,legacy_delta=delta,h4_reference_delta=prior_delta)
        if bits==3: continue
        for label,weights in (("q2-task-scales",COMMON),("q2-weight50-scales",(.5,.25,.25))):
            t=time.monotonic()
            sys=hs.ScaleSystem(z,x,g,c,bits,weights,device=device,
                               dtype=torch.float32 if device.type=="cuda" else torch.float64)
            proposal,solver=hs.solve(sys,s,maxiter=args.cg_iterations)
            proposal=proposal.cpu()
            del sys
            if device.type=="cuda": torch.cuda.empty_cache()
            ss,guard=hs.rounded_guard(z,x,g,c,bits,s,proposal,weights)
            record(label,c,ss,bits,fit_weights=list(weights),solver=solver,guard=guard,
                   code_changes=0,scale_changes=int((ss!=s).sum()),
                   seconds=time.monotonic()-t)
        del c,s
    row["q2_development_winner"]=min((k for k in row["candidates"] if k.startswith("q2-")),
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
    report={"status":"running","protocol":"qh-1-fixed-code-scales-v1","torch":torch.__version__,
            "threads":a.threads,"cg_iterations":a.cg_iterations,"projections":[],
            "scope":"offline fixed-code Hadamard scale fit on reused H3 fit/development; no confirmation or deployment"}
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
