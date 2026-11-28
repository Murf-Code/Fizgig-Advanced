"""QH-4B: matched row precision allocation curve across all captured projections."""
import argparse
import json
import math
import time
import traceback
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file,load_file
from fit_queue import atomic_json
import native_codec as codec
import geometry_codec as geo
import binary_rank_screen as base
import hadamard_scale_core as hs
import allocation_core as ac
from hadamard_combined_screen import operator
from binary_qat_coverage import KEYS,validate_bank,read_panel
from rank_representation_screen import panel_score

COMMON=(.2,.4,.4)


def pack_rows(c,bits):
    # Chunking is byte-identical because each row is a multiple of 8 values.
    p=torch.cat([codec._pack(t,bits) for t in c.split(128)]) if len(c) else torch.empty(0,dtype=torch.uint8)
    step=c.shape[1]*bits//8
    for i in range(0,len(c),128):
        j=min(i+128,len(c))
        decoded=codec._unpack(p[i*step:j*step],(j-i)*c.shape[1],bits).reshape_as(c[i:j]).byte()
        if not torch.equal(decoded,c[i:j]): raise ValueError("packed code mismatch")
    return p


@torch.no_grad()
def run(w,key,args,stock,trace,split,int8,row,save):
    m,n=w.shape
    row["stage"]="capture_read"
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # FP32 action matmuls, FP64 scalar reductions on both devices; bounded
    # memory for MLP. Dev continues using original CPU BF16 execution.
    torch.set_float32_matmul_precision("highest")
    fit=[read_panel(stock,key,r,trace,m,n) for r in split["fit"]]
    dev=[read_panel(stock,key,r,trace,m,n) for r in split["development"]]
    row["stage"]="stock_reference"
    save()
    native=codec.stock_layer(w,int8)
    # Reference backward uses autograd even though allocation is offline.
    with torch.enable_grad():
        dr=[base.refs(w,c,native) for c in dev]
    row["stage"]="quantizer_initialization"
    save()
    z=geo.hadamard(w.float())
    x=geo.hadamard(torch.cat([c["x"].float() for c in fit]))
    g=torch.cat([c["dy"].float() for c in fit])
    c2,s2=hs.initial_codes(z,2,tile=8192)
    c3,s3=hs.initial_codes(z,3,tile=8192)
    d2,d3=hs.decode(c2,s2,2),hs.decode(c3,s3,3)
    row["stage"]="allocation_initialization"
    save()
    model=ac.Allocation(z,x,g,d2,d3,device=device,dtype=torch.float32)
    zg=z.to(device)
    row.update(shape=[m,n],device=str(device),group=64,transform=64,fit_count=len(fit),fit_rows=len(x),
               candidates={},confirmation_scored=False,common_fit_weights=list(COMMON),fit_precision="FP32 actions, FP64 sums",
               allocation_rounds=8,shortlist=32)
    prior=json.loads(args.reference.read_text(encoding="utf-8"))
    previous=next((r["candidates"] for r in prior["projections"] if r["key"]==key),{})
    def direct(d):
        e=d.to(zg)-zg
        v=[float(e.square().sum(dtype=torch.float64)),
           float((model.x@e.T).square().sum(dtype=torch.float64)),
           float((model.g@e).square().sum(dtype=torch.float64))]
        # Compute teacher energies from the same original model, not candidate.
        vals=[v[i]/energy[i] for i in range(3)]
        if not all(math.isfinite(t) for t in vals): raise ValueError("nonfinite direct fit")
        return dict(weight_relative_squared=vals[0],forward_relative_squared=vals[1],
                    backward_relative_squared=vals[2],objective=sum(a*b for a,b in zip(COMMON,vals)))
    energy=[float(zg.square().sum(dtype=torch.float64)),
            float((model.x@zg.T).square().sum(dtype=torch.float64)),
            float((model.g@zg).square().sum(dtype=torch.float64))]
    def archive(name,tensors,expected,**meta):
        nb=sum(t.nbytes for t in tensors.values())
        if nb!=expected: raise ValueError("packed size mismatch")
        path=args.out.with_name(args.out.stem+"-"+key.replace(".","_")+"-"+name+".safetensors")
        if path.exists(): raise ValueError("archive already exists: "+str(path))
        tmp=path.with_suffix(".tmp")
        try:
            save_file(tensors,str(tmp),metadata={"key":key,"shape":json.dumps([m,n]),
                "protocol":"qh-allocation-v1","codebook":"odd-symmetric","transform":"sylvester-h64",**meta})
            actual=load_file(str(tmp))
            if set(actual)!=set(tensors) or not all(torch.equal(t,actual[k]) for k,t in tensors.items()):
                raise ValueError("archive roundtrip mismatch")
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
        return dict(file=path.name,sha256=codec.sha256(path),payload_bytes=nb,file_bytes=path.stat().st_size)
    def record(name,d,ar,fit_score=None,**extra):
        row["stage"]="score:"+name
        score=direct(d) if fit_score is None else fit_score
        ds=panel_score(operator(d,64),dev,dr)
        row["candidates"][name]={"payload_bytes":ar["payload_bytes"],"effective_bits":8*ar["payload_bytes"]/(m*n),
            "development":ds,"pooled_fit_fp32":score,"weight_error":math.sqrt(score["weight_relative_squared"]),
            "archive":ar,**extra}
        if name in previous:
            prev=previous[name]["development"]
            delta={k:ds[k]-prev[k] for k in ("objective","raw_forward_rms","raw_dx_rms")}
            row["candidates"][name]["previous_development_delta"]=delta
            if name in ("q2-baseline","q3-baseline","q2-h64-g32") and max(map(abs,delta.values()))>1e-5:
                raise ValueError("prior baseline parity failed: "+name)
        print(key,name,"bits",row["candidates"][name]["effective_bits"],"dev",ds["objective"],
              "dx",ds["raw_dx_rms"],"fit",score["objective"],flush=True)
        save()
    def uniform(name,c,s,bits,group,d):
        ar=archive(name,{"codes":pack_rows(c,bits),"scales":s.contiguous()},
                   m*(n*bits//8+2*n//group),group=str(group),bits=str(bits))
        record(name,d,ar,group=group)
    uniform("q2-baseline",c2,s2,2,64,d2)
    uniform("q3-baseline",c3,s3,3,64,d3)
    cc,ss=hs.initial_codes(z,2,group=32,tile=8192)
    uniform("q2-h64-g32",cc,ss,2,32,hs.decode(cc,ss,2))
    del cc,ss,z,x,g
    priority=model.priority().cpu()
    weight=model.wdelta.cpu()
    if not torch.isfinite(priority).all() or not torch.isfinite(weight).all(): raise ValueError("invalid priorities")
    random=torch.randperm(m,generator=torch.Generator().manual_seed(42))
    ranks={"random":random,"weight":weight.argsort(stable=True),"targeted":priority.argsort(stable=True)}
    for cap in ac.CAPS:
        count,nbcap=ac.budget(m,n,cap)
        if not 0<count<m: raise ValueError("nontrivial mixed budget required")
        tag=f"{cap/8:g}"
        masks={}
        for label,order in ranks.items():
            mask=torch.zeros(m,dtype=torch.bool); mask[order[:count]]=True; masks[label]=mask
        row["stage"]="refinement:"+tag
        save()
        joint,info=model.refine(masks["targeted"])
        # Independent direct full-action FP32 check for the final joint mask.
        # Fall back to targeted if the direct check does not confirm improvement.
        target_score=direct(torch.where(masks["targeted"][:,None],d3,d2))
        joint_score=direct(torch.where(joint[:,None],d3,d2))
        accepted=joint_score["objective"]<=target_score["objective"]+1e-9
        masks["joint"]=joint if accepted else masks["targeted"].clone()
        for label,mask in masks.items():
            if int(mask.sum())!=count: raise ValueError("allocation count mismatch")
            d=torch.where(mask[:,None],d3,d2)
            tensors={"row_mask":codec._pack(mask,1),"codes2":pack_rows(c2[~mask],2),
                     "codes3":pack_rows(c3[mask],3),"scales":torch.where(mask[:,None],s3,s2).contiguous()}
            nb=(m+7)//8+(m-count)*n//4+count*n*3//8+2*m*(n//64)
            if nb>nbcap or not torch.equal(codec._unpack(tensors["row_mask"],m,1).bool(),mask):
                raise ValueError("mixed mask or budget mismatch")
            name=f"mix-{tag}-{label}"
            ar=archive(name,tensors,nb,group="64",bits="mixed-2-3",q3_rows=str(count),cap_bits=tag)
            extra=dict(cap_bits=cap/8,q3_rows=count,q3_fraction=count/m,
                       isolated_delta_sum=float(priority[mask].sum()),allocation=label)
            if label=="joint": extra.update(refinement=info,direct_guard_accepted=accepted,
                                            direct_proposed_fit=joint_score)
            score=target_score if label=="targeted" else (joint_score if accepted else target_score) if label=="joint" else None
            record(name,d,ar,fit_score=score,**extra)
        # Compare the old 2.5-bit random/targeted scores; differences in
        # floating point ranking are recorded rather than hidden.
        if cap==20:
            for label in ("random","targeted"):
                oldname="mixed-q2q3-"+label
                if oldname in previous:
                    current=row["candidates"][f"mix-{tag}-{label}"]
                    current["previous_development_delta"]={k:current["development"][k]-previous[oldname]["development"][k]
                        for k in ("objective","raw_forward_rms","raw_dx_rms")}
        save()
    row["budget_winners"]={f"{cap/8:g}":min((k for k,v in row["candidates"].items() if v["effective_bits"]<=cap/8),
        key=lambda k:row["candidates"][k]["development"]["objective"]) for cap in ac.CAPS}
    row["status"]="finished"
    row["stage"]="finished"
    save()


def preflight(int8):
    """Exercise the actual 19-arm runner with synthetic tensors and real INT8 code."""
    rng=torch.Generator().manual_seed(123)
    m,n=41,128
    w=torch.randn(m,n,generator=rng).bfloat16()
    panels={}
    for i in range(4):
        panels[i]={"x":torch.randn(17,n,generator=rng).bfloat16(),
                   "dy":torch.randn(17,m,generator=rng).bfloat16(),
                   "step":i,"item":str(i),"timestep":.5}
    split={"fit":[0,1],"development":[2,3]}
    with tempfile.TemporaryDirectory(prefix="qh-allocation-preflight-") as tmp:
        root=Path(tmp)
        ref=root/"reference.json"
        ref.write_text(json.dumps({"projections":[]}),encoding="utf-8")
        args=SimpleNamespace(reference=ref,out=root/"test.json")
        row={}
        with patch(__name__+".read_panel",side_effect=lambda stock,key,r,trace,m,n:panels[r]):
            run(w,"synthetic",args,root,{},split,int8,row,lambda:None)
        if row.get("status")!="finished" or len(row["candidates"])!=19:
            raise ValueError("allocation preflight incomplete")
        if len(list(root.glob("*.safetensors")))!=19:
            raise ValueError("allocation preflight archive count mismatch")
        return {"status":"passed","candidates":19,"device":row["device"]}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw",type=Path)
    ap.add_argument("--stock",type=Path)
    ap.add_argument("--out",type=Path)
    ap.add_argument("--reference",type=Path)
    ap.add_argument("--self-test",action="store_true",help="run only a small end-to-end synthetic preflight")
    ap.add_argument("--threads",type=int,default=8)
    ap.add_argument("--key",choices=KEYS,default=None,help="omit to test all three captured projections")
    a=ap.parse_args()
    if a.threads<1: ap.error("positive threads required")
    torch.set_num_threads(a.threads)
    root=Path(__file__).resolve().parents[3]
    int8=codec.import_file("binary_int8",root/"src/fizgig/modules/int8_train.py")
    source_sha=codec.sha256(Path(__file__))
    print("QH-4B runner revision: preflight-v2",source_sha,flush=True)
    if a.self_test:
        print("Preflight:",preflight(int8),flush=True)
        return
    if any(v is None for v in (a.raw,a.stock,a.out,a.reference)) or a.out.exists():
        ap.error("raw, stock, out, reference and fresh report required")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    report={"status":"running","protocol":"qh-allocation-v1","torch":torch.__version__,
            "runner_revision":"preflight-v2","runner_sha256":source_sha,
            "threads":a.threads,"budgets":[v/8 for v in ac.CAPS],"projections":[],
            "scope":"offline QH-4B allocation curve on reused H3 fit/development; no confirmation or deployment"}
    def save():
        report["seconds"]=time.monotonic()-start
        atomic_json(a.out,report)
    try:
        report["stage"]="preflight"
        save()
        report["preflight"]=preflight(int8)
        report["stage"]="bank_verification"
        save()
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
            if (ref.get("protocol")!="qh-combined-v1" or ref.get("status")!="completed"
                or ref.get("raw_sha256")!=raw_hash or ref.get("split")!=report["split"]):
                raise ValueError("combined reference raw hash or split mismatch")
        report["fit_partition_used"]=True
        report["confirmation_scored"]=False
        save()
        report["stage"]="projection_screen"
        from contextlib import ExitStack
        with ExitStack() as stack:
            raw=stack.enter_context(safe_open(str(a.raw),framework="pt",device="cpu"))
            for key in ((a.key,) if a.key else KEYS):
                w=raw.get_tensor(key)
                if w.dtype!=torch.bfloat16 or not torch.isfinite(w).all(): raise ValueError("finite Raw BF16 required")
                row={"key":key,"status":"running"}
                report["projections"].append(row)
                run(w,key,a,a.stock,trace,split,int8,row,save)
                del w
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        if codec.sha256(a.raw)!=raw_hash: raise ValueError("raw changed")
        for records in report["split"].values():
            for r in records:
                if codec.sha256(a.stock/r["file"])!=r["sha256"]: raise ValueError("capture changed")
        report["status"]="completed"
        report["stage"]="completed"
    except Exception as e:
        report.update(status="error",error=str(e),traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__=="__main__":
    main()
