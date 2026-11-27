"""Real V5 capture screen for binary factorization (offline, CPU only)."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from safetensors import safe_open
import native_codec as codec
import private_correction as pc
import binary_rank_core as core
from offline_corrections import read_capture
from fit_queue import atomic_json

KEYS=("blocks.0.attn.wk.weight","blocks.14.attn.wk.weight","blocks.0.mlp.up.weight")


def refs(w,c,int8):
    x,g=c["x"].bfloat16(),c["dy"].bfloat16()
    z=x.detach().requires_grad_(True)
    y=int8(z)
    dx,=torch.autograd.grad(y,z,g)
    return (F.linear(x,w),g@w,y.detach(),dx.detach())


def score(op,c,ref):
    y,dx=op(c["x"],c["dy"])
    ry,rg,iy,ig=ref
    return {"step":c["step"],"item":c["item"],"timestep":c["timestep"],
            "raw_forward":pc.relative(y,ry),"raw_dx":pc.relative(dx,rg),
            "int8_forward":pc.relative(y,iy),"int8_dx":pc.relative(dx,ig)}


def objective(rows):
    vals=[]
    for r in rows:
        v=[r[k]**2 for k in ("raw_forward","raw_dx") if r[k] is not None]
        if v: vals.append(sum(v)/len(v))
    return sum(vals)/len(vals) if vals else None


def run(w,key,args,stock,trace,split,int8,archive,row,save):
    m,n=w.shape
    int8=codec.stock_layer(w,int8)
    dev=[read_capture(stock,key,r,trace) for r in split["development"]]
    dr=[refs(w,c,int8) for c in dev]
    row.update(shape=[m,n],native_int8_bytes=m*n+4*m,candidates={})
    def add(name,op,nbytes,rank,weight,kind):
        scores=[score(op,c,r) for c,r in zip(dev,dr)]
        v=objective(scores)
        row["candidates"][name]={"kind":kind,"rank":rank,"payload_bytes":nbytes,
            "effective_bits":8*nbytes/(m*n),"weight_error":weight,
            "dense_equivalent_macs_per_token":rank*(m+n) if rank else m*n,
            "development":scores,"development_objective":v}
        print(key,name,"bits",round(8*nbytes/(m*n),3),"objective",v,flush=True)
        save()
        return float("inf") if v is None else v
    # Direct Q4/Q6 controls. Q6 reads existing V5 data if supplied, not re-encoded.
    controls={}
    for name in ("q4-had","q6-had"):
        if name=="q6-had" and archive is not None:
            p=archive.get_tensor(key); s=codec.manifest(archive)[key]
            if s["codec"]!=name or s["shape"]!=[m,n]: raise ValueError("Q6 archive mismatch")
            codec.validate(p,s)
        else:
            p,s=codec.encode(w,name)
        op=lambda x,g,p=p,s=s:pc.operator(p,s,x,g)
        add(name,op,p.nbytes,0,pc.weight_error(w,p,s),"uniform-control")
        controls[name]=(p,s)
    ranks=core.rank_grid(m,n,args.max_rank)
    row["rank_grid"]=ranks
    print(key,"computing one shared randomized SVD sketch",ranks[-1],flush=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        u,s,v=torch.svd_lowrank(w.float(),q=min(ranks[-1]+16,m,n),niter=4)
    row["svd_scope"]="randomized sketch, not a certified optimal rank-error lower bound"
    best={}
    # Equal rank/bytes for the two binary controls. Scaled variant charges
    # additional row/column bytes; select within explicit total bit caps.
    caps=(.25,.5,1.,2.,3.)
    row["bit_caps"]=list(caps)
    def consider(name,a,b,scales,rank,bytes_,val):
        if not math.isfinite(val): return
        for cap in caps:
            if 8*bytes_>cap*m*n: continue
            old=best.get(cap)
            if old is None or (val,bytes_,name)<old[:3]:
                # Shared references when the same candidate wins multiple caps.
                best[cap]=(val,bytes_,name,a,b,scales)
    for r in ranks:
        a,b=core.balanced(u,s,v,r)
        # Unbinarized BF16 factor control isolates rank/sketch loss. Not
        # eligible for binary budget selection.
        af,bf=a.bfloat16(),b.bfloat16()
        add(f"svd-r{r}",lambda x,g:core.actions(x,g,af,bf),af.nbytes+bf.nbytes,
            r,core.weight_error(w,af,bf),"rank-diagnostic")
        del af,bf
        for rotated in (False,True):
            aa,bb=core.rotate(a,b) if rotated else (a,b)
            da,db,nbytes=core.binarize(aa,bb,codec._pack,codec._unpack)
            if nbytes!=core.binary_bytes(m,n,r): raise ValueError("binary payload accounting mismatch")
            name=f"{'lb2-block128' if rotated else 'lb1'}-r{r}"
            val=add(name,lambda x,g:core.actions(x,g,da,db),nbytes,r,
                    core.weight_error(w,da,db),"binary")
            consider(name,da,db,None,r,nbytes,val)
            if rotated:
                rr,cc=core.scale_fit(w,da,db)
                scaled=name+"-rowcol"
                charged=nbytes+rr.nbytes+cc.nbytes
                if charged!=core.binary_bytes(m,n,r,True): raise ValueError("scale accounting mismatch")
                val=add(scaled,lambda x,g:core.actions(x,g,da,db,rr,cc),charged,r,
                        core.weight_error(w,da,db,rr,cc),"binary-scaled")
                consider(scaled,da,db,(rr,cc),r,charged,val)
            del aa,bb,da,db
        del a,b
    del u,s,v
    row["development_winners"]={str(cap):best[cap][2] if cap in best else None for cap in caps}
    save()
    # Confirmation capture is first read after selection is frozen.
    conf=[read_capture(stock,key,r,trace) for r in split["confirmation"]]
    cr=[refs(w,c,int8) for c in conf]
    finalists={name:(lambda x,g,p=p,s=s:pc.operator(p,s,x,g)) for name,(p,s) in controls.items()}
    for _,_,name,a,b,scales in best.values():
        finalists[name]=(lambda x,g,a=a,b=b,scales=scales:core.actions(x,g,a,b,*(scales or (None,None))))
    for name,op in finalists.items():
        scores=[score(op,c,r) for c,r in zip(conf,cr)]
        row["candidates"][name]["confirmation"]=scores
        row["candidates"][name]["confirmation_objective"]=objective(scores)
    row["status"]="finished"
    save()


def main(run_fn=run, protocol="binary-rank-v1"):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw",type=Path,required=True)
    ap.add_argument("--stock",type=Path,required=True)
    ap.add_argument("--archive",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--max-rank",type=int,default=2048)
    ap.add_argument("--threads",type=int,default=8)
    ap.add_argument("--key",action="append",choices=sorted(codec.ELIGIBLE))
    a=ap.parse_args()
    if min(a.max_rank,a.threads)<1 or a.out.exists(): ap.error("positive settings and fresh report required")
    torch.set_num_threads(a.threads)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    report={"status":"running","protocol":protocol,"torch":torch.__version__,
            "threads":a.threads,"max_rank":a.max_rank,"projections":[],
            "scope":"offline local actions; no compression archive or training; no new independent LoRA holdout"}
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
        split=pc.split_captures(trace)
        report["split"]={k:[{"file":r["file"],"step":r["step"],"sha256":codec.sha256(a.stock/r["file"])}
                           for r in records] for k,records in split.items()}
        # Fit partition is reserved for a later task-aware fitting experiment.
        report["fit_partition_used"]=False
        save()
        root=Path(__file__).resolve().parents[3]
        int8=codec.import_file("binary_int8",root/"src/fizgig/modules/int8_train.py")
        from contextlib import ExitStack
        with ExitStack() as stack:
            raw=stack.enter_context(safe_open(str(a.raw),framework="pt",device="cpu"))
            archive=None
            if a.archive:
                report["archive_sha256"]=codec.sha256(a.archive)
                archive=stack.enter_context(safe_open(str(a.archive),framework="pt",device="cpu"))
                if (archive.metadata() or {}).get("source_sha256")!=raw_hash:
                    raise ValueError("archive source mismatch")
            for key in dict.fromkeys(a.key or KEYS):
                w=raw.get_tensor(key)
                if w.dtype!=torch.bfloat16: raise ValueError("Raw BF16 required")
                row={"key":key,"status":"running"}
                report["projections"].append(row)
                run_fn(w,key,a,a.stock,trace,split,int8,archive,row,save)
                del w
        if codec.sha256(a.raw)!=raw_hash: raise ValueError("raw changed")
        if a.archive and codec.sha256(a.archive)!=report["archive_sha256"]: raise ValueError("archive changed")
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
