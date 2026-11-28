"""LB-H10: progressive primary/residual QAT with matched reset control."""
import argparse
import json
import math
import time
from pathlib import Path
import torch
from torch import nn
from safetensors import safe_open
from fit_queue import atomic_json
import native_codec as codec
import binary_rank_screen as base
import binary_rank_core as old
import qat_core as q
import binary_qat_screen as e
import binary_qat_objective as h
from binary_qat_coverage import KEYS,validate_bank,read_panel,pooled_fit,score

WEIGHTS=(.2,.4,.4)
STEPS=100
SHARES=(50,75,25)


def allocations(m,n,cap,top):
    # Same total rank for all arms, including padding and both scale headers.
    equal=q.ranks(m,n,cap,top,2)
    if not equal: raise ValueError("no feasible binary allocation")
    total=sum(equal)
    result={}
    for share in SHARES:
        small=total//4
        a={50:total//2,75:total-small,25:small}[share]
        b=total-a
        if min(a,b)<1 or q.size(m,n,a)+q.size(m,n,b)>cap:
            raise ValueError("rank split exceeds budget or has empty path")
        result[share]=(a,b)
    return result


@torch.no_grad()
def backward_metric(w,g):
    w=w.double(); g=g.double()
    we=w.square().sum()
    ge=(g@w).square().sum()
    if not all(torch.isfinite(v) and v>0 for v in (we,ge)):
        raise ValueError("positive finite original teacher energies required")
    return torch.linalg.cholesky(torch.eye(w.shape[0],dtype=w.dtype,device=w.device)
                                 +(WEIGHTS[2]/WEIGHTS[0])*(we/ge)*(g.T@g))


@torch.no_grad()
def unwhiten(a,b,L):
    b=torch.linalg.solve_triangular(L.T,b.double(),upper=True).float()
    a=a.float()
    # Product-preserving latent rescaling balances individual factor norms.
    ar=a.norm(dim=1); br=b.norm(dim=0)
    scale=torch.where((ar>0)&(br>0),(br/ar.clamp_min(1e-30)).sqrt(),torch.ones_like(ar))
    return (a*scale[:,None]).contiguous(),(b/scale[None,:]).contiguous()


def continuous_components(a,b):
    a,b=a.detach().cpu().bfloat16(),b.detach().cpu().bfloat16()
    m,r=b.shape; n=a.shape[1]
    return (a,b,torch.ones(m,dtype=torch.bfloat16),torch.ones(n,dtype=torch.bfloat16),
            torch.ones(r,dtype=torch.bfloat16))


@torch.no_grad()
def residual_target(w,first):
    target=w.float().clone()
    cs,nb=e.packed([first],*w.shape)
    c=tuple(t.to(w.device,dtype=torch.float32) for t in cs[0])
    for i in range(0,len(w),256):
        idx=slice(i,min(i+256,len(w)))
        target[idx]-=q.weight(c,idx)
    return target


def run(w,key,args,stock,trace,split,int8,row,save):
    import copy
    import progressive_core as schedule
    import latent_scale_core as ls
    from safetensors.torch import save_file,load_file
    m,n=w.shape
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("highest")
    budget=m*n//8; top=min(m,n,args.max_rank)
    ranks=q.ranks(m,n,budget,top,2)
    if not ranks: raise ValueError("no feasible equal ranks")
    r1,r2=ranks
    native=codec.stock_layer(w,int8)
    panels=[read_panel(stock,key,r,trace,m,n) for r in split["fit"]]
    dev=[read_panel(stock,key,r,trace,m,n) for r in split["development"]]
    dr=[base.refs(w,c,native) for c in dev]
    wg=w.to(device)
    fit=pooled_fit(wg,panels)
    wf=w.double()
    x=torch.cat([c["x"].double() for c in panels])
    g=torch.cat([c["dy"].double() for c in panels])
    L=backward_metric(w,fit[0][1].cpu())
    Lg=L.to(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        u,s,v=torch.svd_lowrank((L.T@wf).float(),q=min(top+16,m,n),niter=4)
    def init(a,b):
        with torch.no_grad():
            a,b=q.rotate(a.to(device),b.to(device))
            return q.Path(a,b)
    a,b=old.balanced(u,s,v,r1)
    a,b=unwhiten(a,b,L)
    primary=init(a,b).cpu()
    del u,s,v,a,b
    row.update(shape=[m,n],device=str(device),budget_bytes=budget,ranks=ranks,
               candidates={},fit_count=len(panels),fit_rows=len(x),confirmation_scored=False,
               objective_weights=list(WEIGHTS),sign_lr=.00005,scale_lr=h.SCALE_LR,
               warmup=20,max_steps=100,phase_boundary=40,selection_rule="fixed final step 100")
    def make_primary():
        return copy.deepcopy(primary).to(device)
    def fit_residual(first):
        target=(Lg.T@residual_target(wg,first).double()).float()
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=="cuda" else []):
            torch.manual_seed(43)
            u,s,v=torch.svd_lowrank(target,q=min(r2+16,m,n),niter=4)
        a,b=old.balanced(u,s,v,r2)
        del target,u,s,v
        a,b=unwhiten(a,b,Lg)
        return init(a,b)
    first=make_primary()
    shared_residual=fit_residual(first).cpu()
    del first
    def archive(name,cs,nb):
        tensors={}
        for i,(a,b,r,s,c) in enumerate(cs):
            tensors.update({f"{i}.a":codec._pack(a>0,1),f"{i}.b":codec._pack(b>0,1),
                            f"{i}.row":r.contiguous(),f"{i}.col":s.contiguous(),f"{i}.latent":c.contiguous()})
        if sum(t.nbytes for t in tensors.values())!=nb: raise ValueError("archive payload mismatch")
        path=args.out.with_name(args.out.stem+"-"+name+".safetensors")
        if path.exists(): raise ValueError("archive already exists: "+str(path))
        tmp=path.with_suffix(".tmp")
        save_file(tensors,str(tmp),metadata={"key":key,"ranks":json.dumps(ranks),"shape":json.dumps([m,n]),"protocol":"lb-h10-progressive-v1"})
        restored=load_file(str(tmp))
        if not all(torch.equal(t,restored[k]) for k,t in tensors.items()): raise ValueError("archive roundtrip mismatch")
        tmp.replace(path)
        return {"file":path.name,"sha256":codec.sha256(path),"payload_bytes":nb,"file_bytes":path.stat().st_size}
    for name in schedule.ARMS:
        started=time.monotonic()
        entry={"history":[],"phases":[]}
        row["candidates"][name]=entry
        def record(paths,stats):
            cs,nb=e.packed(paths,m,n)
            if nb>budget or nb!=sum(q.size(m,n,p.a.shape[0]) for p in paths):
                raise ValueError("packed budget mismatch")
            A,B,c=ls.atoms(cs)
            record={**stats,"paths":len(paths),"payload_bytes":nb,"effective_bits":8*nb/(m*n),
                    "development":score(w,dev,dr,cs),"pooled_fit_fp64":ls.metrics(A,B,c,wf,x,g),
                    "weight_error":e.error(w,cs)}
            entry["history"].append(record)
            print(key,name,stats["phase"],stats["step"],"dev",record["development"]["objective"],
                  "dx",record["development"]["raw_dx_rms"],flush=True)
            save()
        def optimize(paths,steps,warmup,callback):
            t=time.monotonic()
            h.optimize(nn.ModuleList(paths),wg,fit,.00005,warmup=warmup,steps=steps,callback=callback,weights=WEIGHTS)
            if device.type=="cuda": torch.cuda.synchronize()
            entry["phases"].append({"paths":len(paths),"updates":steps,"warmup":warmup,
                                    "seed":42,"seconds_including_evaluations":time.monotonic()-t})
            save()
        def make_residual(first):
            return fit_residual(first) if name=="progressive" else copy.deepcopy(shared_residual).to(device)
        paths=schedule.run_schedule(name,make_primary,make_residual,optimize,record)
        cs,nb=e.packed(paths,m,n)
        if len(paths)!=2 or nb!=sum(q.size(m,n,r) for r in ranks): raise ValueError("final payload mismatch")
        if entry["history"][-1]["step"]!=100: raise ValueError("missing final evaluation")
        entry.update(final=entry["history"][-1],archive=archive(name,cs,nb),seconds=time.monotonic()-started)
        del paths,cs
        if device.type=="cuda": torch.cuda.empty_cache()
        save()
    joint=row["candidates"]["joint"]["history"]
    reset=row["candidates"]["joint-reset"]["history"]
    for step in (0,20):
        a=next(v for v in joint if v["step"]==step)
        b=next(v for v in reset if v["step"]==step)
        if abs(a["development"]["objective"]-b["development"]["objective"])>1e-5 or abs(a["pooled_fit_fp64"]["objective"]-b["pooled_fit_fp64"]["objective"])>1e-5:
            raise ValueError("joint/reset prefix parity failed")
    row["joint_reset_prefix_parity"]=True
    row["development_winner"]=min(row["candidates"],key=lambda k:row["candidates"][k]["final"]["development"]["objective"])
    row["status"]="finished"
    save()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw",type=Path,required=True)
    ap.add_argument("--stock",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--max-rank",type=int,default=2048)
    ap.add_argument("--threads",type=int,default=8)
    ap.add_argument("--key",choices=KEYS,default=KEYS[1])
    a=ap.parse_args()
    if min(a.max_rank,a.threads)<1 or a.out.exists(): ap.error("positive settings and fresh report required")
    torch.set_num_threads(a.threads)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    report={"status":"running","protocol":"lb-h10-progressive-v1","torch":torch.__version__,
            "threads":a.threads,"max_rank":a.max_rank,"projections":[],
            "scope":"offline progressive primary/residual QAT on reused H3 fit/development; no confirmation or deployment"}
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
