"""LB-H5: same-budget binary rank allocation and task-aware initialization."""
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
    m,n=w.shape
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("highest")
    budget=m*n//8; top=min(m,n,args.max_rank)
    ranks=allocations(m,n,budget,top)
    native=codec.stock_layer(w,int8)
    panels=[read_panel(stock,key,r,trace,m,n) for r in split["fit"]]
    dev=[read_panel(stock,key,r,trace,m,n) for r in split["development"]]
    dr=[base.refs(w,c,native) for c in dev]
    fr=[base.refs(w,c,native) for c in panels]
    wg=w.to(device)
    fit=pooled_fit(wg,panels)
    # Original H3 ordinary primary SVD sketch exactly as before.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        u,s,v=torch.svd_lowrank(w.float(),q=min(top+16,m,n),niter=4)
    L=backward_metric(w,fit[0][1].cpu())
    # Fixed original metric for BOTH primary and residual, not residual-normalized.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        uw,sw,vw=torch.svd_lowrank((L.T@w.double()).float(),q=min(top+16,m,n),niter=4)
    Lg=L.to(device)
    row.update(shape=[m,n],device=str(device),budget_bytes=budget,
               allocations={str(k):list(v) for k,v in ranks.items()},
               candidates={},qat={},fit_count=len(panels),confirmation_scored=False,
               objective_weights=list(WEIGHTS),sign_lr=.00005,scale_lr=h.SCALE_LR,
               warmup=20,max_steps=STEPS,selection_rule="control plus two best noncontrol initial dev objectives")
    def init(a,b):
        with torch.no_grad():
            a,b=q.rotate(a.to(device),b.to(device))
            return q.Path(a,b)
    def diagnostic(cs):
        return {"development":score(w,dev,dr,cs),"weight_error":e.error(w,cs)}
    # Keep only CPU initial path state. GPU live memory bounded to one arm.
    initial={}
    for family in ("ordinary","weighted"):
        for share,(r1,r2) in ranks.items():
            name=f"{family}-{share}"
            a,b=old.balanced(*( (u,s,v) if family=="ordinary" else (uw,sw,vw)),r1)
            if family=="weighted": a,b=unwhiten(a,b,L)
            pre=diagnostic([continuous_components(a,b)])
            first=init(a,b)
            del a,b
            c1,nb1=e.packed([first],m,n)
            binary_primary=diagnostic(c1)
            target=residual_target(wg,first)
            if family=="weighted":
                target=(Lg.T@target.double()).float()
            with torch.random.fork_rng(devices=[device.index or 0] if device.type=="cuda" else []):
                torch.manual_seed(43)
                ur,sr,vr=torch.svd_lowrank(target,q=min(r2+16,m,n),niter=4)
            a,b=old.balanced(ur,sr,vr,r2)
            del target,ur,sr,vr
            if family=="weighted": a,b=unwhiten(a,b,Lg)
            pre_residual=diagnostic(c1+[continuous_components(a,b)])
            second=init(a,b)
            del a,b
            paths=nn.ModuleList([first,second])
            cs,nb=e.packed(paths,m,n)
            if nb!=q.size(m,n,r1)+q.size(m,n,r2) or nb>budget:
                raise ValueError("binary packed budget mismatch")
            row["candidates"][name]={"family":family,"ranks":[r1,r2],
                "payload_bytes":nb,"effective_bits":8*nb/(m*n),
                "continuous_primary":pre,"binary_primary":binary_primary,
                "binary_primary_continuous_residual":pre_residual,
                "binary_initial":diagnostic(cs)}
            initial[name]=paths.cpu()
            print(key,name,"initial dev",row["candidates"][name]["binary_initial"]["development"]["objective"],flush=True)
            del first,second,paths,cs,c1
            save()
    del u,s,v,uw,sw,vw,L,Lg
    control="ordinary-50"
    others=sorted((k for k in initial if k!=control),
                  key=lambda k:(row["candidates"][k]["binary_initial"]["development"]["objective"],k))
    finalists=[control,*others[:2]]
    row["qat_finalists"]=finalists
    save()
    initial={k:initial[k] for k in finalists}
    for name in finalists:
        paths=initial.pop(name).to(device)
        history=[]
        best=None
        row["active_arm"]={"name":name,"history":history}
        def checkpoint(ps,stats):
            nonlocal best
            cs,nb=e.packed(ps,m,n)
            if nb>budget: raise ValueError("budget exceeded")
            ds=score(w,dev,dr,cs)
            fs=score(w,panels,fr,cs)
            err=e.error(w,cs)
            entry={**stats,"development":ds,"fixed_fit":fs,"weight_error":err}
            history.append(entry)
            if best is None or ds["objective"]<best["development"]["objective"]:
                best=entry
            print(key,name,"QAT",stats["step"],"dev",ds["objective"],"dx",ds["raw_dx_rms"],flush=True)
            save()
        checkpoint(paths,{"step":0})
        h.optimize(paths,wg,fit,.00005,warmup=20,steps=STEPS,callback=checkpoint,weights=WEIGHTS)
        row["qat"][name]={"selected_step":best["step"],"development":best["development"],
                           "fixed_fit":best["fixed_fit"],"weight_error":best["weight_error"],
                           "history":history,"payload_bytes":row["candidates"][name]["payload_bytes"],
                           "ranks":row["candidates"][name]["ranks"]}
        del row["active_arm"],paths
        save()
    row["development_winner"]=min(row["qat"],key=lambda k:row["qat"][k]["development"]["objective"])
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
    report={"status":"running","protocol":"lb-h5-binary-geometry-v1","torch":torch.__version__,
            "threads":a.threads,"max_rank":a.max_rank,"projections":[],
            "scope":"offline binary initialization screen and matched QAT on reused H3 fit/development; no confirmation or deployment"}
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
