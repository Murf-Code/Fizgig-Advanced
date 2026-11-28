"""LB-H4 rank-matched continuous references, not one-bit deployment candidates."""
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
import private_correction as pc
import mixed_factor_screen as mix
from binary_qat_coverage import KEYS,validate_bank,read_panel,pooled_fit

WEIGHTS=(.2,.4,.4)
STEPS=100
CHECKPOINTS=(10,20,50,100)
LR=.003


@torch.no_grad()
def truncated(w,rank):
    # Full FP64 economy SVD: no randomized sketch.
    u,s,vh=torch.linalg.svd(w.double(),full_matrices=False)
    root=s[:rank].sqrt()
    a=root[:,None]*vh[:rank]
    b=u[:,:rank]*root
    tail=float(s[rank:].square().sum()/s.square().sum())
    return a.contiguous(),b.contiguous(),tail


@torch.no_grad()
def backward_reference(w,g,rank,ww=.2,bw=.4):
    """Exact (up to numerical precision) rank-r minimizer of
    ww*||E||_F^2/||W||_F^2 + bw*||G E||_F^2/||G W||_F^2.
    Row convention W is m x n, G is samples x m.
    H=L L^T; transformed target L^T W; undo L^T after truncation.
    """
    w=w.double(); g=g.double()
    we=w.square().sum()
    ge=(g@w).square().sum()
    if not all(torch.isfinite(v) and v>0 for v in (we,ge)) or min(ww,bw)<=0:
        raise ValueError("positive finite energies and weights required")
    alpha=ww/we; beta=bw/ge
    H=torch.eye(w.shape[0],device=w.device,dtype=w.dtype)+(beta/alpha)*(g.T@g)
    L=torch.linalg.cholesky(H)
    u,s,vh=torch.linalg.svd(L.T@w,full_matrices=False)
    root=s[:rank].sqrt()
    a=root[:,None]*vh[:rank]
    b=torch.linalg.solve_triangular(L.T,u[:,:rank]*root,upper=True)
    lower=float(alpha*s[rank:].square().sum())
    return a.contiguous(),b.contiguous(),lower


def pooled_loss(a,b,w,x,g,y,dx):
    err=b@a-w
    return (.2*err.square().sum()/w.square().sum()
            +.4*((x@a.T)@b.T-y).square().sum()/y.square().sum()
            +.4*((g@b)@a-dx).square().sum()/dx.square().sum())


def panel_score(op,panels,refs):
    rows=[base.score(op,c,r) for c,r in zip(panels,refs)]
    if not rows or any(v[k] is None or not math.isfinite(v[k])
                       for v in rows for k in ("raw_forward","raw_dx")):
        raise ValueError("invalid score")
    return {"rows":rows,"objective":base.objective(rows),
            **{k+"_rms":math.sqrt(sum(v[k]**2 for v in rows)/len(rows))
               for k in ("raw_forward","raw_dx")}}


def run(w,key,args,stock,trace,split,int8,row,save):
    m,n=w.shape
    torch.set_float32_matmul_precision("highest")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rr=q.ranks(m,n,m*n//8,min(m,n,args.max_rank),2)
    if not rr: raise ValueError("no binary rank allocation")
    rank=sum(rr)
    native=codec.stock_layer(w,int8)
    fit=[read_panel(stock,key,r,trace,m,n) for r in split["fit"]]
    dev=[read_panel(stock,key,r,trace,m,n) for r in split["development"]]
    dr=[base.refs(w,c,native) for c in dev]
    row.update(shape=[m,n],device=str(device),rank=rank,binary_ranks=rr,
               binary_payload_bytes=sum(q.size(m,n,r) for r in rr),
               candidates={},weights=list(WEIGHTS),confirmation_scored=False,
               scope="reused H3 fit/development; no confirmation tensor evaluation",
               continuous_steps=STEPS,relative_factor_lr=LR)
    def record(name,op,nb,err,**extra):
        ds=panel_score(op,dev,dr)
        row["candidates"][name]={"payload_bytes":nb,"effective_bits":8*nb/(m*n),
                                "weight_error":err,"development":ds,**extra}
        print(key,name,"dev",ds["objective"],"dx",ds["raw_dx_rms"],flush=True)
        save()
    for bits in (2,3):
        op,nb,err=mix.direct_lowbit(w,bits)
        record(f"direct-q{bits}-odd-had",op,nb,err,kind="direct-quantization")
        del op
    for name in ("q4-had","q6-had"):
        p,s=codec.encode(w,name)
        record(name,lambda x,g:pc.operator(p,s,x,g),p.nbytes,pc.weight_error(w,p,s),
               kind="direct-quantization")
        del p,s
    # Native INT8 control, measured relative to the same raw BF16 reference.
    def native_op(x,g):
        z=x.detach().bfloat16().requires_grad_(True)
        y=native(z)
        dx,=torch.autograd.grad(y,z,g.bfloat16())
        return y.detach(),dx.detach()
    record("stock-int8",native_op,m*n+4*m,None,kind="native-control")
    wf=w.to(device,dtype=torch.float32)
    x,g,y,dx=pooled_fit(wf,fit)[0]
    def continuous_metrics(a,b):
        # FP32 pooled fit objective plus BF16 factor execution on dev.
        ac,bc=a.detach().cpu(),b.detach().cpu()
        af,bf=ac.bfloat16(),bc.bfloat16()
        return (af,bf,old.weight_error(w,af,bf),
                float(pooled_loss(a.to(device,dtype=torch.float32),b.to(device,dtype=torch.float32),wf,x,g,y,dx)))
    def continuous_record(name,a,b,**extra):
        af,bf,err,loss=continuous_metrics(a,b)
        record(name,lambda x,g:old.actions(x,g,af,bf),af.nbytes+bf.nbytes,err,
               kind="continuous-rank-diagnostic",rank=rank,fit_pooled_fp32=loss,**extra)
    print(key,"full FP64 weight SVD",rank,flush=True)
    a0,b0,tail=truncated(w,rank)
    row["optimal_weight_relative_error_fp64"]=math.sqrt(tail)
    row["joint_fit_lower_bound_weight_only"]=WEIGHTS[0]*tail
    continuous_record("rank-svd",a0,b0,selected_step=0)
    print(key,"full FP64 backward-weighted SVD",rank,flush=True)
    a1,b1,lower=backward_reference(w,g.cpu(),rank)
    row["joint_fit_lower_bound_weight_backward"]=lower
    continuous_record("rank-backward-weighted",a1,b1,selected_step=0)
    # Fix initial factors by lower fit-only joint objective, not dev score.
    initial=min(((a0,b0),(a1,b1)),key=lambda ab:float(pooled_loss(
        ab[0].to(device,dtype=torch.float32),ab[1].to(device,dtype=torch.float32),wf,x,g,y,dx)))
    row["joint_initialization"]="rank-svd" if initial[0] is a0 else "rank-backward-weighted"
    # Normalize parameter RMS so LR is relative to the factor scale.
    a,b=(t.to(device,dtype=torch.float32) for t in initial)
    sa=a.square().mean().sqrt(); sb=b.square().mean().sqrt()
    if min(float(sa),float(sb))<=0: raise ValueError("zero factor RMS")
    aa=nn.Parameter(a/sa); bb=nn.Parameter(b/sb)
    opt=torch.optim.Adam([aa,bb],lr=LR)
    history=[]; best=None
    def checkpoint(step):
        nonlocal best
        with torch.no_grad():
            a=(aa*sa).detach(); b=(bb*sb).detach()
            af,bf,err,loss=continuous_metrics(a,b)
        ds=panel_score(lambda x,g:old.actions(x,g,af,bf),dev,dr)
        history.append({"step":step,"fit_pooled_fp32":loss,"development":ds,"weight_error":err})
        if best is None or ds["objective"]<best[0]:
            best=(ds["objective"],af,bf,err,loss,step)
        row["active_arm"]={"name":"rank-joint","history":history}
        print(key,"rank-joint",step,"of",STEPS,"fit",loss,"dev",ds["objective"],flush=True)
        save()
    checkpoint(0)
    for step in range(1,STEPS+1):
        opt.zero_grad(set_to_none=True)
        loss=pooled_loss(aa*sa,bb*sb,wf,x,g,y,dx)
        if not torch.isfinite(loss): raise ValueError("nonfinite continuous objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([aa,bb],1.,error_if_nonfinite=True)
        opt.step()
        if step in CHECKPOINTS: checkpoint(step)
    _,af,bf,err,loss,step=best
    record("rank-joint",lambda x,g:old.actions(x,g,af,bf),af.nbytes+bf.nbytes,err,
           kind="continuous-rank-diagnostic",rank=rank,selected_step=step,
           fit_pooled_fp32=loss,history=history)
    del row["active_arm"]
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
    report={"status":"running","protocol":"lb-h4-rank-reference-v1","torch":torch.__version__,
            "threads":a.threads,"max_rank":a.max_rank,"projections":[],
            "scope":"offline rank diagnostics on reused H3 fit/development only; no confirmation or deployment"}
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
