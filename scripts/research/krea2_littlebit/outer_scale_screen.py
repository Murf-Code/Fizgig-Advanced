"""LB-H7: alternating conditional outer and latent scales, fixed signs."""
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
from binary_qat_coverage import KEYS,validate_bank,read_panel,score
from binary_geometry_screen import backward_metric,unwhiten,residual_target
import latent_scale_core as ls
import outer_scale_core as oscale


def run(w,key,args,stock,trace,split,int8,row,save):
    m,n=w.shape
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("highest")
    budget=m*n//8; top=min(m,n,args.max_rank)
    ranks=q.ranks(m,n,budget,top,2)
    if not ranks: raise ValueError("no feasible equal ranks")
    r1,r2=ranks
    row.update(shape=[m,n],device=str(device),solve_device="cpu",solve_dtype="float64",
               budget_bytes=budget,ranks=ranks,candidates={},confirmation_scored=False,
               objective_weights=list(ls.WEIGHTS),phase_seconds={},qat_steps=0)
    def timed(name,fn):
        print(key,name,flush=True)
        if device.type=="cuda": torch.cuda.synchronize()
        t=time.monotonic()
        value=fn()
        if device.type=="cuda": torch.cuda.synchronize()
        row["phase_seconds"][name]=time.monotonic()-t
        save()
        return value
    def load():
        fit=[read_panel(stock,key,r,trace,m,n) for r in split["fit"]]
        dev=[read_panel(stock,key,r,trace,m,n) for r in split["development"]]
        native=codec.stock_layer(w,int8)
        dr=[base.refs(w,c,native) for c in dev]
        return fit,dev,dr
    fit,dev,dr=timed("load_fit_and_dev",load)
    x=torch.cat([c["x"].double() for c in fit])
    g=torch.cat([c["dy"].double() for c in fit])
    wf=w.double()
    wg=w.to(device)
    row["fit_count"]=len(fit)
    row["fit_rows"]=len(x)
    def sketch(target,seed,rank):
        devices=[device.index or 0] if target.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            return torch.svd_lowrank(target,q=min(rank+16,m,n),niter=4)
    L=timed("backward_metric",lambda:backward_metric(w,g))
    Lg=L.to(device)
    @torch.no_grad()
    def initialize(family):
        target=w.float() if family=="ordinary" else (L.T@wf).float()
        u,s,v=sketch(target,42,top)
        a,b=old.balanced(u,s,v,r1)
        del target,u,s,v
        if family=="weighted": a,b=unwhiten(a,b,L)
        a,b=q.rotate(a.to(device),b.to(device))
        first=q.Path(a,b)
        del a,b
        target=residual_target(wg,first)
        if family=="weighted": target=(Lg.T@target.double()).float()
        u,s,v=sketch(target,43,r2)
        a,b=old.balanced(u,s,v,r2)
        del target,u,s,v
        if family=="weighted": a,b=unwhiten(a,b,Lg)
        a,b=q.rotate(a.to(device),b.to(device))
        second=q.Path(a,b)
        return e.packed(nn.ModuleList([first,second]),m,n)
    cs,nb=timed("weighted-50_initialization",lambda:initialize("weighted"))
    cs=[tuple(t.cpu() for t in path) for path in cs]
    if nb!=sum(q.size(m,n,r) for r in ranks) or nb>budget:
        raise ValueError("packed budget mismatch")
    original=cs
    entry={"ranks":ranks,"payload_bytes":nb,"effective_bits":8*nb/(m*n),
           "history":[],"substeps":[]}
    row["candidates"]["weighted-50"]=entry
    def fit_metrics(candidate):
        A,B,c=ls.atoms(candidate)
        return ls.metrics(A,B,c,wf,x,g)
    current=fit_metrics(cs)
    def evaluate(stage):
        record={"stage":stage,"development":score(w,dev,dr,cs),
                "pooled_fit_fp64":current,"weight_error":e.error(w,cs)}
        entry["history"].append(record)
        print(key,stage,"dev",record["development"]["objective"],
              "dx",record["development"]["raw_dx_rms"],"fit",current["objective"],flush=True)
        save()
    def step(label,axis):
        nonlocal cs,current
        if axis==4:
            A,B,c0=ls.atoms(cs)
            H,rhs=ls.system(A,B,wf,x,g)
            c,info=ls.solve(H,rhs,c0)
            proposal=ls.rounded_components(cs,c)
        else:
            v,rhs,diag,op,estimate=oscale.outer_system(cs,wf,x,g,axis,args.gram_mib*1024**2)
            delta,info=oscale.pcg(rhs,diag,op,maxiter=args.cg_iterations)
            info["gram_bytes_bound"]=estimate
            del op
            proposal=oscale.rounded_outer(cs,v+delta,axis)
        measured=fit_metrics(proposal)
        accepted=measured["objective"]<=current["objective"]+1e-12
        record={"stage":label,"axis":axis,"solver":info,
                "before":current,"proposed":measured,"accepted":accepted}
        if accepted: cs,current=proposal,measured
        entry["substeps"].append(record)
        print(key,label,"accepted",accepted,"fit",current["objective"],flush=True)
        save()
    timed("initial_evaluation",lambda:evaluate("initial"))
    timed("latent_only",lambda:step("latent_only",4))
    timed("latent_evaluation",lambda:evaluate("latent_only"))
    for i in range(1,args.sweeps+1):
        for axis,name in ((2,"row"),(3,"column"),(4,"latent")):
            label=f"sweep{i}_{name}"
            timed(label,lambda:step(label,axis))
        timed(f"sweep{i}_evaluation",lambda:evaluate(f"sweep{i}"))
    if not all(torch.equal(p[j],v[j]) for p,v in zip(original,cs) for j in (0,1)):
        raise ValueError("binary signs changed")
    entry["binary_sign_planes_changed"]=False
    entry["final_scales"]=[{"row":p[2].tolist(),"column":p[3].tolist(),"latent":p[4].tolist()} for p in cs]
    entry["completed_sweeps"]=args.sweeps
    save()
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
    ap.add_argument("--sweeps",type=int,default=2,choices=(1,2,3))
    ap.add_argument("--cg-iterations",type=int,default=40)
    ap.add_argument("--gram-mib",type=int,default=2048)
    a=ap.parse_args()
    if min(a.max_rank,a.threads,a.cg_iterations,a.gram_mib)<1 or a.out.exists(): ap.error("positive settings and fresh report required")
    torch.set_num_threads(a.threads)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    report={"status":"running","protocol":"lb-h7-alternating-scales-v1","torch":torch.__version__,
            "threads":a.threads,"max_rank":a.max_rank,"sweeps":a.sweeps,"cg_iterations":a.cg_iterations,"gram_mib":a.gram_mib,"projections":[],
            "scope":"offline fixed-sign alternating scale solve on reused H3 fit/development; no QAT, confirmation or deployment"}
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
