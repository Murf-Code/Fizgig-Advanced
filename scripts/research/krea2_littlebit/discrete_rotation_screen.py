"""LB-H8/H9: matched five-arm batched sign and rotation comparison."""
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
import discrete_rotation_core as discrete


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
    obj=timed("task_objective",lambda:discrete.Objective(wf,x,g))
    @torch.no_grad()
    def initialize(mode):
        target=(L.T@wf).float()
        u,s,v=sketch(target,42,top)
        a,b=old.balanced(u,s,v,r1)
        del target,u,s,v
        a,b=unwhiten(a,b,L)
        a,b=q.rotate(a.to(device),b.to(device))
        first=q.Path(a,b)
        first_cs,nb=e.packed([first],m,n)
        stats=[]
        if mode!="baseline":
            rotobj=obj if mode=="task" else discrete.Objective(wf,x,g,mode="weight")
            first_cs[0],info=discrete.rotate_path(first_cs[0],a,b,rotobj,wf,seed=1000)
            stats.append(info)
        del a,b
        if mode=="baseline":
            target=residual_target(wg,first)
        else:
            target=wg.float().clone()
            c=tuple(t.to(device,dtype=torch.float32) for t in first_cs[0])
            for k in range(0,m,256):
                idx=slice(k,min(k+256,m))
                target[idx]-=q.weight(c,idx)
            del c
        del first
        teacher=target.cpu().double() if mode!="baseline" else None
        target=(Lg.T@target.double()).float()
        u,s,v=sketch(target,43,r2)
        a,b=old.balanced(u,s,v,r2)
        del target,u,s,v
        a,b=unwhiten(a,b,Lg)
        a,b=q.rotate(a.to(device),b.to(device))
        second=q.Path(a,b)
        second_cs,nb2=e.packed([second],m,n)
        if mode!="baseline":
            second_cs[0],info=discrete.rotate_path(second_cs[0],a,b,rotobj,teacher,seed=1100)
            stats.append(info)
        return first_cs+second_cs,nb+nb2,stats
    def fit_metrics(candidate):
        A,B,c=ls.atoms(candidate)
        return ls.metrics(A,B,c,wf,x,g)
    def clone(cs):
        return [tuple(t.clone() for t in p) for p in cs]
    def start_arm(name,cs,nb,rotation=None):
        if nb!=sum(q.size(m,n,r) for r in ranks) or nb>budget:
            raise ValueError("packed budget mismatch")
        entry={"ranks":ranks,"payload_bytes":nb,"effective_bits":8*nb/(m*n),
               "history":[],"substeps":[],"rotation":rotation or []}
        row["candidates"][name]=entry
        return entry
    def evaluate(name,entry,cs,stage):
        record={"stage":stage,"development":score(w,dev,dr,cs),
                "pooled_fit_fp64":fit_metrics(cs),"weight_error":e.error(w,cs)}
        entry["history"].append(record)
        print(key,name,stage,"dev",record["development"]["objective"],
              "dx",record["development"]["raw_dx_rms"],"fit",record["pooled_fit_fp64"]["objective"],flush=True)
        save()
    def scale_step(name,entry,cs,axis,label):
        before=fit_metrics(cs)
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
        after=fit_metrics(proposal)
        accepted=after["objective"]<=before["objective"]+1e-12
        entry["substeps"].append({"stage":label,"axis":axis,"solver":info,"before":before,"proposed":after,"accepted":accepted})
        save()
        return proposal if accepted else cs
    def scales(name,entry,cs,label):
        for axis,tag in ((2,"row"),(3,"column"),(4,"latent")):
            stage=label+"_"+tag
            cs=timed(name+"_"+stage,lambda:scale_step(name,entry,cs,axis,stage))
        return cs
    def warmup(name,entry,cs):
        timed(name+"_initial_eval",lambda:evaluate(name,entry,cs,"initial"))
        cs=timed(name+"_latent_only",lambda:scale_step(name,entry,cs,4,"latent_only"))
        for i in range(1,3):
            cs=scales(name,entry,cs,f"warmup{i}")
        timed(name+"_warmup_eval",lambda:evaluate(name,entry,cs,"warmup2"))
        return cs
    def archive(name,cs,entry):
        from safetensors.torch import save_file,load_file
        tensors={}
        for i,(a,b,r,s,c) in enumerate(cs):
            tensors.update({f"{i}.a":codec._pack(a>0,1),f"{i}.b":codec._pack(b>0,1),
                            f"{i}.row":r.contiguous(),f"{i}.col":s.contiguous(),f"{i}.latent":c.contiguous()})
        nb=sum(t.numel()*t.element_size() for t in tensors.values())
        if nb!=entry["payload_bytes"]: raise ValueError("archive payload mismatch")
        path=args.out.with_name(args.out.stem+"-"+name+".safetensors")
        if path.exists(): raise ValueError("archive already exists: "+str(path))
        tmp=path.with_suffix(".tmp")
        save_file(tensors,str(tmp),metadata={"key":key,"ranks":json.dumps(ranks),"shape":json.dumps([m,n]),"protocol":"lb-h89-batched-v1"})
        restored=load_file(str(tmp))
        if not all(torch.equal(t,restored[k]) for k,t in tensors.items()): raise ValueError("archive roundtrip mismatch")
        tmp.replace(path)
        entry["archive"]={"file":path.name,"sha256":codec.sha256(path),"payload_bytes":nb,"file_bytes":path.stat().st_size}
        save()
    cs,nb,_=timed("baseline_initialization",lambda:initialize("baseline"))
    control=start_arm("H7-control",cs,nb)
    shared=warmup("H7-control",control,cs)
    timed("H7-control_archive",lambda:archive("H7-control",shared,control))
    for name,mode,refit in (("H8-A","baseline",False),("H8-B","baseline",True),
                            ("H9-A","weight",True),("H9-B","task",True)):
        if mode=="baseline":
            cs=clone(shared)
            entry=start_arm(name,cs,nb)
            entry["history"].append(dict(control["history"][-1],stage="shared_H7"))
        else:
            cs,nb2,rot=timed(name+"_initialization",lambda:initialize(mode))
            entry=start_arm(name,cs,nb2,rot)
            cs=warmup(name,entry,cs)
        for sweep in range(1,args.sign_sweeps+1):
            label=f"sign{sweep}"
            before=fit_metrics(cs)
            proposal,info=timed(name+"_"+label,lambda:discrete.sign_sweep(cs,obj))
            after=fit_metrics(proposal)
            if after["objective"]>before["objective"]+1e-12:
                raise ValueError("direct FP64 sign sweep objective increased")
            cs=proposal
            entry["substeps"].append({"stage":label,"solver":info,"before":before,"proposed":after,"accepted":True})
            if refit: cs=scales(name,entry,cs,f"refit{sweep}")
            timed(name+"_"+label+"_eval",lambda:evaluate(name,entry,cs,label))
        timed(name+"_archive",lambda:archive(name,cs,entry))
    row["sign_sweeps"]=args.sign_sweeps
    row["rotation_policy"]="one disjoint Givens pass per path, six angles, pair-local proposals with exact joint guard"
    row["development_winner"]=min(row["candidates"],key=lambda k:row["candidates"][k]["history"][-1]["development"]["objective"])
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
    ap.add_argument("--sign-sweeps",type=int,default=2,choices=(1,2,3))
    ap.add_argument("--cg-iterations",type=int,default=40)
    ap.add_argument("--gram-mib",type=int,default=2048)
    a=ap.parse_args()
    if min(a.max_rank,a.threads,a.cg_iterations,a.gram_mib)<1 or a.out.exists(): ap.error("positive settings and fresh report required")
    torch.set_num_threads(a.threads)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    report={"status":"running","protocol":"lb-h89-batched-v1","torch":torch.__version__,
            "threads":a.threads,"max_rank":a.max_rank,"sign_sweeps":a.sign_sweeps,"cg_iterations":a.cg_iterations,"gram_mib":a.gram_mib,"projections":[],
            "scope":"offline matched sign and rotation comparison on reused H3 fit/development; no QAT, confirmation or deployment"}
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
