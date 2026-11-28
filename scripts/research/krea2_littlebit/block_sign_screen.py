"""LB-H8.1: blockwise sign refinement from verified packed H7 control."""
import argparse
import json
import math
import time
from pathlib import Path
import torch
from safetensors import safe_open
from fit_queue import atomic_json
import native_codec as codec
import binary_rank_screen as base
import qat_core as q
import binary_qat_screen as e
from binary_qat_coverage import KEYS,validate_bank,read_panel,score
import latent_scale_core as ls
import outer_scale_core as oscale
import discrete_rotation_core as discrete
import block_sign_core as block_sign


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
    row["fit_count"]=len(fit)
    row["fit_rows"]=len(x)
    obj=timed("task_objective",lambda:discrete.Objective(wf,x,g))
    prior=json.loads(args.h89_report.read_text(encoding="utf-8"))
    oldrow=next(p for p in prior["projections"] if p["key"]==key)
    prev=oldrow["candidates"]["H7-control"]
    ref=prev["archive"]
    archive_path=args.h89_report.parent/ref["file"]
    if codec.sha256(archive_path)!=ref["sha256"]: raise ValueError("H7 archive hash mismatch")
    if prev["ranks"]!=ranks: raise ValueError("H7 rank mismatch")
    def load_archive():
        with safe_open(str(archive_path),framework="pt",device="cpu") as packed:
            meta=packed.metadata()
            if (meta.get("key")!=key or json.loads(meta["shape"])!=[m,n] or json.loads(meta["ranks"])!=ranks
                or meta.get("protocol")!="lb-h89-batched-v1"):
                raise ValueError("H7 archive metadata mismatch")
            expected={f"{i}.{name}" for i in range(len(ranks)) for name in ("a","b","row","col","latent")}
            if set(packed.keys())!=expected: raise ValueError("unexpected archive fields")
            tensors={k:packed.get_tensor(k) for k in expected}
        nb=sum(t.nbytes for t in tensors.values())
        cs=[]
        for i,r in enumerate(ranks):
            a,b,row,col,latent=[tensors[f"{i}.{name}"] for name in ("a","b","row","col","latent")]
            if a.dtype!=torch.uint8 or b.dtype!=torch.uint8 or a.numel()!=(r*n+7)//8 or b.numel()!=(m*r+7)//8:
                raise ValueError("invalid packed signs")
            for t,shape in zip((row,col,latent),((m,),(n,),(r,))):
                if t.dtype!=torch.bfloat16 or tuple(t.shape)!=shape or not torch.isfinite(t).all():
                    raise ValueError("invalid stored scales")
            cs.append(((2*codec._unpack(a,r*n,1).reshape(r,n)-1).bfloat16(),
                       (2*codec._unpack(b,m*r,1).reshape(m,r)-1).bfloat16(),row,col,latent))
        if nb!=ref["payload_bytes"] or nb!=prev["payload_bytes"]: raise ValueError("H7 payload mismatch")
        return cs,nb
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
        save_file(tensors,str(tmp),metadata={"key":key,"ranks":json.dumps(ranks),"shape":json.dumps([m,n]),"protocol":"lb-h81-block-v1"})
        restored=load_file(str(tmp))
        if not all(torch.equal(t,restored[k]) for k,t in tensors.items()): raise ValueError("archive roundtrip mismatch")
        tmp.replace(path)
        entry["archive"]={"file":path.name,"sha256":codec.sha256(path),"payload_bytes":nb,"file_bytes":path.stat().st_size}
        save()
    shared,nb=timed("load_H7_archive",load_archive)
    control=start_arm("H7-control",shared,nb)
    timed("H7-control_eval",lambda:evaluate("H7-control",control,shared,"loaded_H7"))
    expected=prev["history"][-1]
    actual=control["history"][-1]
    delta={k:actual["development"][k]-expected["development"][k] for k in ("objective","raw_forward_rms","raw_dx_rms")}
    fit_delta=actual["pooled_fit_fp64"]["objective"]-expected["pooled_fit_fp64"]["objective"]
    row["H7_reference_parity"]={"archive":str(archive_path),"sha256":ref["sha256"],"development_delta":delta,
                                "fit_delta":fit_delta,"within_tolerance":max([abs(v) for v in delta.values()]+[abs(fit_delta)])<=1e-5}
    if not row["H7_reference_parity"]["within_tolerance"]: raise ValueError("H7 archive score mismatch")
    row["historical_H8_B"]=oldrow["candidates"]["H8-B"]["history"]
    save()
    for block in (64,16):
        name=f"H81-block{block}"
        cs=clone(shared)
        entry=start_arm(name,cs,nb)
        entry["block"]=block
        entry["history"].append(dict(actual,stage="shared_H7"))
        for sweep in range(1,args.sign_sweeps+1):
            label=f"sign{sweep}"
            before=fit_metrics(cs)
            proposal,info=timed(name+"_"+label,lambda:block_sign.block_sweep(cs,obj,block,seed=41+sweep))
            after=fit_metrics(proposal)
            if after["objective"]>before["objective"]+1e-12:
                raise ValueError("direct FP64 sign sweep objective increased")
            cs=proposal
            entry["substeps"].append({"stage":label,"solver":info,"before":before,"proposed":after,"accepted":True})
            print(name,label,"accepted flips",info["flips"],"proposed",info["proposed"],"fit",after["objective"],flush=True)
            cs=scales(name,entry,cs,f"refit{sweep}")
            timed(name+"_"+label+"_eval",lambda:evaluate(name,entry,cs,label))
        timed(name+"_archive",lambda:archive(name,cs,entry))
    row["sign_sweeps"]=args.sign_sweeps
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
    ap.add_argument("--h89-report",type=Path,required=True)
    ap.add_argument("--sign-sweeps",type=int,default=2,choices=(1,2,3))
    ap.add_argument("--cg-iterations",type=int,default=40)
    ap.add_argument("--gram-mib",type=int,default=2048)
    a=ap.parse_args()
    if min(a.max_rank,a.threads,a.cg_iterations,a.gram_mib)<1 or a.out.exists(): ap.error("positive settings and fresh report required")
    torch.set_num_threads(a.threads)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    report={"status":"running","protocol":"lb-h81-block-v1","torch":torch.__version__,
            "threads":a.threads,"max_rank":a.max_rank,"sign_sweeps":a.sign_sweeps,"cg_iterations":a.cg_iterations,"gram_mib":a.gram_mib,"projections":[],
            "scope":"offline blockwise sign plus scale comparison from H7 archive on reused H3 fit/development; no QAT, confirmation or deployment"}
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
        if a.h89_report is not None:
            prior=json.loads(a.h89_report.read_text(encoding="utf-8"))
            if (prior.get("sign_sweeps")!=a.sign_sweeps or prior.get("status")!="completed" or prior.get("protocol")!="lb-h89-batched-v1"
                or prior.get("raw_sha256")!=raw_hash or prior.get("split")!=report["split"]):
                raise ValueError("H89 reference raw hash or split mismatch")
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
