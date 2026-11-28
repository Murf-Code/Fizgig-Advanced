"""LB-H3: offline pure-binary coverage ablation on the locked 80-image bank."""
import copy
import json
import math
from pathlib import Path
import torch
from torch import nn
import native_codec as codec
import binary_rank_screen as base
import binary_rank_core as old
import qat_core as q
import binary_qat_screen as e
import binary_qat_objective as h
from calibration_bank import KEYS, COUNT, FIT_SIZES, partition

ARMS=(("control",(.2,.4,.4)),("weight50",(.5,.25,.25)))
STEPS=200
RTOL=1e-4


def validate_bank(stock, trace):
    bank=json.loads((stock/"capture-bank.json").read_text(encoding="utf-8"))
    if (trace.get("protocol")!="lb-h3-stock-bank-v1" or bank.get("protocol")!=trace["protocol"]
        or bank.get("status")!="completed" or trace.get("completed_updates")!=len(trace["steps"])
        or bank.get("keys")!=list(KEYS) or trace.get("capture_keys")!=list(KEYS)
        or bank.get("fit_sizes")!=list(FIT_SIZES)):
        raise ValueError("completed H3 capture bank required")
    for field in ("steps","calibration","split_steps"):
        if bank[field]!=trace[field]: raise ValueError("bank/trace mismatch: "+field)
    records=trace["calibration"]
    steps={s["step"]:s for s in trace["steps"]}
    if len(steps)!=len(trace["steps"]) or len(records)!=COUNT:
        raise ValueError("wrong capture count or duplicate steps")
    locked=partition(records,trace["steps"])
    if locked!=trace["split_steps"]: raise ValueError("locked partition mismatch")
    by_step={r["step"]:r for r in records}
    if len(by_step)!=COUNT: raise ValueError("duplicate capture steps")
    hashes=set()
    for r in records:
        path=stock/r["file"]
        if path.resolve().parent!=stock.resolve(): raise ValueError("invalid capture path")
        digest=codec.sha256(path)
        if digest!=r["sha256"] or digest in hashes: raise ValueError("capture hash mismatch or duplicate")
        hashes.add(digest)
        step=steps[r["step"]]; tok=r["tokens"]
        if len(step["items"])!=1 or not step["items"][0] or not 0<=step["timestep"]<=1:
            raise ValueError("invalid capture identity/timestep")
        ni,nt=tok["sampled_image"],tok["sampled_text"]
        idx=tok["indices"]
        if (not 0<ni+nt<=32 or len(idx)!=ni+nt or len(set(idx))!=len(idx)
            or ni>tok["image_tokens"] or nt>tok["valid_text_tokens"]
            or any(not 0<=j<tok["image_tokens"] for j in idx[:ni])
            or any(not tok["image_tokens"]<=j<tok["sequence_tokens"] for j in idx[ni:])):
            raise ValueError("invalid sampled token layout")
    return {name:[by_step[i] for i in ids] for name,ids in locked.items()}


def read_panel(stock,key,r,trace,m,n):
    c=base.read_capture(stock,key,r,trace)
    if c["x"].shape!=(len(c["x"]),n) or c["dy"].shape!=(len(c["x"]),m):
        raise ValueError("capture shape mismatch")
    if not c["x"].is_floating_point() or not c["dy"].is_floating_point():
        raise ValueError("capture dtype mismatch")
    return c


def geometry(fit,dev):
    """Row Gram eigensystem in FP64 CPU; no feature-square or weight matrix."""
    out={}
    for name in ("x","dy"):
        x=torch.cat([c[name].double() for c in fit])
        z=torch.cat([c[name].double() for c in dev])
        gram=x@x.T
        vals,u=torch.linalg.eigh(gram)
        vals=vals.clamp_min(0).flip(0)
        u=u.flip(1)
        energy=vals.sum()
        if not torch.isfinite(energy) or energy<=0 or z.square().sum()<=0:
            raise ValueError("zero/nonfinite coverage energy")
        p=vals/energy
        nz=p[p>0]
        keep=vals>vals[0]*RTOL**2
        # Orthonormal right singular vectors via the row Gram eigenvectors.
        basis=(u[:,keep].T@x)/vals[keep].sqrt()[:,None]
        projected=(z@basis.T).square().sum()/z.square().sum()
        if not torch.isfinite(projected) or float(projected)>1.00001:
            raise ValueError("unstable span computation")
        cum=p.cumsum(0)
        def energy_rank(f): return int(torch.searchsorted(cum,f))+1
        out[name]={"rows":len(x),"features":x.shape[1],
                   "numerical_rank":int(keep.sum()),"relative_singular_threshold":RTOL,
                   "entropy_effective_rank":float((-nz*nz.log()).sum().exp()),
                   "participation_rank":float(1/p.square().sum()),
                   "energy_rank95":energy_rank(.95),"energy_rank99":energy_rank(.99),
                   "development_energy_in_fit_span":min(1.,float(projected)),
                   "singular_values":vals.sqrt().tolist()}
    return out


def pooled_fit(wg,panels):
    # Concatenation preserves the H2 pooled objective in exact arithmetic.
    # One graph per update, not one graph per image. FP32 reduction order differs.
    with torch.no_grad():
        x=torch.cat([c["x"].float() for c in panels]).to(wg.device)
        g=torch.cat([c["dy"].float() for c in panels]).to(wg.device)
        wf=wg.float()
        return [(x,g,x@wf.T,g@wf)]


def score(w,panels,refs,cs):
    values=[base.score(lambda x,g:e.packed_actions(cs,x,g),c,r)
            for c,r in zip(panels,refs)]
    if not values or any(v[k] is None or not math.isfinite(v[k])
                         for v in values for k in ("raw_forward","raw_dx")):
        raise ValueError("invalid evaluation action energy")
    return {"rows":values,"objective":base.objective(values),
            **{k+"_rms":math.sqrt(sum(v[k]**2 for v in values)/len(values))
               for k in ("raw_forward","raw_dx")}}


def run(w,key,args,stock,trace,split,int8,row,save):
    m,n=w.shape
    if key not in KEYS: raise ValueError("projection not captured")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("highest")
    native=codec.stock_layer(w,int8)
    panels={k:[read_panel(stock,key,r,trace,m,n) for r in split[k]]
            for k in ("fit","development")}
    dev=panels["development"]
    dr=[base.refs(w,c,native) for c in dev]
    fr=[base.refs(w,c,native) for c in panels["fit"]]
    wg=w.to(device)
    row.update(shape=[m,n],device=str(device),candidates={},coverage={},
               objective="same H2 pooled energies with concatenated fit rows; common balanced per-capture BF16 evaluation",
               checkpoint_steps=[0,*h.CHECKPOINTS],max_steps=STEPS,bit_cap=1.,
               fit_sizes=list(FIT_SIZES),sign_lr=.00005,scale_lr=h.SCALE_LR,warmup=20)
    for size in FIT_SIZES:
        row["coverage"][str(size)]=geometry(panels["fit"][:size],dev)
        print(key,"fit",size,"span coverage",
              {k:round(v["development_energy_in_fit_span"],4) for k,v in row["coverage"][str(size)].items()},flush=True)
        save()
    top=min(m,n,args.max_rank)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        u,s,v=torch.svd_lowrank(w.float(),q=min(top+16,m,n),niter=4)
    def init(a,b):
        with torch.no_grad():
            a,b=q.rotate(a.to(device),b.to(device))
            return q.Path(a,b)
    def residual(first,rank):
        with torch.no_grad():
            target=wg.float()
            c=first.components()
            for i in range(0,m,256):
                idx=slice(i,min(i+256,m))
                target[idx]-=q.weight(c,idx)
            del c
            with torch.random.fork_rng(devices=[device.index or 0] if device.type=="cuda" else []):
                torch.manual_seed(43)
                ur,sr,vr=torch.svd_lowrank(target,q=min(rank+16,m,n),niter=4)
            a,b=old.balanced(ur,sr,vr,rank)
            del target,ur,sr,vr
            return init(a,b)
    cap=1.
    budget=int(cap*m*n//8)
    rr=q.ranks(m,n,budget,top,2)
    if not rr: raise ValueError("no two-path allocation")
    a,b=old.balanced(u,s,v,rr[0])
    first=init(a,b); del a,b,u,s,v
    dual=nn.ModuleList([first,residual(first,rr[1])])
    selected={}
    def record(name,cs,nb,err,step,history,size=None,weights=None):
        if nb>budget: raise ValueError("packed storage budget exceeded")
        ds=score(w,dev,dr,cs)
        row["candidates"][name]={"payload_bytes":nb,"effective_bits":8*nb/(m*n),
            "ranks":rr,"weight_error":err,"selected_step":step,"fit_count":size,
            "objective_weights":weights,"history":history,"development":ds,
            "dense_equivalent_macs_per_token":sum(rr)*(m+n),
            "decoded_bf16_factor_bytes":sum(t.nbytes for c in cs for t in c)}
        selected[name]=cs
        print(key,name,"dev",ds["objective"],"dx",ds["raw_dx_rms"],flush=True)
        save()
    cs,nb=e.packed(dual,m,n)
    record("H0-E3",cs,nb,e.error(w,cs),0,[])
    del cs
    for size in FIT_SIZES:
        subset=panels["fit"][:size]
        fit=pooled_fit(wg,subset)
        for label,weights in ARMS:
            name=f"H3-{size:02d}-{label}"
            paths=copy.deepcopy(dual)
            best=None; history=[]
            row["active_arm"]={"name":name,"history":history}
            def checkpoint(ps,stats):
                nonlocal best
                cs,nb=e.packed(ps,m,n)
                ds=score(w,dev,dr,cs)
                fs=score(w,subset,fr[:size],cs)
                err=e.error(w,cs)
                history.append({**stats,"development":ds,"fixed_fit":fs,"weight_error":err})
                if best is None or ds["objective"]<best[0]:
                    best=(ds["objective"],cs,nb,err,stats["step"])
                print(key,name,"checkpoint",stats["step"],"dev",ds["objective"],
                      "dx",ds["raw_dx_rms"],"fit",fs["objective"],flush=True)
                save()
            checkpoint(paths,{"step":0})
            h.optimize(paths,wg,fit,.00005,warmup=20,steps=STEPS,
                       callback=checkpoint,weights=weights)
            _,cs,nb,err,step=best
            record(name,cs,nb,err,step,history,size,list(weights))
            del row["active_arm"]
            save()
            del paths,best,cs
        del fit
    # Freeze every checkpoint and the winner before loading confirmation tensors.
    row["development_winner"]=min(row["candidates"],
        key=lambda k:row["candidates"][k]["development"]["objective"])
    row["selection_locked"]=True
    save()
    conf=[read_panel(stock,key,r,trace,m,n) for r in split["confirmation"]]
    cr=[base.refs(w,c,native) for c in conf]
    for name,cs in selected.items():
        row["candidates"][name]["confirmation"]=score(w,conf,cr,cs)
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
    a=ap.parse_args()
    if min(a.max_rank,a.threads)<1 or a.out.exists(): ap.error("positive settings and fresh report required")
    torch.set_num_threads(a.threads)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    report={"status":"running","protocol":"lb-h3-coverage-v1","torch":torch.__version__,
            "threads":a.threads,"max_rank":a.max_rank,"projections":[],
            "scope":"offline H3 local actions; new disjoint capture IDs; not independent LoRA training"}
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
        # Fit partition is reserved for a later task-aware fitting experiment.
        report["fit_partition_used"]=True
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
