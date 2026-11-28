"""LB-H2 pure binary objective ablation. Optional separate stock capture bank."""
import argparse
import copy
import json
import math
import sys
from pathlib import Path
import torch
from torch import nn
import native_codec as codec
import binary_rank_screen as base
import binary_rank_core as old
import qat_core as q
import binary_qat_screen as e
from binary_qat_stabilization import components, SCALE_LR

CHECKPOINTS = (20, 50, 100, 150, 200)
STEPS = 200
ARMS = (
    ("H2-control", (.2,.4,.4)),
    ("H2-backward60", (.2,.2,.6)),
    ("H2-backward70", (.2,.1,.7)),
    ("H2-weight50", (.5,.25,.25)),
    ("H2-backward-only", (0.,0.,1.)),
)
FRESH_STOCK = None


def fresh_bank(stock,trace,raw_path):
    if FRESH_STOCK is None: return None
    folder=FRESH_STOCK.resolve()
    if folder==stock.resolve(): raise ValueError("fresh stock must be a different folder")
    tr=json.loads((folder/"trace.json").read_text(encoding="utf-8"))
    prep=json.loads((folder/"prepare.json").read_text(encoding="utf-8"))
    if tr.get("case")!="stock" or not tr.get("completed_updates") or prep.get("status")!="stock_verified":
        raise ValueError("fresh stock must be a completed verified stock run")
    if prep["raw_sha256"]!=codec.sha256(raw_path):
        raise ValueError("fresh stock raw mismatch")
    def items(t):
        steps={s["step"]:s for s in t["steps"]}
        return {item for r in t["calibration"] for item in steps[r["step"]]["items"]}
    if items(trace)&items(tr):
        raise ValueError("fresh bank overlaps previous captured item IDs; use a genuinely separate dataset")
    records=tr["calibration"]
    if len(items(tr))<3: raise ValueError("need at least three distinct fresh item IDs")
    old_hashes={codec.sha256(stock/r["file"]) for r in trace["calibration"]}
    hashes=[codec.sha256(folder/r["file"]) for r in records]
    if old_hashes.intersection(hashes) or len(set(hashes))!=len(hashes):
        raise ValueError("duplicate fresh capture tensors")
    return folder,tr,records,hashes


def optimize(paths, w, fit, sign_lr, warmup=0, steps=STEPS, callback=None,
             seed=42, weights=(.2,.4,.4)):
    """Same row stream across arms; pooled target energy; separate Adam groups."""
    if len(weights)!=3 or min(weights)<0 or not math.isclose(sum(weights),1.):
        raise ValueError("invalid objective weights")
    ww,fw,bw=weights
    signs = [v for p in paths for v in (p.a, p.b)]
    scales = [v for p in paths for v in (p.row, p.col, p.latent)]
    opt = torch.optim.Adam([{"params": signs, "lr": sign_lr},
                            {"params": scales, "lr": SCALE_LR}])
    gen = torch.Generator(device=w.device).manual_seed(seed)
    batch = min(256, len(w))
    with torch.no_grad():
        wenergy = sum(float(w[i:i+256].float().square().sum(dtype=torch.float64))
                      for i in range(0, len(w), 256))
        fenergy = sum(float(y.square().sum(dtype=torch.float64)) for x,g,y,dx in fit)
        genergy = sum(float(dx.square().sum(dtype=torch.float64)) for x,g,y,dx in fit)
    if not all(math.isfinite(v) and v>0 for v in (wenergy,fenergy,genergy)):
        raise ValueError("zero teacher energy")
    initial = [v.detach() >= 0 for v in signs]
    prev = [v.clone() for v in initial]
    flips = [0] * len(signs)
    for local in range(1, steps + 1):
        step = local
        lr_now = sign_lr if step>warmup else 0.
        active = lr_now > 0
        opt.param_groups[0]["lr"] = lr_now
        for v in signs:
            v.requires_grad_(active)
        idx = torch.randperm(len(w), device=w.device, generator=gen)[:batch]
        opt.zero_grad(set_to_none=True)
        cs = [components(p, "smooth" if active else "hard") for p in paths]
        loss = ww * (sum(q.weight(c, idx) for c in cs) - w[idx].float()).square().sum() * (len(w) / batch / wenergy) if ww else 0.
        for x,g,y,dx in fit:
            pairs = [q.actions(c,x,g) for c in cs]
            loss = loss + fw*(sum(t[0] for t in pairs)-y).square().sum()/fenergy
            loss = loss + bw*(sum(t[1] for t in pairs)-dx).square().sum()/genergy
        if not torch.isfinite(loss):
            raise ValueError("nonfinite local objective")
        loss.backward()
        sn = float(torch.nn.utils.clip_grad_norm_(signs, 1., error_if_nonfinite=True)) if active else 0.
        sc = float(torch.nn.utils.clip_grad_norm_(scales, 1., error_if_nonfinite=True))
        opt.step()
        with torch.no_grad():
            for j,v in enumerate(signs):
                now = v >= 0
                flips[j] += int((now != prev[j]).sum())
                prev[j] = now
            if callback and (step in CHECKPOINTS or local == steps):
                stats = {"step":step, "sign_lr_current":lr_now, "sign_active":active, "batch_objective_before_update":float(loss),
                         "sign_grad_norm":sn, "scale_grad_norm":sc,
                         "net_flip_fraction": [float((a != b).float().mean()) for a,b in zip(initial,prev)],
                         "cumulative_flips_per_element": [f/v.numel() for f,v in zip(flips,signs)],
                         "scale_multiplier_min": min(float(v.min()) for v in scales),
                         "scale_multiplier_max": max(float(v.max()) for v in scales),
                         "scale_near_zero_fraction": sum(int((v.abs()<.01).sum()) for v in scales)/sum(v.numel() for v in scales),
                         "smooth_saturation_fraction": sum(int((v.abs()>.03).sum()) for v in signs)/sum(v.numel() for v in signs)}
                with torch.enable_grad():
                    callback(paths, stats)
    for v in signs:
        v.requires_grad_(True)


def run(w,key,args,stock,trace,split,int8,archive,row,save):
    m,n = w.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("highest")
    native = codec.stock_layer(w,int8)
    dev = [base.read_capture(stock,key,r,trace) for r in split["development"]]
    conf = [base.read_capture(stock,key,r,trace) for r in split["confirmation"]]
    dr = [base.refs(w,c,native) for c in dev]
    cr = [base.refs(w,c,native) for c in conf]
    fit_panels=[base.read_capture(stock,key,r,trace) for r in split["fit"]]
    fit_refs=[base.refs(w,c,native) for c in fit_panels]
    fresh=fresh_bank(stock,trace,args.raw)
    if fresh:
        folder,tr,records,hashes=fresh
        fresh_panels=[base.read_capture(folder,key,r,tr) for r in records]
        fresh_refs=[base.refs(w,c,native) for c in fresh_panels]
    else:
        fresh_panels=[]; fresh_refs=[]
    wg = w.to(device)
    fit = []
    with torch.no_grad():
        wf = wg.float()
        for r in split["fit"]:
            c = base.read_capture(stock,key,r,trace)
            x,g = c["x"].to(device,dtype=torch.float32),c["dy"].to(device,dtype=torch.float32)
            fit.append((x,g,x@wf.T,g@wf))
        del wf
    if not fit: raise ValueError("missing fit captures")
    row.update(shape=[m,n],device=str(device),candidates={},
               scope="pure binary objective ablation; prior dev selects checkpoints; optional fresh stock confirmation only",
               objective="pooled fit energy; unbiased weight row estimate; FP32 fit, BF16 evaluation",
               checkpoint_steps=list(CHECKPOINTS),scale_lr=SCALE_LR,max_steps=STEPS,bit_cap=1.0)
    row['fresh_capture_bank']=None if fresh is None else {'folder':str(folder),'captures':[{'file':r['file'],'sha256':s,'step':r['step']} for r,s in zip(records,hashes)]}
    def score(op, panels, refs):
        values = [base.score(op,c,r) for c,r in zip(panels,refs)]
        obj = base.objective(values)
        if obj is None or not math.isfinite(obj): raise ValueError("invalid score")
        return values,obj
    def record(name,op,nb,ranks,err,cap=None,history=None,extra=None):
        if cap is not None and nb>int(cap*m*n//8): raise ValueError("budget exceeded")
        ds,dv=score(op,dev,dr)
        cs,cv=score(op,conf,cr)
        row["candidates"][name]={"payload_bytes":nb,"effective_bits":8*nb/(m*n),
            "ranks":ranks,"weight_error":err,"prior_development":ds,"prior_development_objective":dv,
            "prior_confirmation":cs,"prior_confirmation_objective":cv,
            "dense_equivalent_macs_per_token":sum(ranks)*(m+n) if ranks else m*n,
            "cap":cap,"history":history,**(extra or {})}
        fs,fv=score(op,fit_panels,fit_refs)
        row["candidates"][name].update(fixed_fit=fs,fixed_fit_objective=fv)
        print(key,name,"bits",round(8*nb/(m*n),5),"dev",dv,"prior conf",cv,flush=True)
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
    cs,nb=e.packed(dual,m,n)
    record("H0-E3",lambda x,g:e.packed_actions(cs,x,g),nb,rr,e.error(w,cs),cap)
    del cs
    # Keep selected decoded factors only if a fresh bank must be scored.
    selected={}
    if fresh: selected["H0-E3"]=e.packed(dual,m,n)[0]
    for label,weights in ARMS:
        paths=copy.deepcopy(dual)
        history=[]; best=None
        row["active_arm"]={"name":label,"history":history}
        def checkpoint(ps,stats):
            nonlocal best
            cs,nb=e.packed(ps,m,n)
            op=lambda x,g:e.packed_actions(cs,x,g)
            ds,val=score(op,dev,dr)
            fs,fv=score(op,fit_panels,fit_refs)
            err=e.error(w,cs)
            history.append({**stats,"prior_development_objective":val,
                            "prior_development":ds,"fixed_fit_objective":fv,
                            "fixed_fit":fs,"weight_error":err})
            if best is None or val<best[0]:
                best=(val,cs,nb,err,stats["step"])
            print(key,label,"checkpoint",stats["step"],"of",STEPS,"dev",val,
                  "fixed fit",fv,flush=True)
            save()
        checkpoint(paths,{"step":0})
        optimize(paths,wg,fit,.00005,warmup=20,steps=STEPS,
                 callback=checkpoint,weights=weights)
        _,cs,nb,err,step=best
        record(label,lambda x,g:e.packed_actions(cs,x,g),nb,rr,err,cap,history,
               {"selected_step":step,"sign_lr":.00005,"warmup":20,
                "objective_weights":list(weights),
                "decoded_bf16_factor_bytes":sum(t.nbytes for c in cs for t in c)})
        if fresh: selected[label]=cs
        del row["active_arm"]
        save()
        del paths,best,cs
    # Select the arm before any fresh scores are evaluated.
    winner=min(row["candidates"],key=lambda k:row["candidates"][k]["prior_development_objective"])
    row["development_winner"]=winner
    if fresh:
        # Report all preregistered arms; do not choose a new winner on this bank.
        for label,cs in selected.items():
            vals,obj=score(lambda x,g:e.packed_actions(cs,x,g),fresh_panels,fresh_refs)
            row["candidates"][label].update(fresh_confirmation=vals,fresh_confirmation_objective=obj)
        if [codec.sha256(folder/r["file"]) for r in records]!=hashes:
            raise ValueError("fresh captures changed during screen")
    row["status"]="finished"; save()


if __name__=="__main__":
    # Strip only the optional new argument before delegating to the proven
    # common parser and source identity validation.
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fresh-stock",type=Path)
    extra,rest=parser.parse_known_args()
    FRESH_STOCK=extra.fresh_stock
    sys.argv=[sys.argv[0],*rest]
    base.main(run_fn=run,protocol="pure-binary-objective-v1",fit_partition_used=True)
