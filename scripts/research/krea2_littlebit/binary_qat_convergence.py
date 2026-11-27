"""LB-H1: pure binary LittleBit convergence, no hybrid or trainer changes."""
import copy
import math
import torch
from torch import nn
import native_codec as codec
import binary_rank_screen as base
import binary_rank_core as old
import qat_core as q
import binary_qat_screen as e
from binary_qat_stabilization import components, SCALE_LR

CHECKPOINTS = (20, 50, 100, 200, 350, 500, 750, 1000)
STEPS = 1000
# (label, sign LR, scale-only warmup, schedule, freeze after update)
ARMS = (
    ("H1-F2-constant", .0001, 0, "constant", None),
    ("H2-F5-constant", .0001, 20, "constant", None),
    ("H3-F5-cosine", .0001, 20, "cosine", None),
    ("H4-F5-low-lr", .00005, 20, "constant", None),
    ("H5-F5-freeze100", .0001, 20, "constant", 100),
    ("H6-scale-only", 0., 0, "constant", None),
)


def rate(step, lr, warmup, schedule, freeze_after):
    if step<1 or lr<0 or schedule not in ("constant","cosine"):
        raise ValueError("invalid sign schedule")
    if step<=warmup or (freeze_after is not None and step>freeze_after):
        return 0.
    if schedule=="constant": return lr
    # Preserve the exact F5 prefix through update 100. Decay to 5% by 500;
    # hold this floor through 1000 to test whether slower movement helps.
    t=min(1.,max(0.,(step-100)/400))
    return lr*(.05+.95*.5*(1+math.cos(math.pi*t)))


def optimize(paths, w, fit, sign_lr, warmup=0, steps=STEPS, callback=None,
             seed=42, schedule="constant", freeze_after=None):
    """Same row stream across arms; pooled target energy; separate Adam groups."""
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
        lr_now = rate(step,sign_lr,warmup,schedule,freeze_after)
        active = lr_now > 0
        opt.param_groups[0]["lr"] = lr_now
        for v in signs:
            v.requires_grad_(active)
        idx = torch.randperm(len(w), device=w.device, generator=gen)[:batch]
        opt.zero_grad(set_to_none=True)
        cs = [components(p, "smooth" if active else "hard") for p in paths]
        weight = (sum(q.weight(c, idx) for c in cs) - w[idx].float()).square().sum()
        loss = .2 * weight * (len(w) / batch / wenergy)
        for x,g,y,dx in fit:
            pairs = [q.actions(c,x,g) for c in cs]
            loss = loss + .4*(sum(t[0] for t in pairs)-y).square().sum()/fenergy
            loss = loss + .4*(sum(t[1] for t in pairs)-dx).square().sum()/genergy
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
               scope="pure binary convergence; old fit and prior scoring panels only; not independent validation",
               objective="pooled fit energy; unbiased weight row estimate; FP32 fit, BF16 evaluation",
               checkpoint_steps=list(CHECKPOINTS),scale_lr=SCALE_LR,max_steps=STEPS,bit_cap=1.0)
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
    for label,lr,warm,schedule,freeze in ARMS:
        paths=copy.deepcopy(dual)
        history=[]; best=None
        row["active_arm"]={"name":label,"history":history}
        def checkpoint(ps,stats):
            nonlocal best
            cs,nb=e.packed(ps,m,n)
            op=lambda x,g:e.packed_actions(cs,x,g)
            _,val=score(op,dev,dr)
            err=e.error(w,cs)
            history.append({**stats,"prior_development_objective":val,"weight_error":err})
            if best is None or val<best[0]:
                best=(val,cs,nb,err,stats["step"])
            print(key,label,"checkpoint",stats["step"],"of",STEPS,"dev",val,flush=True)
            # Persist progress even if a later arm is interrupted.
            save()
        checkpoint(paths,{"step":0})
        optimize(paths,wg,fit,lr,warmup=warm,steps=STEPS,callback=checkpoint,
                 schedule=schedule,freeze_after=freeze)
        _,cs,nb,err,step=best
        record(label,lambda x,g:e.packed_actions(cs,x,g),nb,rr,err,cap,history,
               {"selected_step":step,"sign_lr":lr,"warmup":warm,"schedule":schedule,
                "freeze_after":freeze,"decoded_bf16_factor_bytes":sum(t.nbytes for c in cs for t in c)})
        del row["active_arm"]
        save()
        del paths,best,cs
    row["status"]="finished"; save()


if __name__=="__main__":
    base.main(run_fn=run,protocol="pure-binary-convergence-v1",fit_partition_used=True)
