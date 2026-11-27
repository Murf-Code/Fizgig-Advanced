"""LB-G: byte-matched SVD8 primary + binary residual, local research only."""
import copy
import binary_qat_stabilization as f
import math
import torch
from torch import nn
import native_codec as codec
import binary_rank_screen as base
import binary_rank_core as old
import mixed_factor_core as mix
import mixed_factor_screen as mixscreen
import private_correction as pc
import qat_core as q
import binary_qat_screen as e

CHECKPOINTS = (20, 50, 100)
CAPS = (.5, 1.)
SCALE_LR = .001


def components(p, mode):
    a, b, row, col, latent = p.components("hard")
    if mode == "smooth":
        def surrogate(z):
            soft = torch.tanh(100 * z)
            return q.signs(z) + (soft - soft.detach())
        a, b = surrogate(p.a), surrogate(p.b)
    return a, b, row, col, latent


def optimize(paths, w, fit, sign_lr, warmup=0, steps=200, callback=None,
             start_step=0, seed=42, fixed=None):
    """Same row stream across arms; pooled target energy; separate Adam groups."""
    signs = [v for p in paths for v in (p.a, p.b)]
    scales = [v for p in paths for v in (p.row, p.col, p.latent)]
    params = signs + scales
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
    # Frozen SVD8 path, decoded once to FP32 for fitting.
    fixed = tuple(t.to(w.device, dtype=torch.float32) for t in fixed) if fixed is not None else None
    fixed_fit = [(q.actions(fixed,x,g) if fixed is not None else (0.,0.)) for x,g,y,dx in fit]
    initial = [v.detach() >= 0 for v in signs]
    prev = [v.clone() for v in initial]
    flips = [0] * len(signs)
    for _ in range(start_step):
        torch.randperm(len(w), device=w.device, generator=gen)
    for local in range(1, steps + 1):
        step = start_step + local
        active = sign_lr > 0 and local > warmup
        for v in signs:
            v.requires_grad_(active)
        idx = torch.randperm(len(w), device=w.device, generator=gen)[:batch]
        opt.zero_grad(set_to_none=True)
        cs = [components(p, "smooth" if active else "hard") for p in paths]
        weight = (sum(q.weight(c, idx) for c in cs) + (q.weight(fixed,idx) if fixed is not None else 0.) - w[idx].float()).square().sum()
        loss = .2 * weight * (len(w) / batch / wenergy)
        for (x,g,y,dx),(fy,fg) in zip(fit,fixed_fit):
            pairs = [q.actions(c,x,g) for c in cs]
            loss = loss + .4*(sum(t[0] for t in pairs)+fy-y).square().sum()/fenergy
            loss = loss + .4*(sum(t[1] for t in pairs)+fg-dx).square().sum()/genergy
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
                stats = {"step":step, "batch_objective_before_update":float(loss),
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
               scope="fit-only optimization; development selects checkpoint; prior confirmation exploratory",
               objective="pooled fit energy; unbiased weight row estimate; FP32 fit, BF16 evaluation",
               checkpoint_steps=list(CHECKPOINTS),scale_lr=SCALE_LR,
               hybrid_scope="fixed SVD8 primary plus optimized binary residual; packed bytes are ceilings")
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
    for bits in (2,3):
        op,nb,err=mixscreen.direct_lowbit(w,bits)
        record(f"direct-q{bits}-odd-had",op,nb,[],err)
        del op
    for name in ("q4-had","q6-had"):
        if name=="q6-had" and archive is not None:
            p,s=archive.get_tensor(key),codec.manifest(archive)[key]
            if s["codec"]!=name or s["shape"]!=[m,n]: raise ValueError("archive mismatch")
            codec.validate(p,s)
        else: p,s=codec.encode(w,name)
        record(name,lambda x,g:pc.operator(p,s,x,g),p.nbytes,[],pc.weight_error(w,p,s))
        del p,s
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
    def binary_arm(name,initial,cap,lr,warmup=0,sequential_rank=None):
        paths=copy.deepcopy(initial)
        history=[]
        best=None
        def checkpoint(ps,stats):
            nonlocal best
            cs,nb=e.packed(ps,m,n)
            op=lambda x,g:e.packed_actions(cs,x,g)
            _,val=score(op,dev,dr)
            err=e.error(w,cs)
            history.append({**stats,"prior_development_objective":val,"weight_error":err})
            if best is None or val<best[0]:
                best=(val,cs,nb,err,stats["step"],[p.a.shape[0] for p in ps])
            print(key,name,"checkpoint",stats["step"],"dev",val,flush=True)
        if sequential_rank is None:
            checkpoint(paths,{"step":0})
            optimize(paths,wg,fit,lr,warmup=warmup,steps=100,callback=checkpoint)
        else:
            # 20 primary updates then 80 joint updates; new residual fitted
            # to the actual rounded primary. Reset Adam at the phase boundary.
            optimize(paths,wg,fit,lr,steps=20)
            paths.append(residual(paths[0],sequential_rank))
            checkpoint(paths,{"step":20,"phase":"residual initialized"})
            optimize(paths,wg,fit,lr,steps=80,start_step=20,callback=checkpoint)
        _,cs,nb,err,step,ranks=best
        record(name,lambda x,g:e.packed_actions(cs,x,g),nb,ranks,err,cap,history,
               {"selected_step":step,"sign_lr":lr,"warmup":warmup,
                "decoded_bf16_factor_bytes":sum(t.nbytes for c in cs for t in c)})
    def svd_pair(rank):
        a,b=old.balanced(u,s,v,rank)
        da,na=mix.quant_rows(a,8,codec._pack,codec._unpack)
        bt,nb=mix.quant_rows(b.T,8,codec._pack,codec._unpack)
        db=bt.T.contiguous()
        if na+nb!=mix.pair_bytes(m,n,rank,8,8):
            raise ValueError("SVD8 bytes mismatch")
        return da,db,na+nb

    def hybrid_arm(prefix,cap,budget,pct):
        # Charge SVD8 factors and binary scales; allocate remainder to residual.
        # Total rank cap applies across both branches.
        r8=mix.budget_rank(m,n,8,8,budget*pct//100,top)
        if not r8: return
        da,db,nb8=svd_pair(r8)
        rr=q.ranks(m,n,budget-nb8,top-r8,1) if top>r8 else []
        if not rr:
            row.setdefault("skipped_hybrids",[]).append({"cap":cap,"svd_percent":pct,"reason":"no residual rank"})
            save(); return
        r=rr[0]
        fixed=(da,db,torch.ones(m,dtype=torch.bfloat16),
               torch.ones(n,dtype=torch.bfloat16),torch.ones(r8,dtype=torch.bfloat16))
        with torch.no_grad():
            # Subtract actual decoded SVD8 product, not continuous SVD.
            target=wg.float()
            fixed_gpu=tuple(t.to(device,dtype=torch.float32) for t in fixed)
            for i in range(0,m,256):
                idx=slice(i,min(i+256,m))
                target[idx]-=q.weight(fixed_gpu,idx)
            del fixed_gpu
            with torch.random.fork_rng(devices=[device.index or 0] if device.type=="cuda" else []):
                torch.manual_seed(43)
                ur,sr,vr=torch.svd_lowrank(target,q=min(r+16,m,n),niter=4)
            a,b=old.balanced(ur,sr,vr,r)
            del target,ur,sr,vr
            paths=nn.ModuleList([init(a,b)])
            del a,b
        history=[]; best=None
        def op_for(cs):
            def op(x,g):
                z=x.detach().bfloat16().requires_grad_(True)
                # Backprop through the sum to preserve BF16 addition ordering.
                y=(z@da.T)@db.T+sum(q.actions(c,z,g.bfloat16())[0] for c in cs)
                dx,=torch.autograd.grad(y,z,g.bfloat16())
                return y.detach(),dx.detach()
            return op
        def checkpoint(ps,stats):
            nonlocal best
            cs,nb=e.packed(ps,m,n)
            _,val=score(op_for(cs),dev,dr)
            err=e.error(w,[fixed,*cs])
            history.append({**stats,"prior_development_objective":val,"weight_error":err})
            if best is None or val<best[0]:
                best=(val,cs,nb,err,stats["step"])
            print(key,prefix+"hybrid"+str(pct),"checkpoint",stats["step"],"dev",val,flush=True)
        checkpoint(paths,{"step":0})
        optimize(paths,wg,fit,.0001,warmup=20,steps=100,callback=checkpoint,fixed=fixed)
        _,cs,nb,err,step=best
        record(prefix+"G"+str(pct),op_for(cs),nb8+nb,[r8,r],err,cap,history,
               {"selected_step":step,"svd_rank":r8,"binary_rank":r,
                "svd_payload_bytes":nb8,"binary_payload_bytes":nb,
                "svd_target_share":pct/100,"svd_actual_share":nb8/(nb8+nb),
                "cap_utilization":(nb8+nb)/budget,
                "decoded_bf16_factor_bytes":da.nbytes+db.nbytes+sum(t.nbytes for c in cs for t in c),
                "sign_lr":.0001,"warmup":20,"fixed_svd8":True})
    seen=set()
    for cap in CAPS:
        budget=int(cap*m*n//8); prefix=f"{cap:g}bit-"
        r8=mix.budget_rank(m,n,8,8,budget,top)
        if r8:
            da,db,nb=svd_pair(r8)
            record(prefix+"G100-svd8",lambda x,g:old.actions(x,g,da,db),nb,[r8],
                   old.weight_error(w,da,db),cap,extra={"svd_rank":r8,"binary_rank":0,
                     "cap_utilization":nb/budget,"decoded_bf16_factor_bytes":da.nbytes+db.nbytes})
            del da,db
        rr=q.ranks(m,n,budget,top,2)
        if rr and tuple(rr) not in seen:
            seen.add(tuple(rr))
            a,b=old.balanced(u,s,v,rr[0])
            first=init(a,b); del a,b
            dual=nn.ModuleList([first,residual(first,rr[1])])
            cs,nb=e.packed(dual,m,n)
            record(prefix+"G0-F0",lambda x,g:e.packed_actions(cs,x,g),nb,rr,e.error(w,cs),cap)
            del cs
            for label,lr,warm in (("G0-F1",0.,0),("G0-F2",.0001,0),("G0-F5",.0001,20)):
                binary_arm(prefix+label,dual,cap,lr,warm)
            binary_arm(prefix+"G0-F6",nn.ModuleList([first]),cap,.0001,sequential_rank=rr[1])
            del dual,first
        elif rr:
            row.setdefault("deduplicated_binary_caps",[]).append(cap); save()
        for pct in (25,50,75):
            hybrid_arm(prefix,cap,budget,pct)
    row["status"]="finished"; save()


if __name__=="__main__":
    base.main(run_fn=run,protocol="hybrid-svd8-binary-v1",fit_partition_used=True)
