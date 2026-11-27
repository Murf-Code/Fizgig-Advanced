"""LB-E1–E6: controlled local binary QAT, not full-model distillation."""
import copy
import math
import torch
from torch import nn
import native_codec as codec
import binary_rank_screen as base
import binary_rank_core as old
import joint_latent as joint
import private_correction as pc
import mixed_factor_screen as mixscreen
import mixed_factor_core as mix
import qat_core as q

CAPS=(.5,1.)
STEPS=100
LR=.001


def packed(paths,m,n):
    result=[]; nb=0
    with torch.no_grad():
        for p in paths:
            a,b,row,col,c=[t.detach().cpu() for t in p.components()]
            pa,pb=codec._pack(a>0,1),codec._pack(b>0,1)
            r=a.shape[0]
            a=(2*codec._unpack(pa,r*n,1).reshape(r,n)-1).bfloat16()
            b=(2*codec._unpack(pb,m*r,1).reshape(m,r)-1).bfloat16()
            row,col,c=[t.bfloat16() for t in (row,col,c)]
            count=pa.nbytes+pb.nbytes+row.nbytes+col.nbytes+c.nbytes
            if count!=q.size(m,n,r): raise ValueError("packed byte mismatch")
            nb+=count; result.append((a,b,row,col,c))
    return result,nb


def packed_actions(cs,x,g):
    z=x.detach().bfloat16().requires_grad_(True)
    y=sum(q.actions(c,z,g.bfloat16())[0] for c in cs)
    dx,=torch.autograd.grad(y,z,g.bfloat16())
    return y.detach(),dx.detach()


def error(w,cs):
    cs=[tuple(t.float() for t in c) for c in cs]
    num=den=0.
    for i in range(0,len(w),256):
        idx=slice(i,min(i+256,len(w)))
        wt=w[idx].float()
        num+=float((sum(q.weight(c,idx) for c in cs)-wt).square().sum(dtype=torch.float64))
        den+=float(wt.square().sum(dtype=torch.float64))
    return math.sqrt(num/den) if den else None


def run(w,key,args,stock,trace,split,int8,archive,row,save):
    m,n=w.shape; device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("highest")
    native=codec.stock_layer(w,int8)
    panels={label:[base.read_capture(stock,key,r,trace) for r in split[source]]
            for label,source in (("prior_development","development"),("prior_confirmation","confirmation"))}
    refs={label:[base.refs(w,c,native) for c in cs] for label,cs in panels.items()}
    row.update(shape=[m,n],candidates={},device=str(device),qat_steps=STEPS,qat_lr=LR,
               scope="local QAT; fit uses reserved stock captures; both scoring panels previously inspected")
    def add(name,op,nb,ranks,err,kind,cap=None,history=None):
        total=sum(ranks)
        rec={"kind":kind,"ranks":ranks,"payload_bytes":nb,"effective_bits":8*nb/(m*n),
             "weight_error":err,"dense_equivalent_macs_per_token":total*(m+n) if total else m*n,
             "decoded_bf16_factor_bytes":2*(total*(m+n+1)+len(ranks)*(m+n)) if total else None,
             "cap":cap,"cap_utilization":nb/(cap*m*n/8) if cap else None,"history":history}
        if cap and nb>int(cap*m*n//8): raise ValueError("budget exceeded")
        for label,cs in panels.items():
            scores=[base.score(op,c,r) for c,r in zip(cs,refs[label])]
            rec[label]=scores; rec[label+"_objective"]=base.objective(scores)
        row["candidates"][name]=rec
        print(key,name,"bits",rec["effective_bits"],"weight error",err,"prior dev",rec["prior_development_objective"],flush=True)
        save()
    for bits in (2,3):
        op,nb,err=mixscreen.direct_lowbit(w,bits)
        add(f"direct-q{bits}-odd-had",op,nb,[],err,"direct-control")
        del op
    for name in ("q4-had","q6-had"):
        if name=="q6-had" and archive is not None:
            p=archive.get_tensor(key); s=codec.manifest(archive)[key]
            if s["codec"]!=name or s["shape"]!=[m,n]: raise ValueError("Q6 archive mismatch")
            codec.validate(p,s)
        else: p,s=codec.encode(w,name)
        add(name,lambda x,g:pc.operator(p,s,x,g),p.nbytes,[],pc.weight_error(w,p,s),"uniform-control")
        del p,s
    top=min(m,n,args.max_rank)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        u,s,v=torch.svd_lowrank(w.float(),q=min(top+16,m,n),niter=4)
    wg=w.to(device); fit=[]
    with torch.no_grad():
        for r in split["fit"]:
            c=base.read_capture(stock,key,r,trace)
            x,g=c["x"].to(device,dtype=torch.float32),c["dy"].to(device,dtype=torch.float32)
            wf=wg.float()
            fit.append((x,g,x@wf.T,g@wf))
            del wf
    if not fit: raise ValueError("missing fit captures")
    seen=set()
    for cap in CAPS:
        budget=int(cap*m*n//8)
        r=q.ranks(m,n,budget,top,1)
        rr=q.ranks(m,n,budget,top,2)
        if not r or not rr: raise ValueError("budget too small")
        sig=(tuple(r),tuple(rr))
        if sig in seen: continue
        seen.add(sig)
        prefix=f"{cap:g}bit-"
        a,b=old.balanced(u,s,v,r[0])
        aa,bb=old.rotate(a,b)
        sa=q.signs(aa).bfloat16(); sb=q.signs(bb).bfloat16()
        c=joint.fit_latent(w,sa,sb); ro=torch.ones(m,dtype=torch.bfloat16); co=torch.ones(n,dtype=torch.bfloat16)
        for _ in range(3):
            ro,co=joint.fit_outer(w,sa,sb,c,ro,co)
            c=joint.fit_latent(w,sa,sb,ro,co)
        # Existing control's original scale folding and CPU BF16 action ordering.
        db=sb*c[None,:]
        add(prefix+"E1",lambda x,g:old.actions(x,g,sa,db,ro,co),q.size(m,n,r[0]),r,
            old.weight_error(w,sa,db,ro,co),"old-blockwise-joint",cap)
        del aa,bb,sa,sb,c,ro,co,db
        # Higher-precision factor control at the same budget.
        r8=mix.budget_rank(m,n,8,8,budget,top)
        if r8:
            a8,b8=old.balanced(u,s,v,r8)
            da,na=mix.quant_rows(a8,8,codec._pack,codec._unpack)
            bt,nb=mix.quant_rows(b8.T,8,codec._pack,codec._unpack)
            db=bt.T.contiguous()
            add(prefix+"svd8",lambda x,g:old.actions(x,g,da,db),na+nb,[r8],
                old.weight_error(w,da,db),"mixed-8bit-control",cap)
            del a8,b8,da,bt,db
        def init(a,b):
            with torch.no_grad():
                a,b=q.rotate(a.to(device),b.to(device))
                return q.Path(a,b)
        def evaluate(name,paths,kind,history=None):
            cs,nb=packed(paths,m,n)
            add(prefix+name,lambda x,g:packed_actions(cs,x,g),nb,[p.a.shape[0] for p in paths],
                error(w,cs),kind,cap,history)
        single=nn.ModuleList([init(a,b)]); del a,b
        evaluate("E2",single,"full-rotation-dual-svid-single")
        a,b=old.balanced(u,s,v,rr[0])
        first=init(a,b); del a,b
        with torch.no_grad():
            # Residual against actual hard signs and BF16-rounded scales.
            c=first.components()
            residual=wg.float()
            for i in range(0,m,256):
                idx=slice(i,min(i+256,m))
                residual[idx]-=q.weight(c,idx)
            del c
            with torch.random.fork_rng(devices=[device.index or 0] if device.type=="cuda" else []):
                torch.manual_seed(43)
                ur,sr,vr=torch.svd_lowrank(residual,q=min(rr[1]+16,m,n),niter=4)
            a,b=old.balanced(ur,sr,vr,rr[1])
            del residual,ur,sr,vr
        dual=nn.ModuleList([first,init(a,b)]); del first,a,b
        evaluate("E3",dual,"full-rotation-dual-svid-residual")
        for label,initial,action in (("E4",single,False),("E5",dual,False),("E6",dual,True)):
            for mode in ("ste","smooth"):
                print(key,prefix+label,mode,"starting",flush=True)
                paths=copy.deepcopy(initial)
                history=q.train(paths,wg,fit if action else [],mode,steps=STEPS,lr=LR,action=action)
                evaluate(label+"-"+mode,paths,"local-action-qat" if action else "local-weight-qat",history)
                del paths
        del single,dual
    row["status"]="finished"; save()


if __name__=="__main__":
    base.main(run_fn=run,protocol="binary-qat-v1",fit_partition_used=True)
