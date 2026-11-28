"""Block-coordinate binary sign refinement with low-rank exact joint guards."""
import torch
from discrete_rotation_core import local, restore


@torch.no_grad()
def block_sweep(cs,obj,block=32,seed=42):
    if block<1 or obj.mode!='task': raise ValueError('positive block and task objective required')
    p=local(cs,obj); E,F,D=obj.residual(p); before=obj.loss((E,F,D)); total=before; stats=[]
    wa,wf,wb=obj.k
    gen=torch.Generator().manual_seed(seed)
    for i,(a,b,r,s,c) in enumerate(p):
        for plane in ('A','B'):
            # B orientation uses E.T, swapped capture roles and forward/backward weights.
            if plane=='A':
                S=a; V=b*r[:,None]*c; scale=s; X,G=obj.x,obj.g
                R,T,Z=E,F,D; k0,k1,k2=wa,wf,wb
            else:
                S=b.T; V=a.T*s[:,None]*c; scale=r; X,G=obj.g,obj.x
                R,T,Z=E.T,D,F; k0,k1,k2=wa,wb,wf
            xn=X.square().sum(0)
            order=torch.randperm(len(c),generator=gen).to(S.device)
            for ix in order.split(block):
                U=V[:,ix]; GU=G@U; uu=U.T@U; gg=GU.T@GU
                d=-2*S[ix]*scale
                h=k0*(U.T@R)+k1*((T@U).T@X)+k2*(GU.T@Z)
                scores=2*d*h+d.square()*((k0*uu.diag()+k2*gg.diag())[:,None]+k1*uu.diag()[:,None]*xn)
                if not torch.isfinite(scores).all(): raise ValueError('nonfinite local scores')
                ids=(scores.flatten() < -1e-15).nonzero().flatten()
                proposed=len(ids); accepted=attempts=0; change=0.
                if proposed:
                    ids=ids[torch.argsort(scores.flatten()[ids],stable=True)]
                    count=proposed
                    while count:
                        attempts+=1
                        mask=torch.zeros_like(d,dtype=torch.bool)
                        mask.view(-1)[ids[:count]]=True
                        v=d*mask; xv=X@v.T; vv=v@v.T
                        delta=float(2*(v*h).sum()+((k0*uu+k2*gg)*vv).sum()+k1*(uu*(xv.T@xv)).sum())
                        if not torch.isfinite(torch.tensor(delta,dtype=torch.float64)): raise ValueError('nonfinite joint delta')
                        if delta < -1e-15:
                            # Low-rank delta to all three residuals, then exact refresh at sweep end.
                            R.addmm_(U,v); T.addmm_(xv,U.T); Z.addmm_(GU,v)
                            S[ix]=torch.where(mask,-S[ix],S[ix]); accepted=count; change=delta; total+=delta
                            break
                        count=count//2
                stats.append(dict(path=i,plane=plane,atoms=len(ix),proposed=proposed,accepted=accepted,attempts=attempts,delta=change))
    actual=obj.residual(p); after=obj.loss(actual); incremental=obj.loss((E,F,D))
    max_res=max(float((a-b).abs().max()) for a,b in zip(actual,(E,F,D)))
    if after>before+1e-12 or abs(after-total)>1e-10*max(1.,before) or abs(after-incremental)>1e-10*max(1.,before):
        raise ValueError('block sweep objective or residual mismatch')
    return restore(p,cs),dict(before=before,after=after,incremental=incremental,predicted=total,max_residual_abs=max_res,block=block,groups=stats,flips=sum(t['accepted'] for t in stats),proposed=sum(t['proposed'] for t in stats))
