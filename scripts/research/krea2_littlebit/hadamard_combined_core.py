"""Offline discrete code updates and row-level precision allocation."""
import math
import torch
from hadamard_scale_core import decode


class Fit:
    def __init__(self,z,x,g,weights,device='cpu',dtype=torch.float64):
        self.z,self.x,self.g=(t.to(device=device,dtype=dtype) for t in (z,x,g))
        self.k=tuple(float(a/b) for a,b in zip(weights,(self.z.square().sum(),(self.x@self.z.T).square().sum(),(self.g@self.z).square().sum())))
        if not all(math.isfinite(k) and k>0 for k in self.k): raise ValueError('invalid fit energies')
        self.xn=self.x.square().sum(0); self.gn=self.g.square().sum(0)

    def residual(self,d):
        e=d.to(self.z)-self.z
        return e,self.x@e.T,self.g@e

    def loss(self,res):
        return sum(k*float(t.square().sum(dtype=torch.float64)) for k,t in zip(self.k,res))

    def gradient(self,res,ix):
        e,f,d=res; a,b,c=self.k
        return a*e[ix]+b*(f[:,ix].T@self.x)+c*(self.g[:,ix].T@d)

    def diagonal(self,ix):
        a,b,c=self.k
        return a+b*self.xn+c*self.gn[ix,None]


@torch.no_grad()
def code_sweep(codes,scales,bits,fit,block=128,seed=42):
    """Block-coordinate proposals over rows, exact BF16 decoded code values.

    A sorted prefix is halved until the joint quadratic decreases. Rebuild
    residuals at sweep end. Caller independently checks FP64 rounded fit.
    """
    if block<1 or bits not in (2,3): raise ValueError('invalid code sweep')
    q=codes.to(fit.z.device).clone(); s=scales.to(fit.z.device)
    E,F,D=fit.residual(decode(q,s,bits)); before=fit.loss((E,F,D)); total=before
    a,b,c=fit.k; stats=[]
    order=torch.randperm(len(q),generator=torch.Generator().manual_seed(seed)).to(q.device)
    for ix in order.split(block):
        old=decode(q[ix],s[ix],bits).to(fit.z); h=fit.gradient((E,F,D),ix); diag=fit.diagonal(ix)
        best=torch.zeros_like(old); new=q[ix].clone(); change=torch.zeros_like(old)
        for j in range(1<<bits):
            value=((2*j-((1<<bits)-1))*s[ix][...,None]).expand(-1,-1,q.shape[1]//s.shape[1]).reshape_as(old).to(fit.z)
            v=value-old; score=2*v*h+diag*v.square(); better=score<best
            best=torch.where(better,score,best); new=torch.where(better,j,new); change=torch.where(better,v,change)
        ids=(best.flatten() < -1e-15).nonzero().flatten(); proposed=len(ids); count=proposed; accepted=attempts=0; delta=0.
        if count: ids=ids[torch.argsort(best.flatten()[ids],stable=True)]
        while count:
            attempts+=1
            mask=torch.zeros_like(old,dtype=torch.bool); mask.view(-1)[ids[:count]]=True
            v=change*mask; fv=fit.x@v.T; dv=fit.g[:,ix]@v
            delta=float(2*(v*h).sum(dtype=torch.float64)+a*v.square().sum(dtype=torch.float64)+b*fv.square().sum(dtype=torch.float64)+c*dv.square().sum(dtype=torch.float64))
            if not math.isfinite(delta): raise ValueError('nonfinite code delta')
            if delta < -1e-15:
                E[ix]+=v; F[:,ix]+=fv; D+=dv
                q[ix]=torch.where(mask,new,q[ix]); total+=delta; accepted=count; break
            count//=2
        stats.append(dict(proposed=proposed,accepted=accepted,attempts=attempts,delta=delta if accepted else 0.))
    actual=fit.residual(decode(q,s,bits)); after=fit.loss(actual); incremental=fit.loss((E,F,D))
    tol=1e-9 if fit.z.dtype==torch.float64 else 2e-6
    if after>before+tol or abs(after-total)>tol or abs(after-incremental)>tol: raise ValueError('code residual/objective mismatch')
    return q.cpu(),dict(before=before,after=after,predicted=total,incremental=incremental,groups=stats,proposed=sum(t['proposed'] for t in stats),accepted=sum(t['accepted'] for t in stats),net_changed=int((q.cpu()!=codes.cpu()).sum()))


@torch.no_grad()
def row_priority(d2,d3,fit):
    """Exact isolated row replacement delta; rankings ignore inter-row coupling."""
    res=fit.residual(d2); v=d3.to(fit.z)-d2.to(fit.z); a,b,c=fit.k
    h=fit.gradient(res,slice(None))
    return (2*(v*h).sum(1)+ (a+c*fit.gn)*v.square().sum(1)+b*(fit.x@v.T).square().sum(0)).cpu()


def mixed_count(m,n):
    """Largest Q3 row count under 2.5 bits, including packed row mask."""
    if n%64 or m<1: raise ValueError('invalid shape')
    cap=m*n*5//16; base=m*(n//4+2*(n//64)); mask=(m+7)//8
    return max(0,min(m,(cap-base-mask)//(n//8)))
