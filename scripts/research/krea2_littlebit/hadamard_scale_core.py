"""Research-only odd-level group quantization and matrix-free scale fitting."""
import math
import torch


@torch.no_grad()
def initial_codes(z, bits, group=64, steps=4, tile=128):
    """Same four-step L2 initialization as mixed_factor_core.quant_rows."""
    if z.ndim!=2 or z.shape[1]%group or bits not in (2,3) or not torch.isfinite(z).all():
        raise ValueError('invalid transformed matrix')
    levels=(1<<bits)-1
    w=z.float().reshape(-1,group)
    codes=torch.empty_like(w,dtype=torch.uint8)
    scales=torch.empty(len(w),device=w.device,dtype=torch.bfloat16)
    for i in range(0,len(w),tile):
        j=min(i+tile,len(w)); t=w[i:j]
        s=(t.abs().amax(1)/levels).bfloat16()
        besterr=torch.full_like(s,float('inf'),dtype=torch.float32)
        for _ in range(steps):
            safe=torch.where(s>0,s,torch.ones_like(s)).float()
            q=((t/safe[:,None]+levels)/2).round().clamp(0,levels)
            v=2*q-levels
            err=(t-v*s.float()[:,None]).square().sum(1)
            keep=err<besterr
            codes[i:j]=torch.where(keep[:,None],q.byte(),codes[i:j])
            scales[i:j]=torch.where(keep,s,scales[i:j])
            besterr=torch.minimum(besterr,err)
            s=((t*v).sum(1)/v.square().sum(1)).clamp_min(0).bfloat16()
        if not torch.isfinite(besterr).all(): raise ValueError('nonfinite initialization')
    return codes.reshape(z.shape),scales.reshape(len(z),-1)


def decode(codes,scales,bits):
    return ((2*codes.to(torch.int32)-((1<<bits)-1)).reshape(*scales.shape,-1).bfloat16()*scales[...,None]).reshape(codes.shape)


class ScaleSystem:
    """Quadratic in continuous scales, with original transformed teacher energies."""
    def __init__(self,z,x,g,codes,bits,weights=(.2,.4,.4),device='cpu',dtype=torch.float64):
        self.z,self.x,self.g=(t.to(device=device,dtype=dtype) for t in (z,x,g))
        m,n=z.shape
        if codes.shape!=z.shape or n%64 or x.shape[1]!=n or g.shape[1]!=m:
            raise ValueError('shape mismatch')
        if len(weights)!=3 or min(weights)<=0 or not math.isclose(sum(weights),1.):
            raise ValueError('invalid objective weights')
        self.v=(2*codes.to(device=device,dtype=dtype)-((1<<bits)-1)).reshape(m,n//64,64)
        energy=[self.z.square().sum(),(self.x@self.z.T).square().sum(),(self.g@self.z).square().sum()]
        if not all(torch.isfinite(t) and t>0 for t in energy): raise ValueError('invalid teacher energy')
        self.k=tuple(float(a/b) for a,b in zip(weights,energy))
        a,b,c=self.k
        xg=self.x.reshape(len(self.x),n//64,64).transpose(0,1).contiguous()
        vg=self.v.transpose(0,1).contiguous()
        xx=xg.transpose(1,2)@xg
        local=((vg@xx)*vg).sum(-1).T
        self.diag=(a+c*self.g.square().sum(0)[:,None])*self.v.square().sum(-1)+b*local
        if not torch.isfinite(self.diag).all() or not (self.diag>0).all(): raise ValueError('invalid diagonal')

    def expand(self,s): return (self.v*s[...,None]).reshape_as(self.z)
    def adjoint(self,e): return (e.reshape_as(self.v)*self.v).sum(-1)
    def normal(self,e):
        a,b,c=self.k
        return a*e+b*((e@self.x.T)@self.x)+c*(self.g.T@(self.g@e))
    def matvec(self,s): return self.adjoint(self.normal(self.expand(s)))
    def rhs(self,s): return -self.adjoint(self.normal(self.expand(s)-self.z))


@torch.no_grad()
def solve(system,s0,maxiter=30,rtol=1e-5):
    """Diagonally preconditioned CG for displacement from initial scales."""
    s=s0.to(system.z); d=system.diag.sqrt()
    rhs=system.rhs(s)/d
    r=rhs.clone(); p=r.clone(); u=torch.zeros_like(r)
    rr=(r*r).sum(); initial=float(rr.sqrt())
    if not math.isfinite(initial): raise ValueError('invalid RHS')
    if initial==0: return s,dict(iterations=0,relative_residual=0.,converged=True)
    for it in range(1,maxiter+1):
        hp=system.matvec(p/d)/d
        curvature=(p*hp).sum()
        if not torch.isfinite(curvature) or curvature<=0: raise ValueError('invalid CG curvature')
        alpha=rr/curvature
        u+=alpha*p; r-=alpha*hp
        new=(r*r).sum()
        if not torch.isfinite(new): raise ValueError('nonfinite CG residual')
        if float(new.sqrt())<=rtol*initial: break
        p=r+(new/rr)*p; rr=new
    # Recompute the true residual; recurrence may drift in FP32.
    rel=float((rhs-system.matvec(u/d)/d).norm())/initial
    return s+u/d,dict(iterations=it,relative_residual=rel,converged=rel<=rtol)


@torch.no_grad()
def metrics(z,x,g,decoded,weights=(.2,.4,.4)):
    """Direct FP64 objective for the actual BF16 decoded product."""
    z,x,g=(t.double() for t in (z,x,g)); e=decoded.double()-z
    vals=[float(e.square().sum()/z.square().sum()),
          float((x@e.T).square().sum()/(x@z.T).square().sum()),
          float((g@e).square().sum()/(g@z).square().sum())]
    if not all(math.isfinite(v) for v in vals): raise ValueError('nonfinite fit metric')
    return dict(weight_relative_squared=vals[0],forward_relative_squared=vals[1],backward_relative_squared=vals[2],objective=sum(a*b for a,b in zip(weights,vals)))


@torch.no_grad()
def rounded_guard(z,x,g,codes,bits,s0,proposal,weights):
    before=metrics(z,x,g,decode(codes,s0,bits),weights)
    proposal=proposal.to(s0.device,dtype=torch.float64)
    if not torch.isfinite(proposal).all(): raise ValueError('nonfinite scale proposal')
    history=[]
    for fraction in (1.,.5,.25,.125,.0625):
        scales=(s0.double()+fraction*(proposal-s0.double())).clamp_min(0).bfloat16()
        if not torch.isfinite(scales).all(): continue
        after=metrics(z,x,g,decode(codes,scales,bits),weights)
        history.append(dict(fraction=fraction,objective=after['objective']))
        if after['objective']<before['objective']-1e-12:
            return scales,dict(before=before,after=after,accepted=True,fraction=fraction,trials=history)
    return s0.clone(),dict(before=before,after=before,accepted=False,fraction=0.,trials=history)
