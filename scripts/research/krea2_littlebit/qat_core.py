"""Independent research-only binary factor optimization."""
import math
import torch
from torch import nn


def size(m,n,r):
    return (m*r+7)//8+(n*r+7)//8+2*(m+n+r)


def ranks(m,n,cap,max_rank,paths):
    if paths not in (1,2): raise ValueError('paths must be 1 or 2')
    # Equal rank per path. Limit total latent width, not each branch separately.
    lo,hi=0,min(m,n,max_rank)//paths
    while lo<hi:
        mid=(lo+hi+1)//2
        if paths*size(m,n,mid)<=cap: lo=mid
        else: hi=mid-1
    return [lo]*paths if lo else []


def signs(p):
    return torch.where(p>=0,1.,-1.)


def binary(p,mode):
    hard=signs(p)
    if mode=='hard': return hard
    if mode=='ste': soft=p.clamp(-1,1)
    elif mode=='smooth': soft=torch.tanh(100*p)
    else: raise ValueError(mode)
    return hard+(soft-soft.detach())


def rounded(p):
    return p+(p.bfloat16().float()-p).detach()


def magnitude(z,steps=20):
    # Nonnegative leading singular vectors of |z| via power iteration.
    t=z.abs().float(); v=torch.ones(t.shape[1],device=t.device)
    v=v/v.norm()
    for _ in range(steps):
        u=t@v; u=u/u.norm().clamp_min(1e-30)
        v=t.T@u; v=v/v.norm().clamp_min(1e-30)
    u=t@v; s=u.norm(); u=u/s.clamp_min(1e-30)
    return u*s.sqrt(),v*s.sqrt()


def rotate(a,b,iterations=20):
    z=torch.cat((a.T,b),0)
    gen=torch.Generator(device=z.device).manual_seed(42)
    q=torch.linalg.qr(torch.randn((a.shape[0],a.shape[0]),device=z.device,generator=gen))[0]
    for _ in range(iterations):
        u,_,vh=torch.linalg.svd(z.T@signs(z@q),full_matrices=False)
        q=u@vh
    return (a.T@q).T.contiguous(),(b@q).contiguous()


class Path(nn.Module):
    def __init__(self,a,b):
        super().__init__()
        al,col=magnitude(a); row,bl=magnitude(b)
        # Preserve initial signs; shadow RMS .01 makes beta100 a useful comparator.
        self.a=nn.Parameter(a.float()*(.01/a.float().square().mean().sqrt().clamp_min(1e-30)))
        self.b=nn.Parameter(b.float()*(.01/b.float().square().mean().sqrt().clamp_min(1e-30)))
        for name,v in [('row',row),('col',col),('latent',al*bl)]:
            self.register_buffer(name+'_base',v.detach())
            setattr(self,name,nn.Parameter(torch.ones_like(v)))
    def components(self,mode='hard'):
        return (binary(self.a,mode),binary(self.b,mode),
                rounded(self.row*self.row_base),rounded(self.col*self.col_base),
                rounded(self.latent*self.latent_base))


def weight(c,idx=slice(None)):
    a,b,row,col,latent=c
    return ((b[idx]*latent)@a)*row[idx,None]*col[None,:]


def actions(c,x,g):
    a,b,row,col,latent=c
    return ((((x*col)@a.T)*latent)@b.T)*row, ((((g*row)@b)*latent)@a)*col


def ratio(x,y):
    return (x-y).square().sum()/y.square().sum().clamp_min(1e-30)


def train(paths,w,fit,mode,steps=100,lr=.001,action=False,batch=256):
    """FP32 matmuls, hard signs and BF16-rounded scales. Fixed final step."""
    params=list(paths.parameters()); opt=torch.optim.Adam(params,lr=lr)
    gen=torch.Generator(device=w.device).manual_seed(42)
    initial=[(p.a.detach()>=0,p.b.detach()>=0) for p in paths]
    if action and not fit: raise ValueError('action QAT needs fit captures')
    history=[]
    for step in range(1,steps+1):
        idx=torch.randperm(w.shape[0],device=w.device,generator=gen)[:batch]
        opt.zero_grad(set_to_none=True)
        cs=[p.components(mode) for p in paths]
        loss=ratio(sum(weight(c,idx) for c in cs),w[idx].float())
        if action:
            f=[]; back=[]
            for x,g,y,dx in fit:
                pairs=[actions(c,x,g) for c in cs]
                f.append(ratio(sum(v[0] for v in pairs),y))
                back.append(ratio(sum(v[1] for v in pairs),dx))
            loss=.2*loss+.4*torch.stack(f).mean()+.4*torch.stack(back).mean()
        if not torch.isfinite(loss): raise ValueError('nonfinite QAT loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        opt.step()
        if step==1 or step%20==0 or step==steps:
            with torch.no_grad():
                flips=[float(((p.a>=0)!=a).sum()+((p.b>=0)!=b).sum())/(a.numel()+b.numel()) for p,(a,b) in zip(paths,initial)]
            history.append({'step':step,'batch_objective_before_update':float(loss.detach()),'sign_change_fraction':flips})
            print('QAT',mode,'action',action,'step',step,'loss',history[-1]['batch_objective_before_update'],'flips',flips,flush=True)
    return history
