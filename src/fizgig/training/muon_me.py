"""Standalone research adaptation of Fizgig BatchedMuon (b299a617).
Newton-Schulz and LR adjustment adapted from PyTorch Muon (BSD-3-Clause,
Copyright (c) 2016- Facebook, Inc. and PyTorch contributors).
Experimental trainer integration. Budget bounds estimated scratch, not total process memory.
"""
import math
import torch

def ns(x, coeff, steps, eps):
    a,b,c=coeff
    x=x.bfloat16(); tall=x.size(-2)>x.size(-1)
    if tall: x=x.transpose(-2,-1)
    x=x/x.norm(dim=(-2,-1),keepdim=True).clamp(min=eps)
    for _ in range(steps):
        gram=x@x.transpose(-2,-1)
        upd=torch.baddbmm(gram,gram,gram,beta=b,alpha=c)
        x=torch.baddbmm(x,upd,x,beta=a)
    return x.transpose(-2,-1) if tall else x

def adjusted_lr(lr, mode, shape):
    a,b=shape
    if mode in (None,'original'): ratio=math.sqrt(max(1,a/b))
    elif mode=='match_rms_adamw': ratio=.2*math.sqrt(max(a,b))
    elif mode=='spectral_unclamped': ratio=math.sqrt(a/b)
    else: raise ValueError(mode)
    return lr*ratio

def scratch_per_matrix(shape, itemsize, nesterov=True):
    """Conservative explicit tensor allocation model; excludes backend workspace.
    Counts stacked input, lookahead, BF16 conversion and intermediates,
    cast output, small Grams, and norm vectors. Intentionally overcounts
    non-overlapping lifetimes; not a measured peak or a guaranteed device limit.
    """
    m,n=shape; d=min(m,n)
    return m*n*(itemsize*(2+int(nesterov))+8)+12*d*d+32

class MuonME(torch.optim.Optimizer):
    def __init__(self,params,lr=.001,weight_decay=.1,momentum=.95,nesterov=True,
                 ns_coefficients=(3.4445,-4.775,2.0315),eps=1e-7,ns_steps=5,
                 adjust_lr_fn=None,chunk_memory_mib=128):
        if not math.isfinite(chunk_memory_mib) or chunk_memory_mib<0: raise ValueError('invalid chunk budget')
        if adjust_lr_fn not in (None,'original','match_rms_adamw','spectral_unclamped'): raise ValueError(adjust_lr_fn)
        super().__init__(params,dict(lr=lr,weight_decay=weight_decay,momentum=momentum,nesterov=nesterov,ns_coefficients=ns_coefficients,eps=eps,ns_steps=ns_steps,adjust_lr_fn=adjust_lr_fn,chunk_memory_mib=chunk_memory_mib))
        if any(p.ndim!=2 for g in self.param_groups for p in g['params']): raise ValueError('2D only')
    def load_state_dict(self,state_dict):
        super().load_state_dict(state_dict)
        for g in self.param_groups: g.setdefault('chunk_memory_mib',self.defaults['chunk_memory_mib'])
    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad(): loss=closure()
        for g in self.param_groups:
            lr,wd,mom=float(g['lr']),float(g['weight_decay']),float(g['momentum'])
            ps=[p for p in g['params'] if p.grad is not None]
            if not ps: continue
            by_shape={}
            for p in ps:
                if p.grad.is_sparse: raise RuntimeError('sparse gradients unsupported')
                by_shape.setdefault((tuple(p.shape),p.device,p.dtype),[]).append(p)
            budget=g['chunk_memory_mib']
            if not math.isfinite(budget) or budget<0: raise ValueError('invalid chunk budget')
            for (shape,dev,dtype),group in by_shape.items():
                cost=scratch_per_matrix(shape,group[0].element_size(),g['nesterov'])
                chunk=len(group) if budget==0 else max(1,min(len(group),int(budget*2**20)//cost))
                alr=adjusted_lr(lr,g['adjust_lr_fn'],shape)
                for start in range(0,len(group),chunk):
                    sub=group[start:start+chunk]; grads=[p.grad for p in sub]; bufs=[]
                    for p,grad in zip(sub,grads):
                        st=self.state[p]
                        if 'momentum_buffer' not in st: st['momentum_buffer']=torch.zeros_like(grad,memory_format=torch.preserve_format)
                        bufs.append(st['momentum_buffer'])
                    torch._foreach_lerp_(bufs,grads,1-mom)
                    ups=torch._foreach_lerp(grads,bufs,mom) if g['nesterov'] else bufs
                    if wd!=0: torch._foreach_mul_(sub,1-lr*wd)
                    ortho=ns(torch.stack(ups),g['ns_coefficients'],int(g['ns_steps']),float(g['eps']))
                    torch._foreach_add_(sub,list(ortho.to(dtype).unbind(0)),alpha=-alr)
                    # Explicitly drop old temporary tensors before next chunk.
                    del ortho,ups
        return loss
