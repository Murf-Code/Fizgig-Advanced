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
                 adjust_lr_fn=None,chunk_memory_mib=128,auto_memory_budget=False,
                 memory_reserve_mib=768,memory_growth_steps=3):
        if not math.isfinite(chunk_memory_mib) or chunk_memory_mib<0: raise ValueError('invalid chunk budget')
        if adjust_lr_fn not in (None,'original','match_rms_adamw','spectral_unclamped'): raise ValueError(adjust_lr_fn)
        super().__init__(params,dict(lr=lr,weight_decay=weight_decay,momentum=momentum,nesterov=nesterov,ns_coefficients=ns_coefficients,eps=eps,ns_steps=ns_steps,adjust_lr_fn=adjust_lr_fn,chunk_memory_mib=chunk_memory_mib))
        if any(p.ndim!=2 for g in self.param_groups for p in g['params']): raise ValueError('2D only')
        if not isinstance(auto_memory_budget,bool): raise ValueError('auto_memory_budget must be boolean')
        self._auto_budget_enabled=auto_memory_budget
        self._memory_reserve_mib=memory_reserve_mib
        self._memory_growth_steps=memory_growth_steps
        self._memory_policies={}
        self.memory_plans={}
        if auto_memory_budget:
            from .memory_budget import MemoryBudget
            MemoryBudget(memory_reserve_mib,memory_growth_steps)
            if any(g['chunk_memory_mib']<=0 for g in self.param_groups):
                raise ValueError('automatic budgeting requires a positive chunk ceiling')

    def _memory_item(self,p,g):
        cost=scratch_per_matrix(p.shape,p.element_size(),g['nesterov'])
        pending=0 if 'momentum_buffer' in self.state.get(p,{}) else p.numel()*p.element_size()
        return cost,pending

    def _memory_free_bytes(self,device):
        if device.type!='cuda':raise RuntimeError('automatic budgeting requires a CUDA free-memory snapshot')
        return torch.cuda.mem_get_info(device)[0]

    def _memory_caps(self):
        """Plan every active group before the first state/parameter mutation."""
        if not self._auto_budget_enabled:return {}
        from .memory_budget import MemoryBudget,ScratchGroup,MIB
        requests={};pending={};self.memory_plans={}
        for g in self.param_groups:
            ceiling=int(g['chunk_memory_mib']*MIB)
            for p in g['params']:
                if p.grad is None:continue
                if p.grad.is_sparse:raise RuntimeError('sparse gradient')
                cost,missing=self._memory_item(p,g)
                key=(id(g),cost,ceiling)
                by=requests.setdefault(p.device,{})
                by[key]=by.get(key,0)+1
                pending[p.device]=pending.get(p.device,0)+missing
        caps={};plans={}
        for device,by in requests.items():
            policy=self._memory_policies.get(device)
            if policy is None:
                policy=MemoryBudget(self._memory_reserve_mib,self._memory_growth_steps)
                self._memory_policies[device]=policy
            groups=[ScratchGroup(key,count,cost,ceiling) for (key,cost,ceiling),count in by.items()]
            plan=policy.plan(self._memory_free_bytes(device),groups,pending_state_bytes=pending[device])
            plans[str(device)]=plan
            for req,cap in zip(groups,plan.caps):
                k=(req.key,device)
                caps[k]=min(caps.get(k,cap),cap)
        self.memory_plans=plans
        return caps
    def load_state_dict(self,state_dict):
        if any(g.get('momentum_storage', 'fp32') != 'fp32' for g in state_dict['param_groups']):
            raise ValueError('compressed Muon-ME checkpoint requires its original momentum storage mode')
        super().load_state_dict(state_dict)
        for g in self.param_groups: g.setdefault('chunk_memory_mib',self.defaults['chunk_memory_mib'])
    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad(): loss=closure()
        caps=self._memory_caps()
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
                cap=caps.get((id(g),dev),int(budget*2**20))
                chunk=len(group) if budget==0 else max(1,min(len(group),cap//cost))
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
                    del ortho,ups
        return loss
