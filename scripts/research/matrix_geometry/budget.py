"""Conservative research budget: preflight before mutating optimizer state.
Bytes throughout. No CUDA OOM recovery, no device allocator guarantees.
"""
from dataclasses import dataclass
from muon_me import scratch_per_matrix
MIB=2**20
@dataclass(frozen=True)
class Plan:
    scratch_bytes:int
    chunks:tuple
    pending_state_bytes:int
    free_bytes:int
    reserve_bytes:int

def plan(free_bytes, groups, pending_state_bytes=0, reserve_bytes=768*MIB, ceiling_bytes=512*MIB):
    """groups: (number of tensors, estimated scratch per tensor).
    Driver free only (do not credit reserved-but-unused caching allocator bytes).
    Pending_state_bytes covers all not-yet-allocated persistent optimizer states.
    """
    if min(free_bytes,pending_state_bytes,reserve_bytes,ceiling_bytes)<0 or ceiling_bytes==0: raise ValueError('invalid memory budget')
    if any(n<1 or cost<1 for n,cost in groups): raise ValueError('invalid group')
    cap=min(ceiling_bytes,max(0,free_bytes-pending_state_bytes-reserve_bytes))
    if any(cost>cap for _,cost in groups): raise MemoryError('preflight: one matrix cannot fit estimated scratch allowance')
    return Plan(cap,tuple(min(n,cap//cost) for n,cost in groups),pending_state_bytes,free_bytes,reserve_bytes)

class Hysteresis:
    """Shrink immediately; grow after three consecutive sufficient snapshots.
    No mutation of optimizer groups or LR. Store per-device instance.
    """
    def __init__(self,grow_after=3):
        if grow_after<1: raise ValueError('grow_after')
        self.current=None; self.n=0; self.grow_after=grow_after; self.candidate=None
    def choose(self,cap):
        if cap<0: raise ValueError('cap')
        if self.current is None or cap<=self.current:
            self.current=cap; self.n=0; self.candidate=None
        else:
            self.n+=1; self.candidate=cap if self.candidate is None else min(cap,self.candidate)
            if self.n>=self.grow_after:
                self.current=self.candidate; self.n=0; self.candidate=None
        return self.current

from muon_me import MuonME
import torch

class AutoMuonME(MuonME):
    """Single-device prototype. A free-memory callback permits deterministic CPU tests.
    Actual CUDA integration must account for asynchronous external work and allocator.
    """
    def __init__(self,params,*,free_memory=None,reserve_mib=768,ceiling_mib=512,**kw):
        super().__init__(params,**kw)
        self.free_memory=free_memory or (lambda d:torch.cuda.mem_get_info(d)[0])
        self.reserve_bytes=int(reserve_mib*MIB); self.ceiling_bytes=int(ceiling_mib*MIB)
        self.policy=Hysteresis();self.last_plan=None
    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        active=[(g,p) for g in self.param_groups for p in g['params'] if p.grad is not None]
        if not active:return loss
        devices={p.device for _,p in active}
        if len(devices)!=1:raise ValueError('single device prototype')
        groups={};pending=0
        for g,p in active:
            if p.grad.is_sparse:raise RuntimeError('sparse gradient')
            key=(id(g),tuple(p.shape),p.dtype,g['nesterov'])
            n,c=groups.get(key,(0,scratch_per_matrix(p.shape,p.element_size(),g['nesterov'])))
            groups[key]=(n+1,c)
            if 'momentum_buffer' not in self.state.get(p,{}):pending+=p.numel()*p.element_size()
        free=int(self.free_memory(next(iter(devices))))
        p0=plan(free,tuple(groups.values()),pending,self.reserve_bytes,self.ceiling_bytes)
        cap=self.policy.choose(p0.scratch_bytes)
        # A newly active shape may require a larger minimum. Never exceed safe snapshot.
        cap=max(cap,max(c for _,c in groups.values()))
        self.last_plan=plan(free,tuple(groups.values()),pending,self.reserve_bytes,cap)
        old=[g['chunk_memory_mib'] for g in self.param_groups]
        try:
            for g in self.param_groups:g['chunk_memory_mib']=cap/MIB
            super().step()
        finally:
            for g,b in zip(self.param_groups,old):g['chunk_memory_mib']=b
        return loss
