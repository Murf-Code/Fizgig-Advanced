"""Local research prototypes. No repository integration or CUDA validation."""
import math
from collections import defaultdict
from dataclasses import dataclass
from contextlib import contextmanager
import torch

MIB=2**20

def pack8(x, block=256, *, stochastic=False, generator=None):
    if block <= 0 or x.numel()==0 or not x.is_floating_point():
        raise ValueError('invalid tensor/block')
    f=x.detach().float().reshape(-1)
    if not torch.isfinite(f).all(): raise ValueError('nonfinite input')
    if f.numel()%block: f=torch.nn.functional.pad(f,(0,(-f.numel())%block))
    f=f.view(-1,block)
    s=f.abs().amax(1)/127
    s=torch.where(s==0,torch.ones_like(s),s.clamp_min(torch.finfo(torch.float32).tiny))
    z=f/s[:,None]
    if stochastic:
        if generator is None: raise ValueError('explicit generator required')
        lo=z.floor(); z=lo+(torch.rand(z.shape,device=z.device,generator=generator)<(z-lo))
    else: z.round_()
    return z.clamp_(-127,127).to(torch.int8).flatten(),s

def unpack8(q,s,shape,block=256):
    return (q.reshape(-1,block).float().mul_(s[:,None])).flatten()[:math.prod(shape)].reshape(shape)

def validate8(q,s,shape,block):
    n=math.prod(shape); k=(n+block-1)//block
    if q.dtype!=torch.int8 or s.dtype!=torch.float32 or q.shape!=(k*block,) or s.shape!=(k,):
        raise ValueError('packed state shape/dtype mismatch')
    if not torch.isfinite(s).all() or not (s>0).all() or (q==-128).any():
        raise ValueError('invalid code or scale')
    if q[n:].count_nonzero(): raise ValueError('nonzero padded tail')

class PackedMomentum(torch.optim.Optimizer):
    """Test harness for state storage and restoration, NOT Muon or a trainer optimizer.
    Uses EMA SGD to isolate codec and loader behavior. Positional mapping is checked
    against explicit stable parameter names supplied by caller. No GPU atomicity claim.
    """
    def __init__(self,named_params,lr=.01,beta=.95,block=256):
        pairs=list(named_params); names=[n for n,p in pairs]
        if not names or len(set(names))!=len(names) or block<=0 or not 0<=beta<1: raise ValueError('bad config')
        if any(p.dtype!=torch.float32 for n,p in pairs): raise ValueError('FP32 only')
        super().__init__([dict(params=[p for n,p in pairs],param_names=names)],dict(lr=lr,beta=beta,block=block,codec_version=1))
    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None: continue
                if p.grad.is_sparse or not torch.isfinite(p.grad).all(): raise ValueError('bad gradient')
                st=self.state[p]; m=unpack8(st['q'],st['scale'],p.shape,g['block']) if 'q' in st else torch.zeros_like(p)
                m.lerp_(p.grad,1-g['beta']); q,s=pack8(m,g['block']); p.add_(m,alpha=-g['lr']); st.update(q=q,scale=s,shape=tuple(p.shape))
    def load_state_dict(self,sd):
        """Validate all records, bypass base-loader floating-point cast entirely.
        Restricted to this harness's known state schema. Stages new packed buffers
        before publishing; old state may overlap (explicit budget required on GPU).
        """
        if set(sd)!= {'state','param_groups'} or len(sd['param_groups'])!=len(self.param_groups): raise ValueError('bad schema')
        mapping={}; groups=[]
        for old,live in zip(sd['param_groups'],self.param_groups):
            if old.get('codec_version')!=1 or old.get('block')!=live['block'] or old.get('param_names')!=live['param_names'] or len(old['params'])!=len(live['params']): raise ValueError('incompatible layout')
            if not math.isfinite(old['lr']) or old['lr']<0 or not 0<=old['beta']<1: raise ValueError('invalid hyperparameters')
            for key,p in zip(old['params'],live['params']):
                if key in mapping: raise ValueError('duplicate ID')
                mapping[key]=(p,old['block'])
            groups.append({**old,'params':live['params']})
        if not set(sd['state'])<=set(mapping): raise ValueError('unknown parameter')
        for key,st in sd['state'].items():
            if not st: continue
            p,b=mapping[key]
            if set(st)!= {'q','scale','shape'} or tuple(st['shape'])!=tuple(p.shape): raise ValueError('invalid state schema')
            validate8(st['q'],st['scale'],p.shape,b)
        new=defaultdict(dict)
        for key,st in sd['state'].items():
            if not st: continue
            p,b=mapping[key]; new[p]={'q':st['q'].to(p.device,copy=True),'scale':st['scale'].to(p.device,copy=True),'shape':tuple(p.shape)}
        self.state=new; self.param_groups=groups

def unique_bytes(tensors):
    seen={}
    for t in tensors:
        st=t.untyped_storage(); key=(str(t.device),st.data_ptr())
        seen[key]=max(seen.get(key,0),st.nbytes())
    return sum(seen.values())

@dataclass(frozen=True)
class Phase:
    name:str
    live_increment:int  # excludes persistent, excludes staging counted below
    staging:int=0

def phase_plan(capacity,persistent,phases,headroom):
    if min(capacity,persistent,headroom)<0 or not phases or any(min(p.live_increment,p.staging)<0 for p in phases): raise ValueError('invalid plan')
    peaks={p.name:persistent+p.live_increment+p.staging for p in phases}
    if len(peaks)!=len(phases): raise ValueError('duplicate phase')
    peak=max(peaks.values()); return {'peaks':peaks,'peak':peak,'fits':peak+headroom<=capacity,'slack':capacity-peak-headroom}

def scratch_plan(available,pending,commitments,margin,ceiling,cost,count):
    if min(available,pending,commitments,margin)<0 or min(ceiling,cost,count)<=0: raise ValueError('invalid budget')
    cap=min(ceiling,max(0,available-pending-commitments-margin))
    if cost>cap: raise MemoryError('single unit exceeds scratch allowance')
    return min(count,cap//cost)

class PeakOwner:
    """Optional exclusive owner of CUDA reset counters. Requires integration of ALL
    existing reset sites. No synchronization; peaks attribute allocator activity by
    host enqueue interval, not precise GPU execution. Disabled on CPU.
    """
    def __init__(self,device):
        self.device=device; self.records=[]; self.whole_alloc=0; self.whole_reserved=0; self.active=False
    def capture(self):
        a=torch.cuda.max_memory_allocated(self.device); r=torch.cuda.max_memory_reserved(self.device)
        self.whole_alloc=max(self.whole_alloc,a);self.whole_reserved=max(self.whole_reserved,r)
        return a,r
    def reset(self):
        self.capture();torch.cuda.reset_peak_memory_stats(self.device)
    @contextmanager
    def phase(self,name):
        if self.active: raise RuntimeError('nested peak scopes unsupported')
        self.reset(); self.active=True
        start=torch.cuda.memory_allocated(self.device)
        try: yield
        finally:
            a,r=self.capture();self.records.append(dict(phase=name,start_allocated=start,peak_allocated=a,peak_reserved=r,end_allocated=torch.cuda.memory_allocated(self.device)));self.active=False
