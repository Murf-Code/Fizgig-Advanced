"""Opt-in Muon-ME state storage; FP32 arithmetic, eager only.

Experimental. Budget is an explicit-tensor estimate, not a CUDA allocation
or residency guarantee. Default Muon-ME is deliberately unchanged.
"""
import math
from collections import defaultdict
from numbers import Real
import torch
from .muon_me import MuonME, ns, adjusted_lr, scratch_per_matrix

MODES = ('bf16', 'int8', 'splitmax')
VERSION = 1
MIB = 2**20


def _blocks(shape, block):
    return (math.prod(shape) + block - 1) // block


def state_bytes(shape, mode, block):
    n = math.prod(shape)
    if mode == 'bf16':
        return 2*n
    k = _blocks(shape, block)
    return k*(block + 4 + (5 if mode == 'splitmax' else 0))


@torch.no_grad()
def encode(x, mode, block, *, check_finite=True):
    if check_finite and not torch.isfinite(x).all():
        raise ValueError('nonfinite momentum')
    if mode == 'bf16':
        buf = x.to(torch.bfloat16)
        if check_finite and not torch.isfinite(buf).all():
            raise ValueError('BF16 momentum overflow')
        return {'buffer': buf}
    n = x.numel()
    f = x.reshape(-1)
    pad = (-n) % block
    f = torch.nn.functional.pad(f, (0, pad)) if pad else f.clone()
    f = f.reshape(-1, block)
    st = {}
    if mode == 'splitmax':
        ix = f.abs().argmax(1, keepdim=True)
        st['outlier'] = f.gather(1, ix).flatten()
        st['index'] = ix.flatten().to(torch.uint8)
        f.scatter_(1, ix, 0.)
    scale = f.abs().amax(1).div_(127)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale.clamp_min(torch.finfo(torch.float32).tiny))
    f.div_(scale[:, None]).round_().clamp_(-127, 127)
    st.update(q=f.to(torch.int8).flatten(), scale=scale)
    return st


def decode(st, shape, mode, block):
    if mode == 'bf16':
        return st['buffer'].float()
    f = st['q'].reshape(-1, block).float().mul_(st['scale'][:, None])
    if mode == 'splitmax':
        f.scatter_(1, st['index'].long()[:, None], st['outlier'][:, None])
    return f.flatten()[:math.prod(shape)].reshape(shape)


def validate(st, shape, mode, block):
    if not isinstance(st, dict):
        raise ValueError('invalid momentum record')
    def check(name, dtype, dims):
        v = st.get(name)
        if not torch.is_tensor(v) or v.layout != torch.strided or v.dtype != dtype or tuple(v.shape) != tuple(dims):
            raise ValueError(f'invalid momentum {name}')
        return v
    if mode == 'bf16':
        if set(st) != {'buffer'} or not torch.isfinite(check('buffer', torch.bfloat16, shape)).all():
            raise ValueError('invalid BF16 momentum')
        return
    k = _blocks(shape, block)
    if set(st) != ({'q','scale','outlier','index'} if mode == 'splitmax' else {'q','scale'}):
        raise ValueError('invalid packed state keys')
    q = check('q', torch.int8, (k*block,))
    s = check('scale', torch.float32, (k,))
    if q.device != s.device or not torch.isfinite(s).all() or not (s > 0).all() or (q == -128).any() or q[math.prod(shape):].count_nonzero():
        raise ValueError('invalid quantization scale/code/padding')
    if mode == 'splitmax':
        ix = check('index', torch.uint8, (k,))
        out = check('outlier', torch.float32, (k,))
        if ix.device != q.device or out.device != q.device or not torch.isfinite(out).all():
            raise ValueError('invalid outlier payload')
        j = ix.long()
        if (j >= block).any() or j[-1] >= math.prod(shape)-(k-1)*block or q.reshape(k,block).gather(1,j[:,None]).count_nonzero():
            raise ValueError('invalid outlier index/code')


class CompressedMuonME(MuonME):
    """State compression occurs AFTER current FP32 momentum is used for update.

    The optimizer stores only compressed momentum. Checkpoints restore packed
    tensors directly without Optimizer.load_state_dict's floating dtype cast.
    Identical parameter order is required; shape and optional param_names checked.
    """
    def __init__(self, params, *, momentum_storage='bf16', momentum_block=64, **kwargs):
        if momentum_storage not in MODES or not isinstance(momentum_block, int) or isinstance(momentum_block, bool) or momentum_block not in (64,128,256):
            raise ValueError('unsupported momentum format')
        super().__init__(params, **kwargs)
        self.defaults.update(momentum_storage=momentum_storage, momentum_block=momentum_block, momentum_format=VERSION)
        for g in self.param_groups:
            g.update(momentum_storage=momentum_storage, momentum_block=momentum_block, momentum_format=VERSION)
            if any(p.dtype != torch.float32 or p.numel()==0 for p in g['params']):
                raise ValueError('compressed Muon-ME requires nonempty FP32 factors')
            self._validate_group(g)

    @staticmethod
    def _validate_group(g):
        for k in ('lr','weight_decay','momentum','eps','chunk_memory_mib'):
            if not isinstance(g[k], Real) or not math.isfinite(g[k]):
                raise ValueError('nonfinite optimizer option '+k)
        if g['lr'] < 0 or g['weight_decay'] < 0 or not 0 <= g['momentum'] < 1 or g['eps'] <= 0 or g['chunk_memory_mib'] <= 0:
            raise ValueError('invalid compressed optimizer options; positive chunk budget required')
        if not isinstance(g['ns_steps'], int) or g['ns_steps'] <= 0 or len(g['ns_coefficients']) != 3 or not all(math.isfinite(c) for c in g['ns_coefficients']):
            raise ValueError('invalid Newton-Schulz options')
        if g['adjust_lr_fn'] not in (None,'original','match_rms_adamw','spectral_unclamped') or not isinstance(g['nesterov'], bool):
            raise ValueError('invalid Muon options')

    def load_state_dict(self, sd):
        if set(sd) != {'state','param_groups'} or len(sd['param_groups']) != len(self.param_groups):
            raise ValueError('invalid compressed checkpoint schema')
        mapping = {}; groups = []
        for old, live in zip(sd['param_groups'], self.param_groups):
            self._validate_group(old)
            if any(old.get(k) != live.get(k) for k in ('momentum_storage','momentum_block','momentum_format','param_names')) or len(old['params']) != len(live['params']):
                raise ValueError('incompatible compressed momentum format or parameter layout')
            for key,p in zip(old['params'],live['params']):
                if key in mapping:
                    raise ValueError('duplicate checkpoint parameter')
                mapping[key] = (p, live)
            groups.append({**old,'params':live['params']})
        if not set(sd['state']) <= set(mapping):
            raise ValueError('unknown checkpoint parameter')
        for key, st in sd['state'].items():
            if not st: continue
            p,g = mapping[key]
            if tuple(st.get('shape', ())) != tuple(p.shape):
                raise ValueError('checkpoint factor shape mismatch')
            validate({k:v for k,v in st.items() if k!='shape'},p.shape,g['momentum_storage'],g['momentum_block'])
        # Stage directly in the original dtype; never send packed state to the
        # base loader. Old and new packed states can overlap during restoration.
        new = defaultdict(dict)
        for key, st in sd['state'].items():
            if not st: continue
            p,_ = mapping[key]
            new[p] = {k:(v.to(device=p.device,copy=True) if torch.is_tensor(v) else tuple(v)) for k,v in st.items()}
        self.__setstate__({'state':new,'param_groups':groups})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        plans = []
        for g in self.param_groups:
            self._validate_group(g)
            by = defaultdict(list)
            for p in g['params']:
                if p.grad is None: continue
                if p.grad.is_sparse or p.grad.dtype != torch.float32 or p.grad.shape != p.shape:
                    raise ValueError('unsupported compressed Muon gradient')
                by[(tuple(p.shape),p.device)].append(p)
            for (shape,_), ps in by.items():
                # Include FP32 reconstruction, codec temporaries, padding and
                # replacement payload in addition to baseline NS scratch.
                cost = scratch_per_matrix(shape,4,g['nesterov']) + 48*(_blocks(shape,g['momentum_block'])*g['momentum_block']) + state_bytes(shape,g['momentum_storage'],g['momentum_block'])
                chunk = int(g['chunk_memory_mib']*MIB)//cost
                if chunk < 1:
                    raise MemoryError('one compressed Muon matrix exceeds estimated chunk budget')
                plans.append((g,shape,ps,chunk))
        for g,shape,ps,chunk in plans:
            mode,b = g['momentum_storage'],g['momentum_block']
            for start in range(0,len(ps),chunk):
                sub=ps[start:start+chunk]; grads=[p.grad for p in sub]; bufs=[]; packed=[]
                for p,grad in zip(sub,grads):
                    st=self.state.get(p,{})
                    m=decode(st,p.shape,mode,b) if st else torch.zeros_like(p)
                    m.lerp_(grad,1-g['momentum']); bufs.append(m)
                # One finite/overflow check per chunk, not a host sync for every factor.
                maxima=torch.stack(torch._foreach_norm(bufs,float('inf')))
                limit=torch.finfo(torch.bfloat16).max if mode=='bf16' else torch.finfo(torch.float32).max
                if not (torch.isfinite(maxima) & (maxima<=limit)).all():
                    raise ValueError('nonfinite/overflow momentum; abort update, do not retry')
                del maxima
                packed=[{**encode(m,mode,b,check_finite=False),'shape':tuple(p.shape)} for p,m in zip(sub,bufs)]
                ups=torch._foreach_lerp(grads,bufs,g['momentum']) if g['nesterov'] else bufs
                ortho=ns(torch.stack(ups),g['ns_coefficients'],g['ns_steps'],g['eps'])
                if g['weight_decay']: torch._foreach_mul_(sub,1-g['lr']*g['weight_decay'])
                torch._foreach_add_(sub,list(ortho.float().unbind(0)),alpha=-adjusted_lr(g['lr'],g['adjust_lr_fn'],shape))
                for p,st in zip(sub,packed): self.state[p]=st
                del ortho,ups,bufs,packed,m,st
        return loss
