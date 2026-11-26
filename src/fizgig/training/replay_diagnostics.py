"""CPU research helpers; no trainer/selector dependencies.
Hashes describe the supplied live objects, not proof of internal forward use.
"""
import hashlib
import json
import math
import struct
import numpy as np
from .replay_evaluator import _array


def localized(delta, *, fraction=.01):
    """Prediction coordinates, NOT semantic features. Top-k uses ceil(fraction*N)."""
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('invalid top fraction')
    d = _array(delta).reshape(-1)
    e = d*d
    if not np.isfinite(e).all():
        raise ValueError('nonfinite energy')
    n = len(e); k = max(1, math.ceil(n*fraction)); total = float(e.sum())
    if not math.isfinite(total): raise ValueError('energy overflow')
    return {'max_abs':float(np.abs(d).max()), 'top_fraction':fraction,
            'top_k':k, 'n':n, 'top_energy_share':float(np.partition(e,n-k)[n-k:].sum()/total) if total else 0.,
            'zero_energy':total==0.}


def depth_shares(shares):
    """Krea 2 28-block reporting, not a functional depth partition."""
    if set(shares)!=set(range(28)) or any(not math.isfinite(v) or v<0 for v in shares.values()):
        raise ValueError('expected 28 finite nonnegative squared movement values')
    total=math.fsum(shares.values())
    groups={'early':range(0,9),'middle':range(9,19),'late':range(19,28)}
    return {'total_squared':total, 'block_squared':dict(sorted(shares.items())),
            'depth_squared':{n:math.fsum(shares[i] for i in ids) for n,ids in groups.items()},
            'depth_fraction':{n:math.fsum(shares[i] for i in ids)/total if total else None for n,ids in groups.items()}}


def digest(value):
    """Typed length-framed SHA256 for dense tensors, arrays and nested state.
    GPU tensors require a live device-to-host copy. Unsupported types fail closed.
    """
    h=hashlib.sha256()
    def field(tag,data):
        tag=tag.encode();h.update(struct.pack('>Q',len(tag)));h.update(tag);h.update(struct.pack('>Q',len(data)));h.update(data)
    def visit(v):
        if v is None: field('none',b'')
        elif isinstance(v,(bool,int,float,str)):
            field(type(v).__name__,json.dumps(v,allow_nan=False).encode())
        elif isinstance(v,bytes):field('bytes',v)
        elif isinstance(v,np.ndarray):
            if v.dtype.kind not in 'biufc':raise TypeError('unsupported array dtype')
            a=np.ascontiguousarray(v)
            field('numpy',json.dumps([v.dtype.str,list(v.shape)]).encode());field('raw',a.tobytes())
        elif type(v).__module__.startswith('torch'):
            import torch
            if not isinstance(v,torch.Tensor) or v.layout!=torch.strided or v.is_quantized:
                raise TypeError('unsupported torch state')
            a=v.detach().cpu().contiguous()
            field('torch',json.dumps([str(a.dtype),list(a.shape),str(v.device)]).encode());field('raw',a.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,(tuple,list)):
            field(type(v).__name__,str(len(v)).encode())
            for x in v:visit(x)
        elif isinstance(v,dict):
            field('dict',str(len(v)).encode())
            for k in sorted(v,key=digest):visit(k);visit(v[k])
        else:raise TypeError(f'unsupported state: {type(v).__name__}')
    visit(value);return h.hexdigest()


def guarded_forward(predict, inputs, *, expected_input_hash, read_state):
    """Fail closed on final observable state changes; caller owns restoration.
    Does not detect transient mutations undone within the forward or prove internal
    consumption of the inputs. read_state must enumerate all mutable state of interest.
    """
    if digest(inputs)!=expected_input_hash:raise ValueError('input manifest mismatch')
    before=digest(read_state())
    try:
        result=predict(inputs)
    finally:
        if digest(inputs)!=expected_input_hash:raise RuntimeError('probe input mutated')
        if digest(read_state())!=before:raise RuntimeError('probe state mutated')
    return result
