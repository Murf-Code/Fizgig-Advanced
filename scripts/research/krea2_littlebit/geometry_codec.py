"""Frozen-base geometry experiments: Hadamard, fitted scales and HQQ residuals.

HQQ reuses Fizgig's rintic-13 implementation (MiniMax H3, issue #102).
All payloads are uint8, with deterministic shape-dependent tile boundaries.
No rotation or residual correction is applied to trainable LoRA parameters.
"""
import math
import torch
import torch.nn.functional as F

TILE = 128
HAD = 64
HQQ_GROUP = 8
UNIFORM = {"q4-had", "q5-had", "q6-had", "q4-opt", "q6-opt", "q4-opt-had", "q6-opt-had"}
HQQ = {"hqq", "hqq-had"}
HYBRID = {"lb2-hqq", "lb2-hqq-had"}
EXTRA = UNIFORM | HQQ | HYBRID


def hadamard(x, size=HAD):
    """Normalized Sylvester butterfly on contiguous groups of the final axis."""
    if size <= 0 or size & (size-1) or x.shape[-1] % size:
        raise ValueError("Hadamard requires a divisible power-of-two group")
    v = x.float().reshape(-1, size)
    step = 1
    while step < size:
        t = v.reshape(-1, size//(2*step), 2, step)
        a, b = t[:, :, 0], t[:, :, 1]
        v = torch.stack((a+b, a-b), dim=2).reshape(-1, size)
        step *= 2
    return (v / math.sqrt(size)).reshape(x.shape).to(x.dtype)


def compatible(name, m, n, rank):
    if name in UNIFORM:
        return n % HAD == 0
    if name in HQQ | HYBRID:
        return n % (2*HQQ_GROUP) == 0 and (not name.endswith('-had') or n % HAD == 0) and (name not in HYBRID or rank <= min(m,n))
    if name in ('q4','q6'):
        return n % 64 == 0
    return rank <= min(m,n) and (not name.endswith('-had') or rank % HAD == 0)


def _bytes(t):
    return t.contiguous().view(torch.uint8).flatten()


def fitted_codes(w, bits, optimize):
    """Fit symmetric group scales; retain best actual BF16-scale L2 error."""
    levels = (1 << (bits-1))-1
    scale = (w.abs().amax(-1)/levels).bfloat16()
    best = None
    for _ in range(9 if optimize else 1):
        safe = torch.where(scale == 0, torch.ones_like(scale), scale).float()
        q = (w/safe[...,None]).round().clamp(-levels, levels)
        err = (w-q*scale.float()[...,None]).square().sum(-1)
        if best is None:
            best = (q, scale, err)
        else:
            keep = err < best[2]
            best = (torch.where(keep[...,None],q,best[0]), torch.where(keep,scale,best[1]), torch.minimum(err,best[2]))
        scale = ((w*q).sum(-1)/q.square().sum(-1).clamp_min(1)).clamp_min(0).bfloat16()
    return (best[0]+levels).byte(), best[1]


def _hqq_api():
    from fizgig.minimax.hqq4 import quantize_hqq4, dequantize_hqq4
    return quantize_hqq4, dequantize_hqq4


def _tile_bytes(spec, rows):
    n = spec['shape'][1]
    if spec['codec'] in UNIFORM:
        return rows*(n*int(spec['codec'][1])//8+2*(n//spec['group']))
    return rows*(n//2+2*(n//HQQ_GROUP)+16)


def _encode_tiles(weight_rows, spec, pack):
    m,n = spec['shape']
    name = spec['codec']
    payload = torch.empty(_tile_bytes(spec,m),dtype=torch.uint8,device='cpu')
    cursor = 0
    for i in range(0,m,TILE):
        j = min(i+TILE,m)
        w = weight_rows(i,j).float()
        if name.endswith('-had'):
            w = hadamard(w)
        if name in UNIFORM:
            bits = int(name[1]); g = spec['group']
            q, scale = fitted_codes(w.reshape(j-i,n//g,g),bits,'-opt' in name)
            data = torch.cat((pack(q,bits),_bytes(scale)))
        else:
            quantize,_ = _hqq_api()
            q,s,z,meta = quantize(w.bfloat16(),group_size=HQQ_GROUP,compute_dtype=torch.bfloat16,device=w.device)
            if (q.dtype,s.dtype,z.dtype,meta.dtype)!=(torch.uint8,torch.uint8,torch.uint8,torch.int32):
                raise ValueError('unexpected HQQ payload dtype')
            if q.numel()!=(j-i)*n//2 or s.numel()!=(j-i)*n//HQQ_GROUP or z.numel()!=s.numel() or meta.shape!=(j-i,4):
                raise ValueError('unexpected HQQ payload shape')
            data = torch.cat(tuple(_bytes(t) for t in (q,s,z,meta)))
        size = _tile_bytes(spec,j-i)
        if data.numel()!=size:
            raise ValueError('tile payload size mismatch')
        payload[cursor:cursor+size] = data.cpu()
        cursor += size
    return payload


def encode(weight, name, rank, group, pack, base_encode, factors):
    m,n = weight.shape
    if name not in EXTRA or not compatible(name,m,n,rank) or group != 64:
        raise ValueError('unsupported geometry codec dimensions/group')
    if name in HYBRID:
        base, bs = base_encode(weight,'littlebit2',rank=rank,group=group)
        a,b = factors(base,bs,torch.float32)
        rs = {'codec':'hqq-had' if name.endswith('-had') else 'hqq','shape':[m,n]}
        # Residual is against the ACTUAL decoded binary factors, not the SVD.
        residual = _encode_tiles(lambda i,j: weight[i:j].float()-b[i:j]@a,rs,pack)
        return torch.cat((base.cpu(),residual)), {'codec':name,'shape':[m,n],'base':bs,'residual':rs,'base_bytes':base.numel()}
    spec = {'codec':name,'shape':[m,n]}
    if name in UNIFORM:
        spec['group'] = group
    return _encode_tiles(lambda i,j:weight[i:j],spec,pack),spec


def validate(p,s,base_validate):
    m,n = s['shape']; name = s['codec']
    if name not in EXTRA or min(m,n)<=0 or p.dtype!=torch.uint8 or p.ndim!=1:
        raise ValueError('invalid geometry payload')
    if name in HYBRID:
        bs,rs,k = s['base'],s['residual'],s['base_bytes']
        if bs['codec']!='littlebit2' or rs['codec']!=('hqq-had' if name.endswith('-had') else 'hqq') or bs['shape']!=s['shape'] or rs['shape']!=s['shape'] or not 0<k<p.numel():
            raise ValueError('invalid hybrid metadata')
        base_validate(p[:k],bs)
        validate(p[k:],rs,base_validate)
        return
    if not compatible(name,m,n,1) or (name in UNIFORM and s.get('group')!=64):
        raise ValueError('invalid geometry dimensions')
    if p.numel()!=_tile_bytes(s,m):
        raise ValueError('geometry payload length mismatch')


def _decode_tile(p,s,i,j,dtype,unpack):
    n = s['shape'][1]; r = j-i; name = s['codec']
    offset = _tile_bytes(s,i); t = p[offset:offset+_tile_bytes(s,r)]
    if name in UNIFORM:
        bits = int(name[1]); g=s['group']; cut=r*n*bits//8
        q = unpack(t[:cut],r*n,bits).reshape(r,n//g,g)
        scale = t[cut:].clone().view(torch.bfloat16).reshape(r,n//g,1).to(dtype)
        return ((q-((1<<(bits-1))-1)).to(dtype)*scale).reshape(r,n)
    _,dequantize = _hqq_api()
    cut=r*n//2; groups=r*n//HQQ_GROUP
    q=t[:cut].reshape(groups//2,HQQ_GROUP).contiguous()
    scale=t[cut:cut+groups].reshape(groups,1).contiguous()
    zero=t[cut+groups:cut+2*groups].reshape(groups,1).contiguous()
    meta=t[cut+2*groups:].clone().view(torch.int32).reshape(r,4)
    return dequantize(q,scale,zero,meta,(r,n),dtype,group_size=HQQ_GROUP)


def forward(x,p,s,unpack,factors):
    name=s['codec']
    if name in HYBRID:
        k=s['base_bytes']; a,b=factors(p[:k],s['base'],x.dtype)
        return F.linear(F.linear(x,a),b)+forward(x,p[k:],s['residual'],unpack,factors)
    m,n=s['shape']
    z=hadamard(x) if name.endswith('-had') else x
    return torch.cat([F.linear(z,_decode_tile(p,s,i,min(i+TILE,m),z.dtype,unpack)) for i in range(0,m,TILE)],dim=-1)


def backward(dy,p,s,unpack,factors):
    name=s['codec']
    if name in HYBRID:
        k=s['base_bytes']; a,b=factors(p[:k],s['base'],dy.dtype)
        return (dy@b)@a+backward(dy,p[k:],s['residual'],unpack,factors)
    m,n=s['shape']
    # Float accumulation between tiles avoids repeated BF16 rounding.
    dx=torch.zeros((*dy.shape[:-1],n),device=dy.device,dtype=torch.float32)
    for i in range(0,m,TILE):
        j=min(i+TILE,m)
        dx.add_((dy[...,i:j]@_decode_tile(p,s,i,j,dy.dtype,unpack)).float())
    if name.endswith('-had'):
        dx=hadamard(dx)
    return dx.to(dy.dtype)
