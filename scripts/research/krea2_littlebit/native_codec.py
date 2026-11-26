"""Independent research codecs inspired by binary latent factorization and ITQ.

References: Lee et al., LittleBit (NeurIPS 2025); Lee & Kim, LittleBit-2
(ICML 2026). This is a simplified experiment, not their implementation:
one SVD path, per-latent magnitude estimates, optional joint rotation.
No borrowed Samsung code, SmoothSign, or binary CUDA kernel. Geometry variants
add independent Hadamard, fitted scales and Fizgig's HQQ residual codec.
"""
import fnmatch
import hashlib
import importlib.util
import json
from pathlib import Path
import os
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
import geometry_codec as geometry

FORMAT = "fizgig-independent-compact-v2"
MODULES = ("attn.wq","attn.wk","attn.wv","attn.gate","attn.wo","mlp.up","mlp.gate","mlp.down")
ELIGIBLE = {f"blocks.{i}.{name}.weight" for i in range(28) for name in MODULES}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stock_layer(w, int8):
    layer = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, device="meta")
    scale = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127
    layer.weight = torch.nn.Parameter((w/scale).round().clamp(-127,127).to(torch.int8), requires_grad=False)
    layer.register_buffer("_int8_wscale",scale.reshape(1,-1).float(),persistent=False)
    layer.forward = int8.int8_train_forward.__get__(layer,type(layer))
    return layer


def _pack(values, bits):
    """Bit planes over groups of eight values; supports padding in the last group."""
    v = values.to(torch.uint8).flatten()
    v = F.pad(v,(0,(-len(v))%8)).reshape(-1,8)
    shifts = torch.arange(8,device=v.device,dtype=torch.int32)
    return torch.stack([(((v >> b)&1).int()<<shifts).sum(1).byte()
                        for b in range(bits)],dim=1).flatten()


def _unpack(p, count, bits):
    shifts = torch.arange(8,device=p.device,dtype=torch.int32)
    words = p.reshape(-1,bits).int()
    out = torch.zeros((len(words),8),device=p.device,dtype=torch.int32)
    for b in range(bits):
        out |= ((words[:,b,None] >> shifts)&1)<<b
    return out.flatten()[:count]


def _bytes(t):
    return t.contiguous().view(torch.uint8).flatten()


def _bf(p):
    return p.clone().contiguous().view(torch.bfloat16)


def encode(weight, codec, rank=128, group=64, qat_steps=0):
    if codec in geometry.EXTRA:
        if qat_steps or weight.ndim != 2 or not weight.is_floating_point() or not torch.isfinite(weight).all():
            raise ValueError('geometry codecs require finite floating matrices and no QAT')
        return geometry.encode(weight,codec,rank,group,_pack,_encode_original,factors)
    latent = codec in ('littlebit-had','littlebit2-had')
    if latent:
        codec = codec.removesuffix('-had')
        if rank % geometry.HAD:
            raise ValueError('latent rank must divide into Hadamard groups')
    return _encode_original(weight,codec,rank,group,qat_steps,latent_hadamard=latent)


def _encode_original(weight, codec, rank=128, group=64, qat_steps=0, latent_hadamard=False):
    if qat_steps:
        raise ValueError("local QAT is not implemented in the independent codec")
    if weight.ndim != 2 or not weight.is_floating_point() or not torch.isfinite(weight).all():
        raise ValueError("finite floating matrix required")
    m,n = weight.shape
    spec = {"codec":codec,"shape":[m,n]}
    if codec in ("q4","q6"):
        bits = int(codec[1:])
        if group <= 0 or group%8 or n%group:
            raise ValueError("group must divide input width and be a multiple of 8")
        levels = (1<<(bits-1))-1
        rowbytes = n*bits//8 + 2*(n//group)
        payload = torch.empty((m,rowbytes),dtype=torch.uint8,device=weight.device)
        for start in range(0,m,128):
            w = weight[start:start+128].float().reshape(-1,n//group,group)
            scale = (w.abs().amax(-1)/levels).bfloat16()
            divisor = torch.where(scale==0,torch.ones_like(scale),scale).float()
            q = ((w/divisor[...,None]).round().clamp(-levels,levels)+levels).byte()
            body = _pack(q,bits).reshape(len(w),-1)
            payload[start:start+len(w)] = torch.cat((body,_bytes(scale).reshape(len(w),-1)),1)
        return payload.flatten(),dict(spec,group=group)
    if codec not in ("littlebit","littlebit2","svd") or not 0<rank<=min(m,n):
        raise ValueError("unsupported codec or rank")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        u,s,v = torch.svd_lowrank(weight.float(),q=min(rank+8,m,n),niter=3)
    root = s[:rank].clamp_min(0).sqrt()
    left = u[:,:rank]*root
    right = v[:,:rank]*root
    if codec=="svd":
        return torch.cat((_bytes(right.T.bfloat16()),_bytes(left.bfloat16()))),dict(spec,rank=rank)
    if latent_hadamard and codec=="littlebit":
        left,right = geometry.hadamard(left),geometry.hadamard(right)
    if codec=="littlebit2":
        # Solve orthogonal Procrustes for joint balanced factors, with a
        # deterministic fixed iteration budget. The rotation is not stored.
        z = torch.cat((left,right),0)
        rotation = torch.eye(rank,device=z.device,dtype=z.dtype)
        if latent_hadamard:
            rotation = geometry.hadamard(rotation)
        for _ in range(8):
            target = torch.where(z@rotation>=0,1.,-1.)
            l,_,rh = torch.linalg.svd(z.T@target,full_matrices=False)
            rotation = l@rh
        left,right = left@rotation,right@rotation
    # Sign matrices and separate per-latent magnitude estimates.
    a,b = right.T.contiguous(),left.contiguous()
    scales = torch.cat((a.abs().mean(1),b.abs().mean(0))).bfloat16()
    payload = torch.cat((_pack(a>=0,1),_pack(b>=0,1),_bytes(scales)))
    return payload,dict(spec,rank=rank,latent_hadamard=latent_hadamard)


def validate(p,s):
    if s["codec"] in geometry.EXTRA:
        return geometry.validate(p,s,validate)
    m,n = s["shape"]
    c = s["codec"]
    if min(m,n)<=0 or p.dtype!=torch.uint8 or p.ndim!=1:
        raise ValueError("invalid packed matrix")
    if c in ("q4","q6"):
        g=s["group"]
        if g<=0 or g%8 or n%g:
            raise ValueError("invalid group")
        size=m*(n*int(c[1:])//8+2*(n//g))
    elif c in ("svd","littlebit","littlebit2"):
        r=s["rank"]
        if not 0<r<=min(m,n):
            raise ValueError("invalid rank")
        size=2*r*(m+n) if c=="svd" else (r*n+7)//8+(r*m+7)//8+4*r
    else:
        raise ValueError("unknown codec")
    if p.numel()!=size:
        raise ValueError("packed length mismatch")


def factors(p,s,dtype=torch.float32):
    m,n=s["shape"]
    r=s["rank"]
    if s["codec"]=="svd":
        cut=2*r*n
        return _bf(p[:cut]).reshape(r,n).to(dtype),_bf(p[cut:]).reshape(m,r).to(dtype)
    cut=(r*n+7)//8
    end=cut+(r*m+7)//8
    scales=_bf(p[end:]).to(dtype)
    a=(_unpack(p[:cut],r*n,1).reshape(r,n)*2-1).to(dtype)*scales[:r,None]
    b=(_unpack(p[cut:end],r*m,1).reshape(m,r)*2-1).to(dtype)*scales[None,r:]
    return a,b


def _rows(p,s,start,end,dtype):
    m,n=s["shape"]
    bits=int(s["codec"][1:])
    g=s["group"]
    cut=n*bits//8
    t=p.reshape(m,-1)[start:end]
    q=_unpack(t[:,:cut].contiguous().flatten(),len(t)*n,bits).reshape(len(t),n//g,g)
    scale=_bf(t[:,cut:].contiguous().flatten()).reshape(len(t),n//g,1).to(dtype)
    return ((q-((1<<(bits-1))-1)).to(dtype)*scale).reshape(len(t),n)


class Frozen(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,p,s):
        ctx.save_for_backward(p)
        ctx.spec=s
        if s["codec"] in geometry.EXTRA:
            return geometry.forward(x,p,s,_unpack,factors)
        if s["codec"] not in ("q4","q6"):
            a,b=factors(p,s,x.dtype)
            return F.linear(F.linear(x,a),b)
        m,n=s["shape"]
        return torch.cat([F.linear(x,_rows(p,s,i,min(i+128,m),x.dtype))
                          for i in range(0,m,128)],dim=-1)

    @staticmethod
    def backward(ctx,dy):
        p,=ctx.saved_tensors
        s=ctx.spec
        if s["codec"] in geometry.EXTRA:
            return geometry.backward(dy,p,s,_unpack,factors),None,None
        if s["codec"] not in ("q4","q6"):
            a,b=factors(p,s,dy.dtype)
            dx=(dy@b)@a
        else:
            m,n=s["shape"]
            dx=torch.zeros((*dy.shape[:-1],n),device=dy.device,dtype=torch.float32)
            for i in range(0,m,128):
                j=min(i+128,m)
                dx.add_((dy[...,i:j]@_rows(p,s,i,j,dy.dtype)).float())
            dx=dx.to(dy.dtype)
        return dx,None,None


def linear(x,p,s):
    return Frozen.apply(x,p,s)


def _forward(self,x):
    y=linear(x,self.weight.data,self._compact_spec)
    return y if self.bias is None else y+self.bias.to(y.dtype)


def install(module,p,s):
    validate(p,s)
    if list(s["shape"]) != [module.out_features,module.in_features]:
        raise ValueError("Linear dimensions do not match compact matrix")
    module.weight=torch.nn.Parameter(p,requires_grad=False)
    module._compact_spec=s
    module.forward=_forward.__get__(module,type(module))


def manifest(f):
    return json.loads((f.metadata() or {}).get(FORMAT,"{}"))


def convert(source,dest,patterns,codec,rank,group,qat_steps=0):
    """Write ONLY selected packed projections; remaining weights use source at runtime."""
    source,dest=Path(source),Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    identity=sha256(source)
    payload,specs={},{}
    with safe_open(str(source),framework="pt",device="cpu") as f:
        keys=sorted(k for k in ELIGIBLE if any(fnmatch.fnmatchcase(k,p) for p in patterns))
        if not keys or not set(keys)<=set(f.keys()):
            raise ValueError("missing selected raw matrices")
        for key in keys:
            w=f.get_tensor(key)
            if w.dtype!=torch.bfloat16:
                raise ValueError("source must be Raw BF16")
            p,s=encode(w,codec,rank,group,qat_steps)
            validate(p,s)
            if p.nbytes>=w.numel()+4*w.shape[0]:
                raise ValueError("compact representation exceeds native INT8 budget")
            payload[key],specs[key]=p,s
    if sha256(source)!=identity:
        raise ValueError("source changed during conversion")
    tmp=dest.with_suffix(".tmp")
    try:
        save_file(payload,str(tmp),metadata={FORMAT:json.dumps(specs),"source_sha256":identity})
        os.link(tmp,dest)
    finally:
        tmp.unlink(missing_ok=True)
    return specs
