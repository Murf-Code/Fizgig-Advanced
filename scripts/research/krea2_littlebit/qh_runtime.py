"""Packed odd-level H64 Q2/Q3 runtime; same shape/bytes across swapped blocks."""
import torch
import torch.nn.functional as F
import geometry_codec as geo
import hadamard_scale_core as hs

NAMES={"qh2","qh3","qh-mix25"}
TILE=128
GROUP=64


def spec(name,m,n):
    if name not in NAMES or m<1 or n<64 or n%64:
        raise ValueError("invalid QH dimensions or codec")
    # Mask charged against the 2.5-bit cap.
    count=(m*n*5//16-m*(n//4+2*n//64)-(m+7)//8)//(n//8) if name=="qh-mix25" else (m if name=="qh3" else 0)
    if not 0<=count<=m or (name=="qh-mix25" and not 0<count<m):
        raise ValueError("unsupported mixed dimensions")
    return {"codec":name,"shape":[m,n],"group":64,"q3_rows":count}


def lengths(s):
    m,n=s["shape"]; k=s["q3_rows"]
    return (m-k)*n//4,k*n*3//8,2*m*(n//64),(m+7)//8 if s["codec"]=="qh-mix25" else 0


def pack_rows(c,bits,pack):
    if not len(c): return torch.empty(0,dtype=torch.uint8)
    return torch.cat([pack(t,bits).cpu() for t in c.split(TILE)])


def assemble(s,c2,c3,scales,mask,pack):
    m,n=s["shape"]; k=s["q3_rows"]
    if c2.shape!=(m-k,n) or c3.shape!=(k,n) or scales.shape!=(m,n//64):
        raise ValueError("QH code/scale shape mismatch")
    if scales.dtype!=torch.bfloat16 or not torch.isfinite(scales).all() or (scales<0).any():
        raise ValueError("invalid QH scales")
    if any(c.dtype!=torch.uint8 or (c>=(1<<b)).any() for c,b in ((c2,2),(c3,3))):
        raise ValueError("invalid QH codes")
    pieces=[pack_rows(c2,2,pack),pack_rows(c3,3,pack),scales.cpu().contiguous().view(torch.uint8).flatten()]
    if s["codec"]=="qh-mix25":
        if mask.shape!=(m,) or mask.dtype!=torch.bool or int(mask.sum())!=k:
            raise ValueError("invalid QH precision mask")
        pieces.append(pack(mask,1).cpu())
    return torch.cat(pieces)


@torch.no_grad()
def encode(w,name,pack):
    m,n=w.shape; s=spec(name,m,n)
    z=geo.hadamard(w.float())
    empty=torch.empty((0,n),dtype=torch.uint8)
    if name!="qh-mix25":
        bits=2 if name=="qh2" else 3
        c,scale=hs.initial_codes(z,bits,tile=8192)
        return assemble(s,c if bits==2 else empty,c if bits==3 else empty,scale,None,pack),s
    c2,s2=hs.initial_codes(z,2,tile=8192); c3,s3=hs.initial_codes(z,3,tile=8192)
    # Weight-only fallback for uncalibrated projections. Avoid dense decoded matrices.
    delta=[]
    for i in range(0,m,TILE):
        j=min(i+TILE,m)
        a=hs.decode(c2[i:j],s2[i:j],2).float()-z[i:j]
        b=hs.decode(c3[i:j],s3[i:j],3).float()-z[i:j]
        delta.append((b.square().sum(1)-a.square().sum(1)).cpu())
    mask=torch.zeros(m,dtype=torch.bool)
    mask[torch.cat(delta).argsort(stable=True)[:s["q3_rows"]]]=True
    return assemble(s,c2[~mask],c3[mask],torch.where(mask[:,None],s3,s2),mask,pack),s


def parts(p,s):
    a,b,c,d=lengths(s); m,n=s["shape"]; k=s["q3_rows"]
    q2=p[:a].reshape(m-k,n//4)
    q3=p[a:a+b].reshape(k,n*3//8)
    scales=p[a+b:a+b+c].view(torch.bfloat16).reshape(m,n//64)
    return q2,q3,scales,p[a+b+c:]


def validate(p,s,unpack):
    m,n=s["shape"]
    if s!=spec(s["codec"],m,n) or p.dtype!=torch.uint8 or p.ndim!=1 or p.numel()!=sum(lengths(s)):
        raise ValueError("invalid QH payload")
    _,_,scales,mask=parts(p,s)
    if not torch.isfinite(scales).all() or (scales<0).any():
        raise ValueError("invalid QH scales")
    if s["codec"]=="qh-mix25":
        raw=unpack(mask,mask.numel()*8,1)
        if int(raw[:m].sum())!=s["q3_rows"] or raw[m:].any():
            raise ValueError("invalid QH mask population/padding")


def prepare(p,s,unpack):
    q2,q3,scales,mask=parts(p,s); m,n=s["shape"]
    if s["codec"]!="qh-mix25": return q2,q3,scales,None,None,None
    mask=unpack(mask,m,1).bool()
    # Fixed-size indices avoid GPU nonzero synchronization and preserve original row order.
    r3=mask.long().cumsum(0)-1
    r2=(~mask).long().cumsum(0)-1
    return q2,q3,scales,mask,r2.clamp(0,len(q2)-1),r3.clamp(0,len(q3)-1)


def rows(state,s,i,j,unpack):
    q2,q3,scales,mask,r2,r3=state; n=s["shape"][1]; r=j-i
    def values(q,b):
        return 2*unpack(q.contiguous().flatten(),r*n,b).reshape(r,n)-((1<<b)-1)
    if mask is None:
        v=values(q3[i:j],3) if s["codec"]=="qh3" else values(q2[i:j],2)
    else:
        v=torch.where(mask[i:j,None],values(q3.index_select(0,r3[i:j]),3),
                      values(q2.index_select(0,r2[i:j]),2))
    # Match the screened BF16 decoded code products, even for FP32 input.
    return (v.reshape(r,n//64,64).bfloat16()*scales[i:j,:,None]).reshape(r,n)


def forward(x,p,s,unpack):
    z=geo.hadamard(x); state=prepare(p,s,unpack); m,_=s["shape"]
    return torch.cat([F.linear(z,rows(state,s,i,min(i+TILE,m),unpack).to(z.dtype))
                      for i in range(0,m,TILE)],dim=-1)


def backward(dy,p,s,unpack):
    m,n=s["shape"]; state=prepare(p,s,unpack)
    dx=torch.zeros((*dy.shape[:-1],n),device=dy.device,dtype=torch.float32)
    for i in range(0,m,TILE):
        j=min(i+TILE,m)
        dx.add_((dy[...,i:j]@rows(state,s,i,j,unpack).to(dy.dtype)).float())
    return geo.hadamard(dx).to(dy.dtype)
