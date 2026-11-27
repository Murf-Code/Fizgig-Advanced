"""Offline binary factor diagnostics, independently implemented.

No runtime format changes. Blockwise Procrustes is an approximation to the
earlier global joint rotation, not a reproduction of the published method.
"""
import math
import torch
import torch.nn.functional as F


def rank_grid(m,n,cap):
    if min(m,n,cap)<1:
        raise ValueError("positive dimensions and cap required")
    top=min(m,n,cap)
    return sorted({min(r,top) for r in (128,512,1024,1536,2048,top)})


def binary_bytes(m,n,r,extra_scales=False):
    # Same sign-plane and two latent BF16 scales as native_codec.
    return (r*n+7)//8+(r*m+7)//8+4*r+(2*(m+n) if extra_scales else 0)


def balanced(u,s,v,r):
    root=s[:r].clamp_min(0).sqrt()
    return (v[:,:r]*root).T.contiguous(),(u[:,:r]*root).contiguous()


def rotate(a,b,chunk=128,steps=8):
    # A^T and B share the latent coordinate. Independent orthogonal blocks
    # preserve their real product but avoid a full rank^3 Procrustes solve.
    if chunk<1 or steps<1:
        raise ValueError("positive rotation settings required")
    aa=a.clone(); bb=b.clone()
    for i in range(0,a.shape[0],chunk):
        j=min(i+chunk,a.shape[0])
        z=torch.cat((a[i:j].T,b[:,i:j]),dim=0)
        q=torch.eye(j-i,device=z.device,dtype=z.dtype)
        for _ in range(steps):
            signs=torch.where(z@q>=0,1.,-1.)
            l,_,rh=torch.linalg.svd(z.T@signs,full_matrices=False)
            q=l@rh
        aa[i:j]=(a[i:j].T@q).T
        bb[:,i:j]=b[:,i:j]@q
    return aa,bb


def binarize(a,b,pack,unpack):
    r,n=a.shape; m=b.shape[0]
    pa=pack(a>=0,1); pb=pack(b>=0,1)
    sa=a.abs().mean(1).bfloat16()
    sb=b.abs().mean(0).bfloat16()
    # Decode from the actual packed bits and rounded scales. Only offline
    # diagnostics retain these dense factors; do not count them as packed.
    da=(unpack(pa,r*n,1).reshape(r,n)*2-1).bfloat16()*sa[:,None]
    db=(unpack(pb,r*m,1).reshape(m,r)*2-1).bfloat16()*sb[None,:]
    return da,db,pa.nbytes+pb.nbytes+sa.nbytes+sb.nbytes


def scale_fit(w,a,b,tile=256):
    """One alternating least-squares row/column sweep on frozen binary product.
    
    Signed BF16 row and column scales are charged to the representation.
    Zero-energy directions keep scale 1. Best scale checkpoint is chosen by
    the caller on development captures, not confirmation.
    """
    af=a.float(); bf=b.float()
    m,n=w.shape
    row=torch.ones(m,device=w.device,dtype=torch.bfloat16)
    col=torch.ones(n,device=w.device,dtype=torch.bfloat16)
    for i in range(0,m,tile):
        j=min(m,i+tile)
        z=bf[i:j]@af
        den=z.square().sum(1)
        val=(w[i:j].float()*z).sum(1)/den.clamp_min(1e-30)
        row[i:j]=torch.where(den>0,val,torch.ones_like(val)).bfloat16()
    num=torch.zeros(n,device=w.device); den=torch.zeros_like(num)
    for i in range(0,m,tile):
        j=min(m,i+tile)
        z=(bf[i:j]@af)*row[i:j,None].float()
        num+=(w[i:j].float()*z).sum(0)
        den+=z.square().sum(0)
    col=torch.where(den>0,num/den.clamp_min(1e-30),torch.ones_like(num)).bfloat16()
    if not torch.isfinite(row).all() or not torch.isfinite(col).all():
        raise ValueError("nonfinite fitted scales")
    return row,col


def weight_error(w,a,b,row=None,col=None,tile=256):
    af=a.float(); bf=b.float()
    num=den=0.
    for i in range(0,w.shape[0],tile):
        j=min(w.shape[0],i+tile)
        z=bf[i:j]@af
        if row is not None:
            z=z*row[i:j,None].float()*col[None,:].float()
        wt=w[i:j].float()
        num+=float((z-wt).square().sum(dtype=torch.float64))
        den+=float(wt.square().sum(dtype=torch.float64))
    return math.sqrt(num/den) if den else None


def actions(x,g,a,b,row=None,col=None):
    # BF16 factor operator in the original input/output basis. Autograd
    # accounts for the actual cast/scale ordering in the proposed operator.
    z=x.detach().bfloat16().requires_grad_(True)
    u=z if col is None else z*col
    y=F.linear(F.linear(u,a),b)
    if row is not None:
        y=y*row
    dx,=torch.autograd.grad(y,z,g.bfloat16())
    return y.detach(),dx.detach()
