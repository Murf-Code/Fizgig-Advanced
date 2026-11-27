"""Offline packed factor and low-bit direct controls; no runtime codec changes."""
import torch


def factor_bytes(rows,cols,bits):
    if min(rows,cols)<1 or bits not in (1,2,3,4,8,16):
        raise ValueError('invalid dimensions/precision')
    return 2*rows*cols if bits==16 else ((rows*cols+7)//8)*bits+2*rows


def pair_bytes(m,n,r,ba,bb):
    return factor_bytes(r,n,ba)+factor_bytes(r,m,bb)


def budget_rank(m,n,ba,bb,cap,max_rank):
    lo,hi=0,min(m,n,max_rank)
    while lo<hi:
        mid=(lo+hi+1)//2
        if pair_bytes(m,n,mid,ba,bb)<=cap: lo=mid
        else: hi=mid-1
    return lo


def quant_rows(w,bits,pack,unpack,steps=4,tile=128):
    """Independent scale per row; full odd symmetric codebook, BF16 scales.

    Quantize in bounded row chunks, then pack the flattened codes once.
    Decode from the actual packed bits, not the original floating factors.
    """
    if w.ndim!=2 or bits not in (1,2,3,4,8,16) or steps<1 or tile<1:
        raise ValueError('invalid quantizer arguments')
    if not torch.isfinite(w).all(): raise ValueError('nonfinite input')
    if bits==16:
        decoded=w.bfloat16().contiguous()
        return decoded,decoded.nbytes
    levels=(1<<bits)-1
    rows,cols=w.shape
    codes=torch.empty((rows,cols),dtype=torch.uint8,device=w.device)
    scales=torch.empty(rows,dtype=torch.bfloat16,device=w.device)
    for i in range(0,rows,tile):
        j=min(i+tile,rows); z=w[i:j].float()
        s=(z.abs().mean(1) if bits==1 else z.abs().amax(1)/levels).bfloat16()
        besterr=torch.full_like(s,float('inf'),dtype=torch.float32)
        for _ in range(steps):
            safe=torch.where(s>0,s,torch.ones_like(s)).float()
            q=((z/safe[:,None]+levels)/2).round().clamp(0,levels)
            v=2*q-levels
            err=(z-v*s.float()[:,None]).square().sum(1)
            keep=err<besterr
            codes[i:j]=torch.where(keep[:,None],q.byte(),codes[i:j])
            scales[i:j]=torch.where(keep,s,scales[i:j])
            besterr=torch.minimum(besterr,err)
            s=((z*v).sum(1)/v.square().sum(1)).clamp_min(0).bfloat16()
        if not torch.isfinite(besterr).all(): raise ValueError('nonfinite quantization error')
    p=pack(codes,bits)
    del codes
    d=(2*unpack(p,rows*cols,bits).reshape(rows,cols)-levels).bfloat16()*scales[:,None]
    nbytes=p.nbytes+scales.nbytes
    if nbytes!=factor_bytes(rows,cols,bits): raise ValueError('packed byte mismatch')
    return d,nbytes
