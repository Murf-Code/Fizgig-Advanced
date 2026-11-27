"""Q6-Hadamard CPU screen: reuse tiles across probes, keep native arithmetic."""
import torch
import torch.nn.functional as F
import geometry_codec as geometry
from native_codec import _unpack


@torch.no_grad()
def q6_probes(p,s,probes):
    if s["codec"]!="q6-had":
        raise ValueError("only Q6 Hadamard supported")
    m,n=s["shape"]
    for x,dy in probes:
        if x.dtype!=torch.bfloat16 or dy.dtype!=x.dtype or x.shape[:-1]!=dy.shape[:-1] or x.shape[-1]!=n or dy.shape[-1]!=m:
            raise ValueError("invalid Q6 probe shape/dtype")
    z=[geometry.hadamard(x) for x,_ in probes]
    ys=[[] for _ in probes]
    dx=[torch.zeros((*dy.shape[:-1],n),device=dy.device,dtype=torch.float32) for _,dy in probes]
    for i in range(0,m,geometry.TILE):
        j=min(i+geometry.TILE,m)
        tile=geometry._decode_tile(p,s,i,j,torch.bfloat16,_unpack)
        for k,(_,dy) in enumerate(probes):
            ys[k].append(F.linear(z[k],tile))
            dx[k].add_((dy[...,i:j]@tile).float())
    return [(torch.cat(y,dim=-1),geometry.hadamard(d).to(torch.bfloat16)) for y,d in zip(ys,dx)]
