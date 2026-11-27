"""Research-only Q6 compute grouping; preserve the original 128-row archive."""
import torch
import torch.nn.functional as F
import geometry_codec as geo
import native_codec as codec


def decode(p, s, start, end, dtype, batched):
    m, n = s["shape"]
    t = geo.TILE
    if s["codec"] != "q6-had" or start % t or end > m or start >= end or (end != m and end % t):
        raise ValueError("invalid Q6 compute tile")
    if not batched:
        return torch.cat([geo._decode_tile(p, s, i, min(i+t, end), dtype, codec._unpack)
                          for i in range(start, end, t)], dim=0)
    # Every archive tile contains its own bit planes followed by its scales.
    # Batch only full storage tiles; the final partial tile has a different stride.
    full = (end-start)//t
    chunks = []
    if full:
        stride = geo._tile_bytes(s, t)
        offset = geo._tile_bytes(s, start)
        body = p[offset:offset+full*stride].reshape(full, stride)
        cut = t*n*6//8
        q = codec._unpack(body[:, :cut].contiguous().flatten(), full*t*n, 6)
        scale = body[:, cut:].contiguous().view(torch.bfloat16).reshape(full*t, n//64, 1).to(dtype)
        chunks.append(((q.reshape(full*t,n//64,64)-31).to(dtype)*scale).reshape(full*t,n))
    tail = start+full*t
    if tail < end:
        chunks.append(geo._decode_tile(p,s,tail,end,dtype,codec._unpack))
    return chunks[0] if len(chunks)==1 else torch.cat(chunks,dim=0)


def forward(x,p,s,rows,batched=True):
    if rows < geo.TILE or rows % geo.TILE:
        raise ValueError("compute rows must be a positive multiple of storage rows")
    z=geo.hadamard(x)
    m,n=s["shape"]
    return torch.cat([F.linear(z,decode(p,s,i,min(i+rows,m),z.dtype,batched))
                      for i in range(0,m,rows)],dim=-1)


def backward(dy,p,s,rows,batched=True):
    if rows < geo.TILE or rows % geo.TILE:
        raise ValueError("compute rows must be a positive multiple of storage rows")
    m,n=s["shape"]
    dx=torch.zeros((*dy.shape[:-1],n),device=dy.device,dtype=torch.float32)
    for i in range(0,m,rows):
        j=min(i+rows,m)
        dx.add_((dy[...,i:j]@decode(p,s,i,j,dy.dtype,batched)).float())
    return geo.hadamard(dx).to(dy.dtype)
