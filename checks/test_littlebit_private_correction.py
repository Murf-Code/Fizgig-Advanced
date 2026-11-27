import sys
from pathlib import Path
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts/research/krea2_littlebit"))
import native_codec as codec
import private_correction as pc


def test_budget_real_key():
    r=pc.ranks(1536,6144)
    assert r["private_rank"]==76
    assert r["shared_rank"]==128
    assert r["q5_bytes"]+2*2*76*(1536+6144)<=r["budget_bytes"]
    assert r["q5_bytes"]+2*128*(6144+2*1536)<=r["budget_bytes"]


def test_split_groups():
    trace={"steps":[{"step":i+1,"items":[k]} for i,k in enumerate(["a","b","a","c"])],
           "calibration":[{"step":i+1} for i in range(4)]}
    s=pc.split_captures(trace)
    groups=[{trace["steps"][r["step"]-1]["items"][0] for r in rows} for rows in s.values()]
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i+1,3))
    assert sum(map(len,s.values()))==4
    trace["calibration"].pop()
    with pytest.raises(ValueError,match="distinct"):
        pc.split_captures(trace)


def test_q5_packed_operator():
    g=torch.Generator().manual_seed(3)
    w=torch.randn(132,128,generator=g).bfloat16()
    p,s=codec.encode(w,"q5-had",rank=0)
    codec.validate(p,s)
    assert p.nbytes==132*(128*5//8+2*2)
    d=torch.cat([v for _,_,v in pc.decoded_tiles(p,s)])
    x=torch.randn(5,128,generator=g).bfloat16()
    dy=torch.randn(5,132,generator=g).bfloat16()
    y,dx=pc.operator(p,s,x,dy)
    assert pc.relative(y,x@d.T.bfloat16())<.02
    assert pc.relative(dx,dy@d.bfloat16())<.02


@pytest.mark.parametrize("shared",[False,True])
def test_correction_fit_and_budget(shared):
    g=torch.Generator().manual_seed(11)
    w=[(torch.randn(128,8,generator=g)@torch.randn(8,128,generator=g)
        +.05*torch.randn(128,128,generator=g)).bfloat16() for _ in range(2)]
    budget=pc.ranks(128,128)
    rank=budget["shared_rank" if shared else "private_rank"]
    a,b=pc.svd_factors(w,rank,shared)
    res=[pc.residual(v,a[0 if shared else i],b[i]) for i,v in enumerate(w)]
    cost=sum(p.nbytes for p,s in res)+sum(v.nbytes for v in a+b)+pc.DESCRIPTOR_BYTES
    assert cost<=budget["budget_bytes"]
    c={"x":torch.randn(8,128,generator=g).bfloat16(),
       "dy":torch.randn(8,128,generator=g).bfloat16()}
    targets=[[pc.fit_target(v,*res[i],c)] for i,v in enumerate(w)]
    fa,fb,stats=pc.fit_factors(a,b,targets,steps=5)
    assert stats["best_fit_objective"]<=stats["initial_fit_objective"]
    assert all(v.dtype==torch.bfloat16 for v in fa+fb)
    for i,v in enumerate(w):
        y,dx=pc.operator(*res[i],c["x"],c["dy"],fa[0 if shared else i],fb[i])
        assert torch.isfinite(y).all() and torch.isfinite(dx).all()
        assert pc.weight_error(v,*res[i],fa[0 if shared else i],fb[i])>=0
