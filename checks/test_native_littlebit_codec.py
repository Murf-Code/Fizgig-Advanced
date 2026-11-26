"""Run with pytest from the repo root; independent codec only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research/krea2_littlebit"))
import native_codec as c
import torch
import pytest


@pytest.mark.parametrize("name",["q4","q6","svd","littlebit","littlebit2"])
def test_forward_backward(name):
    torch.manual_seed(12)
    w=torch.randn(32,64).bfloat16()
    p,s=c.encode(w,name,rank=8)
    c.validate(p,s)
    x=torch.randn(4,64,requires_grad=True)
    y=c.linear(x,p,s)
    dy=torch.randn_like(y)
    dx,=torch.autograd.grad(y,x,dy)
    if name in ("q4","q6"):
        dense=c._rows(p,s,0,32,x.dtype)
    else:
        a,b=c.factors(p,s)
        dense=b@a
    assert torch.allclose(y,x@dense.T,atol=2e-4,rtol=2e-4)
    assert torch.allclose(dx,dy@dense,atol=2e-4,rtol=2e-4)
    with pytest.raises(ValueError):
        c.validate(p[:-1],s)
    layer=torch.nn.Linear(64,32,bias=False)
    c.install(layer,p,s)
    assert not layer.weight.requires_grad
    assert torch.equal(layer(x),y)


def test_pack_roundtrip():
    for bits in (1,4,6):
        v=torch.arange(65)%(1<<bits)
        assert torch.equal(c._unpack(c._pack(v,bits),len(v),bits),v)
