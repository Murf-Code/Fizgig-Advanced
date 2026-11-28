from pathlib import Path
import sys
import torch
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research/krea2_littlebit"))
import binary_qat_coverage as h


def test_rank_and_span():
    fit=[{"x":torch.tensor([[1.,0,0],[0,1,0]]),
          "dy":torch.tensor([[1.,0],[1.,0]])}]
    dev=[{"x":torch.tensor([[1.,0,1.]]),"dy":torch.tensor([[1.,1.]])}]
    r=h.geometry(fit,dev)
    assert r["x"]["numerical_rank"]==2
    assert r["dy"]["numerical_rank"]==1
    for k in ("x","dy"):
        assert r[k]["development_energy_in_fit_span"]==pytest.approx(.5)
    with pytest.raises(ValueError,match="energy"):
        h.geometry([{"x":torch.zeros(2,3),"dy":torch.zeros(2,2)}],dev)


def test_pooled_actions_and_gradient():
    torch.manual_seed(42)
    w=torch.randn(7,11)
    panels=[{"x":torch.randn(k,11),"dy":torch.randn(k,7)} for k in (3,5,2)]
    x,g,y,dx=h.pooled_fit(w,panels)[0]
    p=torch.randn_like(w,requires_grad=True)
    a=(x@p.T-y).square().sum()/y.square().sum()+(g@p-dx).square().sum()/dx.square().sum()
    ga,=torch.autograd.grad(a,p)
    b=sum((c["x"]@p.T-c["x"]@w.T).square().sum() for c in panels)/y.square().sum()
    b=b+sum((c["dy"]@p-c["dy"]@w).square().sum() for c in panels)/dx.square().sum()
    gb,=torch.autograd.grad(b,p)
    assert torch.allclose(a,b,atol=1e-6)
    assert torch.allclose(ga,gb,atol=1e-6)
