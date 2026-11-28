from pathlib import Path
import sys
import torch
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research/krea2_littlebit"))
import rank_representation_screen as h


def test_weight_and_backward_rank_optima():
    torch.manual_seed(42)
    w=torch.randn(19,31,dtype=torch.float64)
    g=torch.randn(9,19,dtype=torch.float64)
    g[:,3:]*=.1
    rank=7
    a0,b0,tail=h.truncated(w,rank)
    a,b,lower=h.backward_reference(w,g,rank)
    def loss(a,b):
        err=b@a-w
        return .2*err.square().sum()/w.square().sum()+.4*(g@err).square().sum()/(g@w).square().sum()
    assert a.shape==(rank,31) and b.shape==(19,rank)
    assert float((b0@a0-w).square().sum()/w.square().sum())==pytest.approx(tail,rel=1e-10)
    assert float(loss(a,b))==pytest.approx(lower,rel=1e-10,abs=1e-12)
    assert loss(a,b)<loss(a0,b0)
    af,bf,zero=h.backward_reference(w,g,19)
    assert zero==0 and torch.allclose(bf@af,w,atol=1e-10)
    with pytest.raises(ValueError,match="energies"):
        h.backward_reference(w,torch.zeros_like(g),rank)


def test_pooled_loss_and_gradient_match_dense_actions():
    torch.manual_seed(42)
    w=torch.randn(7,11,dtype=torch.float64)
    x=torch.randn(5,11,dtype=w.dtype)
    g=torch.randn(5,7,dtype=w.dtype)
    a=torch.randn(3,11,dtype=w.dtype,requires_grad=True)
    b=torch.randn(7,3,dtype=w.dtype,requires_grad=True)
    y,dx=x@w.T,g@w
    loss=h.pooled_loss(a,b,w,x,g,y,dx)
    err=b@a-w
    dense=.2*err.square().sum()/w.square().sum()+.4*(x@err.T).square().sum()/y.square().sum()+.4*(g@err).square().sum()/dx.square().sum()
    assert torch.allclose(loss,dense,atol=1e-12)
    for ga,gb in zip(torch.autograd.grad(loss,(a,b)),torch.autograd.grad(dense,(a,b))):
        assert torch.allclose(ga,gb,atol=1e-12)
