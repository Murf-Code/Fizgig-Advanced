import torch
import pytest
from fizgig.training.soap import SOAP
from fizgig.training.soap_memory import (ChunkedSOAP, OffloadedSOAP, CompactSOAP,
                                         _rotate, _reflectors, _covariance_factor,
                                         _transport_variance, _variance_floor)


def test_compact_covariance_and_rotation():
    torch.manual_seed(42)
    g = torch.randn(48, 8)
    f = _covariance_factor(None, g, .95, 8)
    assert torch.allclose(f @ f.T, .05 * (g @ g.T), atol=2e-5)
    bases = [_reflectors(f), _reflectors(_covariance_factor(None, g.T, .95, 8))]
    assert torch.allclose(_rotate(_rotate(g, bases), bases, True), g, atol=1e-5)


@pytest.mark.parametrize("cls,kwargs", [(ChunkedSOAP, {"chunk_rows": 5}),
                                        (OffloadedSOAP, {})])
def test_full_state_variants_cpu(cls, kwargs):
    torch.manual_seed(42)
    p = torch.nn.Parameter(torch.randn(12, 4))
    q = torch.nn.Parameter(p.detach().clone())
    ref = SOAP([p], lr=.001, precondition_frequency=100)
    opt = cls([q], lr=.001, precondition_frequency=100, **kwargs)
    for _ in range(4):
        g = torch.randn_like(p)
        p.grad, q.grad = g.clone(), g.clone()
        ref.step()
        opt.step()
    assert torch.allclose(p, q, atol=2e-5, rtol=2e-5)


def test_compact_refresh_finite():
    torch.manual_seed(42)
    p = torch.nn.Parameter(torch.randn(48, 8))
    opt = CompactSOAP([p], compact_rank=8, precondition_frequency=2)
    for _ in range(7):
        p.grad = torch.randn_like(p)
        opt.step()
    assert torch.isfinite(p).all()
    assert opt.state[p]["factors"][0].shape == (48, 8)
    assert opt.state[p]["step"] == 6
    assert opt.state[p]["variance_age"] == 6
    assert opt.state[p]["exp_avg_sq"].sum() > 0


def test_variance_transport_matches_dense():
    torch.manual_seed(123)
    var = torch.rand(37, 7)
    old = [_reflectors(torch.randn(d, min(d, 5))) for d in var.shape]
    new = [_reflectors(torch.randn(d, min(d, 6))) for d in var.shape]
    def full(b):
        return torch.ormqr(b[0], b[1], torch.eye(b[0].shape[0]), left=True)
    r0, r1 = [(full(n).T @ full(o)).square() for o, n in zip(old, new)]
    expected = r0 @ var @ r1.T
    actual = _transport_variance(var, old, new, tile=3)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)
    assert torch.allclose(actual.sum(), var.sum(), atol=1e-4)


def test_post_refresh_near_zero_gradient_does_not_amplify():
    torch.manual_seed(7)
    p = torch.nn.Parameter(torch.randn(48, 8))
    opt = CompactSOAP([p], compact_rank=8, lr=.001, weight_decay=0.,
                      precondition_frequency=3)
    for _ in range(4):
        p.grad = torch.randn_like(p)
        opt.step()
    before = p.detach().clone()
    p.grad = torch.zeros_like(p)
    opt.step()
    assert torch.isfinite(p).all()
    assert (p-before).norm() < .003 * p.numel()**.5


def test_zero_and_suspended_state():
    p = torch.nn.Parameter(torch.zeros(64, 8))
    q = torch.nn.Parameter(torch.ones(64, 8))
    opt = CompactSOAP([p, q], compact_rank=8, precondition_frequency=2)
    for _ in range(4):
        p.grad = torch.zeros_like(p)
        q.grad = torch.randn_like(q)
        opt.step()
    assert torch.count_nonzero(p) == 0
    assert torch.count_nonzero(opt.state[p]["factors"][0]) == 0
    def clone(x):
        if isinstance(x, torch.Tensor): return x.clone()
        if isinstance(x, (tuple, list)): return [clone(v) for v in x]
        if isinstance(x, dict): return {k: clone(v) for k,v in x.items()}
        return x
    def same(a,b):
        if isinstance(a, torch.Tensor): return torch.equal(a,b)
        if isinstance(a, (tuple,list)): return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        if isinstance(a, dict): return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
        return a==b
    frozen = clone(opt.state[q])
    old_weight = q.detach().clone()
    for _ in range(3):
        p.grad = torch.randn_like(p)
        q.grad = None
        opt.step()
    assert same(frozen, opt.state[q])
    assert torch.equal(q, old_weight)
    q.grad = torch.randn_like(q)
    opt.step()
    assert opt.state[q]["step"] == frozen["step"]+1
