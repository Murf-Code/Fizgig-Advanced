import torch
import pytest
from fizgig.training.soap import SOAP
from fizgig.training.soap_memory import (ChunkedSOAP, OffloadedSOAP, CompactSOAP,
                                         _rotate, _reflectors, _covariance_factor)


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
    assert opt.state[p]["variance_age"] == 0
