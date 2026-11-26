import pytest
import torch
from fizgig.training.replay_forks import Pair
from fizgig.training.replay_predictors import activation_metrics, UpdateObserver
from fizgig.training.muon import BatchedMuon


def test_activation_metrics():
    x = torch.tensor([1., 2.])
    assert activation_metrics(x, 2*x)["input_output_cosine"] == pytest.approx(1.)
    assert activation_metrics(x, 2*x)["residual_ratio"] == pytest.approx(1.)
    assert activation_metrics(x, x)["residual_ratio"] == 0.
    assert activation_metrics(x, -x)["input_output_cosine"] == pytest.approx(-1.)


def test_observer_sign_and_coverage():
    a, b = [torch.nn.Parameter(torch.ones(2,2)) for _ in range(2)]
    for p in (a,b):
        p.grad = torch.ones_like(p)
    obs = UpdateObserver([Pair("x", 0, a,b,1.)])
    obs([a,b], [a.grad,b.grad], -.25, True)
    row = obs.result()[0]
    assert row["descent_cosine"] == pytest.approx(1.)
    assert row["signed_first_order_utility"] == pytest.approx(2.)
    assert row["learning_update_norm"] == pytest.approx(2**.5/2)
    with pytest.raises(ValueError):
        obs([a], [a.grad], -.25, True)


def test_muon_observer_does_not_change_optimizer_result():
    g = torch.Generator().manual_seed(12)
    p = [torch.nn.Parameter(torch.randn(4,4,generator=g)) for _ in range(2)]
    q = [torch.nn.Parameter(x.detach().clone()) for x in p]
    for a,b in zip(p,q):
        a.grad = torch.randn(4,4,generator=g)
        b.grad = a.grad.clone()
    first, second = BatchedMuon(p), BatchedMuon(q)
    observer = UpdateObserver([Pair("x",0,*p,1.)])
    first.step(proposal_sink=observer)
    second.step()
    assert 0 in observer.result()
    for a,b in zip(p,q):
        assert torch.equal(a,b)
        assert torch.equal(first.state[a]["momentum_buffer"],second.state[b]["momentum_buffer"])
