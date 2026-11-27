"""Unit checks for the opt-in WIP port, not a substitute for a CUDA run."""
import math
import pytest
import torch
from fizgig.training.selective_compensation import active_count_multiplier
from fizgig.training.muon import Muon, BatchedMuon


def test_schedule():
    assert active_count_multiplier(28, 28) == 1.
    assert active_count_multiplier(14, 28) == math.sqrt(2)
    assert active_count_multiplier(1, 28) == 1.5
    with pytest.raises(ValueError):
        active_count_multiplier(0, 28)


@pytest.mark.parametrize("cls", [Muon, BatchedMuon])
def test_only_learning_term_changes(cls):
    gen = torch.Generator().manual_seed(19)
    initial = torch.randn(4, 8, generator=gen)
    grad = torch.randn(4, 8, generator=gen)
    results = []
    for scale in (1., math.sqrt(2)):
        a, b = [torch.nn.Parameter(initial.clone()) for _ in range(2)]
        opt = cls([a,b], lr=.001, weight_decay=.01, adjust_lr_fn="match_rms_adamw")
        a.grad = grad.clone()
        b.grad = grad.clone()
        opt.step()
        before = a.detach().clone()
        frozen = b.detach().clone()
        frozen_momentum = opt.state[b]["momentum_buffer"].clone()
        a.grad = grad.clone()
        b.grad = None
        opt.step(learning_multiplier=scale)
        assert torch.equal(b, frozen)
        assert torch.equal(opt.state[b]["momentum_buffer"], frozen_momentum)
        assert opt.param_groups[0]["lr"] == .001
        results.append((a.detach().clone() - before * (1-.001*.01),
                        opt.state[a]["momentum_buffer"].clone()))
    assert torch.equal(results[0][1], results[1][1])
    assert torch.allclose(results[1][0], results[0][0]*math.sqrt(2), atol=3e-7, rtol=1e-4)


@pytest.mark.parametrize("cls", [Muon, BatchedMuon])
def test_default_is_native_and_invalid_scale_rejected(cls):
    a, b = [torch.nn.Parameter(torch.ones(4, 8)) for _ in range(2)]
    x,y = [cls([p],lr=.001) for p in (a,b)]
    a.grad = b.grad = torch.ones_like(a)
    x.step()
    y.step(learning_multiplier=1.)
    assert torch.equal(a,b)
    with pytest.raises(ValueError):
        y.step(learning_multiplier=float("nan"))
    with pytest.raises(ValueError):
        y.step(learning_multiplier=1.6)
