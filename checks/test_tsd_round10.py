"""Round 10 native tangent controls, adapted from Grok's independent audit."""
import math
import pytest
import torch
from fizgig.training.lora_tsd import LoRATSD, _decay_factor, _product_norm
from _tsd_reference_97339ab import LoRATSD as Reference


def pairs(dtype=torch.float32):
    g = torch.Generator().manual_seed(19)
    return [(torch.nn.Parameter((torch.randn(4, 24, generator=g)/24**.5).to(dtype)),
             torch.nn.Parameter(torch.zeros(32, 4, dtype=dtype)))]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("balance", [True, False])
def test_default_parity(dtype, balance):
    p, q = pairs(dtype), pairs(dtype)
    kw = dict(lr=.05, magnitude_scale=3, ball_iters=1, balance=balance, log_every=0)
    a, b = LoRATSD(p, **kw), Reference(q, **kw)
    rng = torch.Generator().manual_seed(42)
    tangent_seen = False
    for step in range(40):
        for x, y in zip(p[0], q[0]):
            x.grad = torch.randn(x.shape, generator=rng).to(dtype)
            y.grad = x.grad.clone()
        tangent_seen |= float(_product_norm(p[0][1].float(), p[0][0].float())) >= .6
        a.step(); b.step()
        for x, y in zip(p[0], q[0]):
            assert torch.equal(x, y)
            assert a.state[x].keys() == b.state[y].keys()
            assert torch.equal(a.state[x]["momentum_buffer"], b.state[y]["momentum_buffer"])
    assert tangent_seen


@pytest.mark.parametrize("opt", [
    dict(tangent_lr_scale=.1),
    dict(tangent_lr_scale=.3, tangent_decay="rsqrt", tangent_horizon=10),
    dict(tangent_lr_scale=2, budget="snr"),
    dict(tangent_lr_scale=2, budget="snr", polar="svd"),
])
def test_cap_and_resume(opt):
    p = pairs()
    a = LoRATSD(p, lr=.05, magnitude_scale=3, max_delta_norm=.05, ball_iters=1,
                balance=False, log_every=0, **opt)
    rng = torch.Generator().manual_seed(5)
    for step in range(30):
        A,B = p[0]
        old = B.detach().double() @ A.detach().double()
        for x in p[0]:
            x.grad = torch.randn(x.shape, generator=rng)
        a.step()
        new = B.detach().double() @ A.detach().double()
        assert torch.isfinite(new).all()
        assert float((new-old).norm()) <= .15 + 1e-8
    import io
    buf = io.BytesIO()
    torch.save(a.state_dict(), buf); buf.seek(0)
    q = [tuple(torch.nn.Parameter(x.detach().clone()) for x in p[0])]
    b = LoRATSD(q, lr=.05, magnitude_scale=3, max_delta_norm=.05, ball_iters=1,
                balance=False, log_every=0, **opt)
    b.load_state_dict(torch.load(buf, weights_only=True))
    for step in range(10):
        for x,y in zip(p[0],q[0]):
            x.grad = torch.randn(x.shape, generator=rng); y.grad = x.grad.clone()
        a.step(); b.step()
        for x,y in zip(p[0],q[0]):
            assert torch.equal(x,y)


def test_decay_and_validation():
    assert _decay_factor("rsqrt", 30, 10, 0) == .5
    assert _decay_factor("linear", 30, 10, .1) == .1
    with pytest.raises(ValueError):
        LoRATSD(pairs(), tangent_decay="rsqrt")
    with pytest.raises(ValueError):
        LoRATSD(pairs(), budget="snr", momentum=1.)
