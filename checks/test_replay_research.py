import torch
from types import SimpleNamespace
from fizgig.training import replay_research as rr
from fizgig.families.lora import TRAINABLE


def test_fingerprint_and_probe():
    a, b = torch.nn.Linear(3, 2, bias=False), torch.nn.Linear(2, 4, bias=False)
    net = SimpleNamespace(wrapped={"blocks.0.test": SimpleNamespace(adapters={TRAINABLE: (a, b)})})
    h = rr.fingerprint(net)
    assert rr.fingerprint(net) == h
    a.weight.grad = torch.ones_like(a.weight)
    assert rr.active_mask(net) == [0]
    before = rr.probe_before(net)
    with torch.no_grad():
        a.weight.add_(1)
    assert rr.fingerprint(net) != h
    rows = rr.probe_after(net, before)
    assert abs(rows[0]["update_rms"] - 1) < 1e-6
    assert rows[0]["gradient_rms"] == 1
    assert rows[1]["update_rms"] == 0
    assert rows[1]["gradient_rms"] is None


def test_default_not_changed():
    assert rr.config({}) is None
    before = torch.are_deterministic_algorithms_enabled()
    with rr.execution(None):
        assert torch.are_deterministic_algorithms_enabled() == before


def test_decay_and_old_momentum_telemetry():
    a, b = torch.nn.Linear(3, 2, bias=False), torch.nn.Linear(2, 4, bias=False)
    net = SimpleNamespace(wrapped={"blocks.0.test": SimpleNamespace(adapters={TRAINABLE: (a, b)})})
    with torch.no_grad():
        a.weight.fill_(2)
        b.weight.fill_(3)
    a.weight.grad = torch.ones_like(a.weight)
    opt = torch.optim.SGD([a.weight, b.weight], lr=0.01, weight_decay=30.0, momentum=0.9)
    opt.state[a.weight]["momentum_buffer"] = -torch.ones_like(a.weight)
    old_momentum = opt.state[a.weight]["momentum_buffer"].clone()
    before = rr.probe_before(net, optimizer=opt)
    assert b.weight not in opt.state
    assert torch.equal(opt.state[a.weight]["momentum_buffer"], old_momentum)
    with torch.no_grad():
        # Mimic decoupled decay, then a known orthogonalized update.
        a.weight.mul_(0.7).add_(0.2)
    rows = rr.probe_after(net, before)
    assert abs(rows[0]["update_rms"] - 0.4) < 1e-6
    assert abs(rows[0]["decay_update_rms"] - 0.6) < 1e-6
    assert abs(rows[0]["muon_update_rms_reconstructed"] - 0.2) < 1e-6
    assert abs(rows[0]["gradient_momentum_cosine"] + 1) < 1e-6
    assert rows[0]["momentum_agreement_matrices"] == 1
    assert rows[1]["decay_update_rms"] == rows[1]["muon_update_rms_reconstructed"] == 0
    assert rows[1]["gradient_momentum_cosine"] is None
    assert rows[1]["gradient_bearing_matrices"] == 0


def test_missing_momentum_is_not_created():
    a, b = torch.nn.Linear(3, 2, bias=False), torch.nn.Linear(2, 4, bias=False)
    net = SimpleNamespace(wrapped={"blocks.0.test": SimpleNamespace(adapters={TRAINABLE: (a, b)})})
    a.weight.grad = torch.ones_like(a.weight)
    opt = torch.optim.SGD([a.weight, b.weight], lr=0.01, weight_decay=0.0)
    before = rr.probe_before(net, optimizer=opt)
    assert not opt.state
    rows = rr.probe_after(net, before)
    assert rows[0]["gradient_momentum_cosine"] is None
    assert rows[0]["momentum_agreement_matrices"] == 0
