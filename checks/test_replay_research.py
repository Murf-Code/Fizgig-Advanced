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
