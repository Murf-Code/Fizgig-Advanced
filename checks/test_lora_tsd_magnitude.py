"""LoRA-TSD's magnitude_scale (as PoLoRA's): 1.0 is the old step; a bigger one scales each step and its cap alike;
the Optimizer Args word is kept for LoRA-TSD (dropped for others) and pre-filled at 3 (no box: Optimizer Args only)."""
import os
import re

import pytest
import torch

from fizgig.training.lora_tsd import LoRATSD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pair(zero_up):
    torch.manual_seed(0)
    A = torch.nn.Parameter(torch.randn(4, 12) * 0.3)
    B = torch.nn.Parameter(torch.zeros(10, 4) if zero_up else torch.randn(10, 4) * 0.3)
    return A, B


def _steps(n=1, zero_up=False, **kw):
    """B A's change over n steps, and the final A, B."""
    A, B = _pair(zero_up)
    torch.manual_seed(1)
    X, target = torch.randn(64, 12), torch.randn(10, 12)
    opt = LoRATSD([(A, B)], **kw)
    before = (B @ A).detach().clone()
    for _ in range(n):
        loss = ((X @ (B @ A).t() - X @ target.t()) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return ((B @ A).detach() - before), A.detach().clone(), B.detach().clone(), opt


@pytest.mark.parametrize("zero_up", [True, False])
def test_magnitude_1_is_the_old_step(zero_up):
    _, A0, B0, opt = _steps(5, zero_up, lr=5e-3)
    assert opt.magnitude_scale == 1.0
    _, A1, B1, _ = _steps(5, zero_up, lr=5e-3, magnitude_scale=1.0)
    assert torch.equal(A0, A1) and torch.equal(B0, B1)


@pytest.mark.parametrize("zero_up", [True, False])
def test_magnitude_3_is_about_3x_the_change(zero_up):
    d1 = _steps(1, zero_up, lr=1e-3)[0].norm().item()
    d3 = _steps(1, zero_up, lr=1e-3, magnitude_scale=3.0)[0].norm().item()
    assert d1 > 0 and d3 / d1 == pytest.approx(3.0, rel=0.05)
    # magnitude m at lr and cap is the same step as magnitude 1 at lr x m and cap x m
    _, A3, B3, _ = _steps(3, zero_up, lr=1e-3, max_delta_norm=0.05, magnitude_scale=3.0)
    _, Al, Bl, _ = _steps(3, zero_up, lr=3e-3, max_delta_norm=0.15)
    assert torch.allclose(A3, Al, atol=1e-6) and torch.allclose(B3, Bl, atol=1e-6)


@pytest.mark.parametrize("zero_up", [True, False])
def test_the_cap_grows_with_the_magnitude(zero_up):
    cap = 1e-3                                                     # far under the step at lr 1: both capped
    d1 = _steps(1, zero_up, lr=1.0, max_delta_norm=cap)[0].norm().item()
    d3, _, _, opt = _steps(1, zero_up, lr=1.0, max_delta_norm=cap, magnitude_scale=3.0)
    assert d1 == pytest.approx(cap, rel=0.05) and d3.norm().item() == pytest.approx(3 * cap, rel=0.05)
    st = opt.last_stats                                            # logged at the first step
    assert st["magnitude"] == 3.0 and st["capped"] == 1 and st["pairs"] == 1
    assert st["avg_change"] == pytest.approx(3 * cap, rel=1e-6)


def test_optimizer_args_keep_magnitude_for_lora_tsd_only():
    from fizgig.training.optimizers import create_optimizer, strip_polora_args
    opt, _ = create_optimizer("lora_tsd", [_pair(True)], 5e-3, "magnitude_scale=3 compile=True curvature_beta=0.9")
    assert isinstance(opt, LoRATSD) and opt.magnitude_scale == 3.0      # PoLoRA's other words are dropped
    opt, _ = create_optimizer("lora_tsd", [_pair(True)], 5e-3, "")
    assert opt.magnitude_scale == 1.0
    assert strip_polora_args("magnitude_scale=3 compile=True ns_steps=5", "lora_tsd") == "magnitude_scale=3 ns_steps=5"
    assert strip_polora_args("magnitude_scale=3 compile=True ns_steps=5", "muon") == "ns_steps=5"


def test_lora_tsd_args_keep_magnitude_and_the_ui_prefills_3():
    from fizgig.families import form
    from fizgig.families.registry import FAMILIES
    desc = FAMILIES["krea2"]
    v = {**form.DEFAULTS, "OPTIMIZER_TYPE": "lora_tsd", "LEARNING_RATE": "5e-3", "OPTIMIZER_ARGS": "magnitude_scale=3"}
    st = form.state(desc, v)
    assert set(form.POLORA_BOXES) <= set(st["hidden"])            # PoLoRA's boxes stay PoLoRA's
    assert "OPTIMIZER_ARGS" not in st["force"]                     # the word stays in the args
    out = form.apply_preset(desc, {**v, "OPTIMIZER_ARGS": "magnitude_scale=3 compile=True"}, {"OPTIMIZER_TYPE": "lora_tsd"})
    assert out["OPTIMIZER_ARGS"] == "magnitude_scale=3"
    out = form.apply_preset(desc, {**v, "OPTIMIZER_ARGS": "magnitude_scale=3"}, {"OPTIMIZER_TYPE": "muon"})
    assert "magnitude_scale" not in out["OPTIMIZER_ARGS"]
    src = open(os.path.join(ROOT, "src/fizgig/web/static/app.js"), encoding="utf-8").read()
    assert 'const TSD_DEFAULT_ARGS = {magnitude_scale: "3"};' in src
    assert re.search(r'stripPoloraArgs\(curArgs, opt\)', src)
    assert "magnitude_scale=3" in form.OPTIMIZER_TYPE_TIPS["lora_tsd"]
