"""SOAP (vendored, MIT), LoRA-TSD (adapted, MIT) and CMuon (our own, fine-tune only): each trains for real on CPU;
LoRA-TSD starts from a LoRA's zero up; CMuon steps each fused part as Muon steps a separate weight; the fine-tune
wiring (planner, per-weight steps, form row, launch flag)."""
import types

import pytest
import torch

from test_save_stop import _losses, _run, data, tiny  # noqa: F401 - the fixtures


def _fit(opt_fn, steps=60):
    torch.manual_seed(0)
    X, Wt = torch.randn(64, 16), torch.randn(8, 16)
    W = torch.nn.Parameter(torch.zeros(8, 16))
    opt = opt_fn([W])
    first = None
    for _ in range(steps):
        loss = ((X @ W.t() - X @ Wt.t()) ** 2).mean()
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return first, ((X @ W.t() - X @ Wt.t()) ** 2).mean().item()


def test_soap_builds_from_the_catalog_and_trains():
    from fizgig.training.optimizers import available_optimizers, create_optimizer
    from fizgig.training.soap import SOAP
    assert {"soap", "lora_tsd"} <= set(available_optimizers()) and "cmuon" not in available_optimizers()
    opt, label = create_optimizer("soap", [torch.nn.Parameter(torch.zeros(4, 4))], 3e-3,
                                  "precondition_frequency=5 magnitude_scale=3")   # PoLoRA's key is dropped
    assert isinstance(opt, SOAP) and opt.param_groups[0]["precondition_frequency"] == 5
    first, last = _fit(lambda ps: SOAP(ps, lr=5e-2, weight_decay=0.0))
    assert last < first * 0.2


def test_lora_tsd_starts_from_a_zero_up_and_caps_each_step():
    from fizgig.training.optimizers import create_optimizer
    from fizgig.training.lora_tsd import LoRATSD
    torch.manual_seed(0)
    X = torch.randn(128, 12)
    target = torch.randn(10, 3) @ torch.randn(3, 12)                 # a rank-3 change to learn
    A = torch.nn.Parameter(torch.randn(4, 12) * 0.3)                  # down: random; up: zero (a LoRA's start)
    B = torch.nn.Parameter(torch.zeros(10, 4))
    opt, _ = create_optimizer("lora_tsd", [(A, B)], 2e-1, "max_delta_norm=0.5 ball_iters=3")
    assert isinstance(opt, LoRATSD) and opt.param_groups[0]["max_delta_norm"] == 0.5
    losses, prev = [], (B @ A).detach().clone()
    for _ in range(80):
        loss = ((X @ (B @ A).t() - X @ target.t()) ** 2).mean()
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
        now = (B @ A).detach()
        assert (now - prev).norm() <= 0.5 * 1.25 + 1e-4             # capped (first-order; balancing keeps B A)
        prev = now.clone()
        assert torch.isfinite(A).all() and torch.isfinite(B).all()
    assert B.abs().sum() > 0 and losses[-1] < losses[0] * 0.1
    with pytest.raises(ValueError):
        create_optimizer("lora_tsd", [torch.nn.Parameter(torch.zeros(3, 3))], 5e-3)   # pairs only, no AdamW


@pytest.mark.parametrize("name,lr", [("lora_tsd", 5e-3), ("soap", 1e-4)])
def test_tiny_lora_runs(tiny, tmp_path, caplog, name, lr):
    T, cfg = tiny
    caplog.set_level("INFO")
    out = tmp_path / name
    assert _run(T, cfg, out, optimizer_type=name, learning_rate=lr, lr_scheduler="constant",
                max_train_epochs=2, save_state=False) == "done"
    log = _losses(out)
    assert log and all(r["lr"] == pytest.approx(lr) for r in log)    # the group's rate: schedulers apply
    assert "falling back" not in caplog.text


def test_cmuon_steps_each_fused_part_as_muon_steps_a_separate_weight():
    from fizgig.training.cmuon import CMuon, chunk_plan
    from fizgig.training.muon import Muon
    named = [("blocks.0.attn.qkv.weight", (96, 32)), ("blocks.0.attn.proj.weight", (32, 32)),
             ("blocks.0.mlp.gate_up.weight", (128, 32)), ("blocks.0.adaLN_modulation.1.weight", (192, 32)),
             ("blocks.0.mlp.down.weight", (32, 64)), ("blocks.0.adaln.lora_down.weight", (8, 32))]
    assert chunk_plan(named) == {"blocks.0.attn.qkv.weight": 3, "blocks.0.mlp.gate_up.weight": 2,
                                 "blocks.0.adaLN_modulation.1.weight": 6}
    torch.manual_seed(0)
    g = torch.randn(96, 32)
    g[:32] *= 50                                                      # Q's gradient dwarfs K's and V's
    fused = torch.nn.Parameter(torch.zeros(96, 32))
    parts = [torch.nn.Parameter(torch.zeros(32, 32)) for _ in range(3)]
    c = CMuon([fused], lr=1e-2, chunks={fused: 3})
    m = Muon(parts, lr=1e-2, weight_decay=0.0, momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw")
    fused.grad = g.clone()
    for p, gg in zip(parts, g.chunk(3)):
        p.grad = gg.clone()
    c.step()
    m.step()
    assert torch.allclose(fused.detach(), torch.cat([p.detach() for p in parts]), atol=2e-4)
    k_rms = fused.detach()[32:64].pow(2).mean().sqrt()
    plain = torch.nn.Parameter(torch.zeros(96, 32))
    plain.grad = g.clone()
    CMuon([plain], lr=1e-2).step()                                    # unchunked: K barely moves
    assert k_rms > 3 * plain.detach()[32:64].pow(2).mean().sqrt()
    first, last = _fit(lambda ps: CMuon(ps, lr=5e-2, chunks={ps[0]: 2}))
    assert last < first * 0.3                                         # a fixed-size step: settles, doesn't diverge


def test_cmuon_fine_tune_wiring(monkeypatch):
    from fizgig.families import form, ft, launch
    from fizgig.families.registry import FAMILIES
    from fizgig.training.cmuon import CMuon
    # per-weight steps from the gradient hooks (free each gradient), chunked
    lin = torch.nn.Linear(8, 24, bias=False)
    fs = ft.FusedSteps(1e-2, "cmuon")
    fs.attach([lin.weight], {lin.weight: 3})
    assert isinstance(fs.opts[lin.weight], CMuon) and fs.opts[lin.weight].chunks[lin.weight] == 3
    before = lin.weight.detach().clone()
    lin(torch.randn(4, 8)).pow(2).sum().backward()
    assert lin.weight.grad is None and not torch.equal(before, lin.weight.detach())
    fs.detach()
    # the planner makes room for its momentum (one bf16 copy of the window's weights)
    seen = {}

    def fake_plan(dit, spec, rot, free_gb, allow_stream, mp, max_parts):
        seen["wf"] = spec.window_factor
        return None, False, [], free_gb
    monkeypatch.setattr(ft, "plan_windows", fake_plan)
    be = types.SimpleNamespace(spec=ft.FTSpec(blocks="blocks", components={"attn": ("qkv",)}), dit=None, rot=None)
    ft.SharedBackend.plan(be, 20.0, opt_state=ft.STATE_FACTOR["cmuon"])
    assert seen["wf"] == pytest.approx(2.0)
    ft.SharedBackend.plan(be, 20.0)
    assert seen["wf"] == pytest.approx(1.0)
    # Fine-tune > Optimizer: a fine-tune row (the section is hidden for a LoRA); the launch flag
    desc = FAMILIES["zimage"]
    lab = next(l for l, k in launch.FT_OPTIMIZERS if k == "cmuon")
    assert launch.ft_optimizer({"FAMILY_FT_OPTIMIZER": lab}) == "cmuon"
    assert launch.ft_optimizer({}) == "adafactor"
    st = form.state(desc, {**form.DEFAULTS}, "")
    assert "finetune" in st["hidden"]
    secs = {sec["id"]: sec for sec in form.form(desc)["sections"]} if isinstance(form.form(desc), dict) else \
        {sec["id"]: sec for sec in form.form(desc)}
    assert any(r.get("key") == "FAMILY_FT_OPTIMIZER" for r in secs["finetune"]["rows"])
    assert not any(r.get("key") == "FAMILY_FT_OPTIMIZER" for sid, sec in secs.items() if sid != "finetune"
                   for r in sec["rows"])
