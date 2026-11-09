"""The opt-in rotated (Hadamard) INT8 backward (modules/int8_train.py): its input gradients against the exact bf16
backward on small random layers - plain, heavy-tailed and spiky gradients, and the LoRA gradients through a chain of
four INT8 layers - plus the form row, its warning, and the launch / trainer flag. CPU: the INT8 numbers are
multiplied in fp32 where torch._int_mm isn't available, so the rounding measured is the GPU's."""
import torch
import torch.nn.functional as F

from fizgig.modules import int8_train as I

GOOD = 0.98


def _layer(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    lin = torch.nn.Linear(k, n, bias=False)
    w = torch.randn(n, k, generator=g) * 0.02
    w[:, :4] *= 20                                       # a few outlier input channels
    s = w.abs().amax(-1, keepdim=True) / 127
    lin.weight.requires_grad_(False)
    lin.weight.data = (w / s).round().clamp(-127, 127).to(torch.int8)
    lin.register_buffer("_int8_wscale", s.reshape(1, -1))
    lin.forward = I.int8_train_forward.__get__(lin, type(lin))
    return lin


def _cos(a, b):
    return F.cosine_similarity(a.flatten().double(), b.flatten().double(), 0).item()


def _grads(lin, x, g):
    out = {}
    for rotated in (False, True):
        I.set_rotated_backward(rotated)
        x.grad = None
        lin(x).backward(g)
        out[rotated] = x.grad.clone()
    I.set_rotated_backward(False)
    return out[False], out[True]


def test_hadamard_is_orthonormal_and_the_block_rule():
    h = I._hadamard(64, "cpu")
    assert torch.allclose(h @ h.t(), torch.eye(64), atol=1e-5) and torch.allclose(h, h.t())
    assert (I._block(6144), I._block(3072), I._block(48), I._block(24), I._block(10)) == (256, 256, 16, 0, 0)


def test_input_gradients_close_to_the_exact_backward(capsys):
    torch.manual_seed(0)
    cases = []
    for m, n, k, kind in [(64, 256, 256, "gauss"), (64, 512, 256, "heavy"), (32, 768, 768, "heavy"),
                          (64, 1024, 512, "spiky"), (64, 3072, 1024, "spiky")]:
        lin = _layer(n, k, len(cases))
        x = torch.randn(m, k, requires_grad=True)
        g = (torch.randn(m, n) if kind == "gauss" else torch.distributions.StudentT(2.0).sample((m, n))
             if kind == "heavy" else torch.randn(m, n).scatter_(1, torch.randint(0, n, (m, 2)),
                                                                torch.randn(m, 2) * 1000))
        exact, rot = _grads(lin, x, g)
        assert getattr(lin, "_int8_rot", None) is not None             # the rotated path really ran
        cases.append((kind, (m, n, k), _cos(rot, exact)))
    with capsys.disabled():
        print("\n[rotated int8 backward] input-grad cosine vs exact:",
              ", ".join(f"{k} {s}: {c:.5f}" for k, s, c in cases))
    assert min(c for *_x, c in cases) > GOOD


def test_lora_gradients_through_four_int8_layers(capsys):
    def run(rotated, spiky):
        torch.manual_seed(1)
        d = 512
        lins = [_layer(d, d, 10 + i) for i in range(4)]
        loras = [(torch.nn.Parameter(torch.randn(4, d) * 0.01), torch.nn.Parameter(torch.randn(d, 4) * 0.01))
                 for _ in lins]
        h = torch.randn(64, d)
        I.set_rotated_backward(rotated)
        for lin, (a, b) in zip(lins, loras):
            h = torch.tanh(lin(h + (h @ a.t()) @ b.t()))
        g = torch.randn_like(h)
        if spiky:
            g.scatter_(1, torch.randint(0, d, (64, 2)), torch.randn(64, 2) * 1000)
        h.backward(g)
        I.set_rotated_backward(False)
        return [torch.cat([a.grad.flatten(), b.grad.flatten()]) for a, b in loras]
    found = {}
    for spiky in (False, True):
        found["spiky" if spiky else "gauss"] = [_cos(r, e) for r, e in zip(run(True, spiky), run(False, spiky))]
    with capsys.disabled():
        print("[rotated int8 backward] LoRA-grad cosine per layer:",
              "; ".join(f"{k} " + " ".join(f"{c:.5f}" for c in v) for k, v in found.items()))
    assert min(min(v) for v in found.values()) > GOOD


def test_off_by_default_odd_sizes_stay_exact():
    assert not I.rotated_backward_on()
    lin = _layer(24, 32, 3)                      # 24 outputs: no Hadamard block of 16 or more
    x = torch.randn(20, 32, requires_grad=True)
    exact, rot = _grads(lin, x, torch.randn(20, 24))
    assert torch.equal(exact, rot) and getattr(lin, "_int8_rot_none", False)


def test_form_row_warning_and_flag():
    from fizgig.families import form, launch
    from fizgig.families.registry import FAMILIES
    from fizgig.families.train import setup_parser
    desc = next(d for d in FAMILIES.values() if "int8" in d.precisions and len(d.precisions) > 1)
    rows = {r["key"]: r for s in form.form(desc)["sections"] for r in s["rows"] if r}
    row = rows["INT8_ROTATED_BACKWARD"]
    assert form.DEFAULTS["INT8_ROTATED_BACKWARD"] is False and "may harm likeness" in row["warn"]
    bf16 = {**launch.PRECISION_LABELS, **dict(desc.precision_labels or ())}["bf16"]
    assert "INT8_ROTATED_BACKWARD" in form.state(desc, {**form.DEFAULTS, "FAMILY_PRECISION": bf16})["hidden"]
    assert "INT8_ROTATED_BACKWARD" not in form.state(desc, {**form.DEFAULTS})["hidden"]
    src = open(launch.__file__, encoding="utf-8").read()
    assert 'cmd.append("--int8_rotated_backward")' in src
    req = ["--family", desc.key, "--dit", "x", "--output_dir", "o", "--output_name", "n"]
    assert setup_parser().parse_known_args(req + ["--int8_rotated_backward"])[0].int8_rotated_backward is True
    assert setup_parser().parse_known_args(req)[0].int8_rotated_backward is False
