"""PoLoRA's compiled sigma_max stays compiled with dynamic shapes. One compiled function (dynamic=True, as Polora
compiles it; the aot_eager backend because the box has no C++ compiler for inductor) is fed group after group,
including stack size == rank, a single pair, rank 1 and a 1x1 factor (a 12-in / 1-out layer, as in Krea 2), and must
match the uncompiled result each time. On torch 2.10 the 1x1 case used to fail tracing and silently fall back."""
import pytest

torch = pytest.importorskip("torch")

from fizgig.training.polora import Polora  # noqa: E402
from fizgig.training.polora.optim import smax_groups  # noqa: E402

# (stack size 2n, A rows r, d_in, d_out): bigger first so the later ones run on the dynamic recompile
GROUPS = [(8, 4, 64, 48), (4, 4, 32, 16), (16, 16, 40, 24), (2, 16, 64, 96), (2, 1, 12, 1), (6, 1, 1, 1),
          (4, 3, 1, 9), (2, 5, 20, 1)]


def _compiled(monkeypatch):
    real = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: real(fn, **{**kw, "backend": "aot_eager"}))


def test_sigma_max_compiles_for_every_group_shape(monkeypatch):
    torch._dynamo.reset()
    fn = torch.compile(smax_groups, dynamic=True, fullgraph=True, backend="aot_eager")
    g = torch.Generator().manual_seed(0)
    for b, r, d_in, d_out in GROUPS:
        WA, WB = torch.randn(b, r, d_in, generator=g), torch.randn(b, d_out, r, generator=g)
        for warm in (False, True):
            vA = torch.randn(b, min(r, d_in), generator=g) if warm else torch.zeros(b, min(r, d_in))
            vB = torch.randn(b, min(d_out, r), generator=g) if warm else torch.zeros(b, min(d_out, r))
            got, want = fn(WA, vA, WB, vB), smax_groups(WA, vA, WB, vB)
            assert got[0].shape == got[2].shape == (b,)
            for x, y in zip(got, want):
                torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)


def test_polora_keeps_compile_on_with_a_1x1_factor(monkeypatch, caplog):
    _compiled(monkeypatch)
    torch._dynamo.reset()
    g = torch.Generator().manual_seed(1)
    pairs = [(torch.nn.Parameter(torch.randn(r, d_in, generator=g) * 0.1), torch.nn.Parameter(torch.randn(d_out, r, generator=g) * 0.01))
             for r, d_in, d_out in [(4, 32, 16), (4, 32, 16), (4, 16, 32), (1, 12, 1), (4, 4, 4), (4, 4, 4)]]
    opt = Polora(pairs=pairs, lr=2e-2, compile=True)
    for _ in range(3):
        for A, B in pairs:
            A.grad, B.grad = torch.randn(A.shape, generator=g) * 1e-2, torch.randn(B.shape, generator=g) * 1e-2
        opt.step()
    assert opt._compiled, caplog.text
    assert "compile failed" not in caplog.text
    assert all(torch.isfinite(p).all() for pr in pairs for p in pr)
