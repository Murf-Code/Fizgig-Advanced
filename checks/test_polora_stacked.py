"""PoLoRA with its state kept stacked per shape group (foreach writeback, batched sigma_max) gives the same update as
the per-pair step it replaced (checks/_polora_reference.py): mixed shapes, several steps, logging steps, a timestep
scale, a forced regroup, compile on (Dynamo with the aot_eager backend - the box has no C++ compiler for inductor),
and resumes between the two in both directions."""
import copy
import io

import pytest

torch = pytest.importorskip("torch")

from _polora_reference import PoloraReference  # noqa: E402
from fizgig.training.polora import Polora  # noqa: E402

# (r, d_in, d_out, count, dtype): wide, tall, square, a singleton, rank 1, and a bf16 group
SHAPES = [(4, 24, 16, 3, torch.float32), (4, 16, 40, 2, torch.float32), (6, 12, 12, 2, torch.float32),
          (3, 10, 30, 1, torch.float32), (1, 8, 5, 2, torch.float32), (2, 9, 7, 2, torch.bfloat16)]
KW = dict(lr=2e-2, magnitude_scale=3.0, log_norms_every=2)


def _pairs(seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for r, d_in, d_out, n, dt in SHAPES:
        for _ in range(n):
            out.append((torch.nn.Parameter((torch.randn(r, d_in, generator=g) / d_in ** 0.5).to(dt)),
                        torch.nn.Parameter((torch.randn(d_out, r, generator=g) * 1e-2).to(dt))))
    return out


def _grads(pairs, step):
    g = torch.Generator().manual_seed(1000 + step)
    for A, B in pairs:
        A.grad = (torch.randn(A.shape, generator=g) * 1e-2).to(A.dtype)
        B.grad = (torch.randn(B.shape, generator=g) * 1e-2).to(B.dtype)


def _run(opt, pairs, steps, first=1, regroup_at=None):
    for t in range(first, first + steps):
        _grads(pairs, t)
        opt.set_timestep_scale(0.7 if t >= 3 else 1.0)     # set by the training loop each step, not saved
        if t == regroup_at:
            opt._layout = None                      # as if the layout changed: the state is rebuilt and kept
        opt.step()
        assert all(float(A.grad.abs().sum()) == 0 and float(B.grad.abs().sum()) == 0 for A, B in pairs)


def _same(p1, s1, p2, s2, rtol=2e-5, atol=1e-7):
    for (A1, B1), (A2, B2) in zip(p1, p2):
        tol = dict(rtol=2e-2, atol=1e-4) if A1.dtype == torch.bfloat16 else dict(rtol=rtol, atol=atol)
        torch.testing.assert_close(A1.detach().float(), A2.detach().float(), **tol)
        torch.testing.assert_close(B1.detach().float(), B2.detach().float(), **tol)
    for i in s2:
        for k, v in s2[i].items():
            torch.testing.assert_close(s1[i][k].float(), v.float(), rtol=1e-4, atol=1e-6, msg=f"pair {i} {k}")


def _compile_on(monkeypatch):
    real = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: real(fn, **{**kw, "backend": "aot_eager"}))


@pytest.mark.parametrize("compiled", [False, True])
def test_same_update_as_the_per_pair_step(compiled, monkeypatch):
    if compiled:
        _compile_on(monkeypatch)
    p_ref, p_new = _pairs(), _pairs()
    ref = PoloraReference(pairs=p_ref, **KW)
    new = Polora(pairs=p_new, compile=compiled, **KW)
    assert len(new._groups) == len(SHAPES)
    _run(ref, p_ref, 6)
    _run(new, p_new, 6, regroup_at=4)
    assert new._compiled is compiled
    _same(p_new, new.pair_state, p_ref, ref.pair_state)
    assert new.last_norms["step"] == ref.last_norms["step"] == 6
    for k in ("sigma_A", "sigma_B", "magnitude"):
        assert new.last_norms[k] == pytest.approx(ref.last_norms[k], rel=1e-5)


def _roundtrip(sd):
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    return torch.load(buf, weights_only=False)


def test_resume_new_old_and_cold_saves():
    # one run of 6 steps, against: a new save resumed by a new optimizer, an old (per-pair) save resumed by the new
    # one, and a new save resumed by the old one (a Fizgig install from before this change)
    p_ref = _pairs()
    ref = PoloraReference(pairs=p_ref, **KW)
    _run(ref, p_ref, 6)

    def _opt(cls, pairs):
        return cls(pairs=pairs, compile=False, **KW) if cls is Polora else cls(pairs=pairs, **KW)

    for save_cls, load_cls in ((Polora, Polora), (PoloraReference, Polora), (Polora, PoloraReference)):
        p1 = _pairs()
        o1 = _opt(save_cls, p1)
        _run(o1, p1, 3)
        sd = _roundtrip(o1.state_dict())
        assert set(sd["polora"]["pairs"][0]) >= {"M_A", "M_B", "Q", "P", "v_sigma_A", "v_sigma_DB"}
        p2 = [(torch.nn.Parameter(A.detach().clone()), torch.nn.Parameter(B.detach().clone())) for A, B in p1]
        o2 = _opt(load_cls, p2)
        o2.load_state_dict(copy.deepcopy(sd))
        assert o2._step_count == 3
        _run(o2, p2, 3, first=4)
        _same(p2, o2.pair_state, p_ref, ref.pair_state)

    # a save made before the first step (no warm-start vectors yet) starts cold, as a fresh optimizer does
    p3, p4 = _pairs(), _pairs()
    sd0 = _roundtrip(PoloraReference(pairs=_pairs(), **KW).state_dict())
    assert "v_sigma_A" not in sd0["polora"]["pairs"][0]
    o3 = Polora(pairs=p3, compile=False, **KW)
    o3.load_state_dict(sd0)
    o4 = PoloraReference(pairs=p4, **KW)
    _run(o3, p3, 2)
    _run(o4, p4, 2)
    _same(p3, o3.pair_state, p4, o4.pair_state)
