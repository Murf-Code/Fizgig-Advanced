"""PoLoRA's skip mode: a LoRA pair whose A and B both have no gradient this step (its block sat the step out, as Krea
2 Global replay's Selective block check does) is left out of the step whole - weights, momentum, Q / P and sigma_max
warm starts untouched, no momentum decay - while every other pair gets the same update as on a step where every pair
had gradients. Half a pair's gradients, or none at all, is still an error."""
import copy
import io

import pytest

torch = pytest.importorskip("torch")

from fizgig.training.polora import Polora  # noqa: E402

# (r, d_in, d_out, count, dtype): several shape groups, a singleton group and a bf16 group
SHAPES = [(4, 24, 16, 4, torch.float32), (4, 16, 40, 3, torch.float32), (3, 10, 30, 1, torch.float32),
          (2, 9, 7, 3, torch.bfloat16)]
KW = dict(lr=2e-2, magnitude_scale=3.0, log_norms_every=2)
N = sum(s[3] for s in SHAPES)


def _pairs(seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for r, d_in, d_out, n, dt in SHAPES:
        for _ in range(n):
            out.append((torch.nn.Parameter((torch.randn(r, d_in, generator=g) / d_in ** 0.5).to(dt)),
                        torch.nn.Parameter((torch.randn(d_out, r, generator=g) * 1e-2).to(dt))))
    return out


def _grads(pairs, step, skip=()):
    g = torch.Generator().manual_seed(1000 + step)
    for i, (A, B) in enumerate(pairs):
        gA = (torch.randn(A.shape, generator=g) * 1e-2).to(A.dtype)
        gB = (torch.randn(B.shape, generator=g) * 1e-2).to(B.dtype)
        A.grad, B.grad = (None, None) if i in skip else (gA, gB)


def _snap(pairs, opt):
    return ([(A.detach().clone(), B.detach().clone()) for A, B in pairs],
            {i: {k: v.clone() for k, v in st.items()} for i, st in opt.pair_state.items()})


def _opt(pairs, compiled=False):
    return Polora(pairs=pairs, compile=compiled, **KW)


def _compile_on(monkeypatch):
    real = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: real(fn, **{**kw, "backend": "aot_eager"}))


# skip sets: part of a group, a whole group (the singleton), part of the bf16 group, and everything but one pair
SKIPS = [{1, 2}, {7}, {0, 4, 9}, {0, 1, 2, 3, 4, 5, 6, 8}, set(range(N)) - {10}]


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("skip", SKIPS)
def test_skipped_pairs_are_untouched_and_the_rest_step_as_usual(skip, compiled, monkeypatch):
    if compiled:
        _compile_on(monkeypatch)
    p_full, p_skip = _pairs(), _pairs()
    full, part = _opt(p_full, compiled), _opt(p_skip, compiled)
    for t in (1, 2, 3):                         # some momentum, curvature and warm starts first
        _grads(p_full, t)
        _grads(p_skip, t)
        full.step()
        part.step()
    w_before, st_before = _snap(p_skip, part)

    _grads(p_full, 4)                           # the same gradients, except the skipped pairs have none
    _grads(p_skip, 4, skip=skip)
    full.step()
    part.step()
    assert part._step_count == full._step_count == 4

    for i, ((A, B), (Af, Bf)) in enumerate(zip(p_skip, p_full)):
        if i in skip:
            assert torch.equal(A, w_before[i][0]) and torch.equal(B, w_before[i][1]), f"pair {i} moved"
            assert A.grad is None and B.grad is None
            for k, v in part.pair_state[i].items():
                assert torch.equal(v, st_before[i][k]), f"pair {i} {k} changed"
        else:
            # the update is per pair, so leaving other pairs out of the batch does not change it (to float rounding:
            # a smaller batch may take another BLAS blocking; on this box it is bit-identical)
            tol = dict(rtol=1e-2, atol=1e-5) if A.dtype == torch.bfloat16 else dict(rtol=1e-5, atol=1e-8)
            torch.testing.assert_close(A.detach(), Af.detach(), **tol, msg=f"pair {i} A differs from the all-pairs step")
            torch.testing.assert_close(B.detach(), Bf.detach(), **tol, msg=f"pair {i} B differs from the all-pairs step")
            assert float(A.grad.abs().sum()) == 0 and float(B.grad.abs().sum()) == 0
            for k, v in part.pair_state[i].items():
                torch.testing.assert_close(v, full.pair_state[i][k], rtol=1e-5, atol=1e-8, msg=f"pair {i} {k}")
            assert not torch.equal(A, w_before[i][0])


def test_a_skipped_pair_picks_up_where_it_paused():
    # pair 2 sits out steps 3-4: afterwards it steps from the state it had at step 2, as if those steps never happened
    p_run, p_ref = _pairs(), _pairs()
    run = _opt(p_run)
    for t in (1, 2):
        _grads(p_run, t)
        run.step()
    ref_state = copy.deepcopy({k: v.clone() for k, v in run.pair_state[2].items()})
    for t in (3, 4):
        _grads(p_run, t, skip={2})
        run.step()
    for k, v in run.pair_state[2].items():
        assert torch.equal(v, ref_state[k])
    _grads(p_run, 5)
    run.step()
    assert all(torch.isfinite(A).all() and torch.isfinite(B).all() for A, B in p_run)
    assert not torch.equal(run.pair_state[2]["M_A"], ref_state["M_A"])


def test_every_pair_with_gradients_is_the_old_path():
    # no skip: the whole-group, in-place path (the one checks/test_polora_stacked.py holds to the reference step)
    p1, p2 = _pairs(), _pairs()
    o1, o2 = _opt(p1), _opt(p2)
    calls = []
    real = o2._update

    def spy(active=None):
        calls.append(active)
        return real(active)
    o2._update = spy
    for t in (1, 2, 3):
        _grads(p1, t)
        _grads(p2, t)
        o1.step()
        o2.step()
    assert calls == [None, None, None]
    for (A1, B1), (A2, B2) in zip(p1, p2):
        assert torch.equal(A1, A2) and torch.equal(B1, B2)


def test_half_a_pair_or_no_gradients_at_all_is_an_error():
    pairs = _pairs()
    opt = _opt(pairs)
    _grads(pairs, 1)
    pairs[3][1].grad = None
    with pytest.raises(ValueError, match="only its A"):
        opt.step()
    _grads(pairs, 1)
    pairs[3][0].grad = None
    with pytest.raises(ValueError, match="only its B"):
        opt.step()
    _grads(pairs, 1, skip=set(range(N)))
    with pytest.raises(ValueError, match="Gradients are required"):
        opt.step()


def test_logged_factor_sizes_cover_the_pairs_that_stepped():
    pairs = _pairs()
    opt = _opt(pairs)
    _grads(pairs, 1)
    opt.step()
    _grads(pairs, 2, skip={0, 1, 2, 3})          # a logging step (log_norms_every=2) with a whole group skipped
    opt.step()
    assert opt.last_norms["step"] == 2
    assert all(map(lambda k: opt.last_norms[k] > 0, ("sigma_A", "sigma_B")))


def test_resume_after_a_skipped_step():
    p1 = _pairs()
    o1 = _opt(p1)
    _grads(p1, 1)
    o1.step()
    _grads(p1, 2, skip={1, 7})
    o1.step()
    buf = io.BytesIO()
    torch.save(o1.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, weights_only=False)
    p2 = [(torch.nn.Parameter(A.detach().clone()), torch.nn.Parameter(B.detach().clone())) for A, B in p1]
    o2 = _opt(p2)
    o2.load_state_dict(sd)
    for t in (3, 4):
        _grads(p1, t, skip={2})
        _grads(p2, t, skip={2})
        o1.step()
        o2.step()
    for (A1, B1), (A2, B2) in zip(p1, p2):
        assert torch.equal(A1, A2) and torch.equal(B1, B2)
