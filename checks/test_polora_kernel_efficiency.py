"""Round 4 same-batch parity against frozen 2323a51 reference (credit: Grok).
CUDA gate: launch with CUBLAS_WORKSPACE_CONFIG=:4096:8 before importing torch.
"""
import copy
import os
import pytest
import torch
from fizgig.training.polora.optim import Polora
from _polora_reference_2323a51.optim import Polora as Reference

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def deterministic():
    old = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    precision = torch.get_float32_matmul_precision()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(old, warn_only=warn)
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = tf32


def _bits_equal(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    dtype = {torch.float32: torch.int32, torch.bfloat16: torch.int16}[a.dtype]
    assert torch.equal(a.detach().contiguous().view(dtype), b.detach().contiguous().view(dtype))


def _compare(pairs, other, opt, ref):
    for pair, rhs in zip(pairs, other):
        for p, q in zip(pair, rhs):
            _bits_equal(p, q)
            if p.grad is None:
                assert q.grad is None
            else:
                _bits_equal(p.grad, q.grad)
    for i, st in opt.pair_state.items():
        assert st.keys() == ref.pair_state[i].keys()
        for k, v in st.items():
            _bits_equal(v, ref.pair_state[i][k])
    assert opt.last_norms == ref.last_norms


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rank", [16, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk", [1, 3, 99])
def test_same_batch_views_and_skip(device, rank, dtype, chunk):
    if device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        pytest.skip("Set CUBLAS_WORKSPACE_CONFIG before Python starts for strict CUDA parity")
    gen = torch.Generator().manual_seed(47)
    pairs = [(torch.nn.Parameter(torch.randn(rank, 96, generator=gen).to(device, dtype)),
              torch.nn.Parameter(torch.zeros(128, rank, device=device, dtype=dtype))) for _ in range(7)]
    other = [tuple(torch.nn.Parameter(p.detach().clone()) for p in pair) for pair in pairs]
    bpp = 4*(16*rank*(96+128)+32*rank*rank+8*(96+128))
    budget = 0 if chunk == 99 else (chunk+0.25)*bpp/2**20
    kw = dict(lr=.02, magnitude_scale=3., compile=False, chunk_memory_mib=budget, log_norms_every=2)
    opt, ref = Polora(pairs=pairs, **kw), Reference(pairs=other, **kw)
    opt.record_every = ref.record_every = 1
    for step in range(8):
        skip = {1, 3} if step % 2 else set()
        for i, (pair, rhs) in enumerate(zip(pairs, other)):
            for p, q in zip(pair, rhs):
                g = None if i in skip else torch.randn(p.shape, generator=gen).to(device, dtype)
                p.grad = g
                q.grad = None if g is None else g.clone()
        opt.set_timestep_scale(.7 if step >= 3 else 1.)
        ref.set_timestep_scale(.7 if step >= 3 else 1.)
        opt.step(); ref.step()
        _compare(pairs, other, opt, ref)
        if step == 4:
            # Cross-load a baseline checkpoint and keep comparing with the same batch boundaries.
            opt.load_state_dict(copy.deepcopy(ref.state_dict()))
    if chunk == 3:
        assert opt._chunk_indices
        old_cache = opt._chunk_indices
        opt._build_groups()
        assert old_cache is opt._chunk_indices and not old_cache
        _compare(pairs, other, opt, ref)


def test_reference_self_determinism():
    # Same CPU kernels and batch shapes on a reproducible input.
    a = torch.nn.Parameter(torch.arange(32, dtype=torch.float32).reshape(4, 8)/32)
    b = torch.nn.Parameter(torch.zeros(8, 4))
    c, d = torch.nn.Parameter(a.detach().clone()), torch.nn.Parameter(b.detach().clone())
    x, y = Reference(pairs=[(a,b)], compile=False), Reference(pairs=[(c,d)], compile=False)
    for _ in range(6):
        for p in (a,b,c,d):
            p.grad = torch.ones_like(p)
        x.step(); y.step()
        _compare([(a,b)], [(c,d)], x, y)
