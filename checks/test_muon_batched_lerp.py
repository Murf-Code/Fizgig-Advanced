"""BatchedMuon with the Nesterov look-ahead as one foreach op (instead of one lerp per matrix) gives the same update
as before (checks/_muon_reference.py), and a state saved by either resumes in the other."""
import io

import pytest

torch = pytest.importorskip("torch")

from _muon_reference import BatchedMuonReference  # noqa: E402
from fizgig.training.muon import BatchedMuon  # noqa: E402

SHAPES = [((4, 24), 3, torch.float32), ((16, 4), 3, torch.float32), ((8, 8), 2, torch.float32), ((3, 10), 1, torch.float32),
          ((2, 9), 2, torch.bfloat16)]


def _params():
    g = torch.Generator().manual_seed(0)
    return [torch.nn.Parameter(torch.randn(s, generator=g).to(dt)) for s, n, dt in SHAPES for _ in range(n)]


def _run(opt, ps, first, steps):
    for t in range(first, first + steps):
        g = torch.Generator().manual_seed(100 + t)
        for p in ps:
            p.grad = torch.randn(p.shape, generator=g).to(p.dtype)
        ps[0].grad = None if t == 2 else ps[0].grad          # a param without a grad this step is skipped
        opt.step()


@pytest.mark.parametrize("nesterov", [True, False])
def test_same_update_and_resume_both_ways(nesterov):
    kw = dict(lr=1e-2, weight_decay=0.01, momentum=0.95, nesterov=nesterov, adjust_lr_fn="match_rms_adamw")
    p_ref, p_new = _params(), _params()
    ref, new = BatchedMuonReference(p_ref, **kw), BatchedMuon(p_new, **kw)
    _run(ref, p_ref, 1, 5)
    _run(new, p_new, 1, 5)
    for a, b in zip(p_new, p_ref):
        assert torch.equal(a, b)
        assert torch.equal(new.state[a]["momentum_buffer"], ref.state[b]["momentum_buffer"])
    for save_cls, load_cls in ((BatchedMuonReference, BatchedMuon), (BatchedMuon, BatchedMuonReference)):
        p1 = _params()
        o1 = save_cls(p1, **kw)
        _run(o1, p1, 1, 3)
        buf = io.BytesIO()
        torch.save(o1.state_dict(), buf)
        buf.seek(0)
        p2 = [torch.nn.Parameter(p.detach().clone()) for p in p1]
        o2 = load_cls(p2, **kw)
        o2.load_state_dict(torch.load(buf))
        _run(o2, p2, 4, 2)
        for a, b in zip(p2, p_ref):
            assert torch.equal(a, b)
