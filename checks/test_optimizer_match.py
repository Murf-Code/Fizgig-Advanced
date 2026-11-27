import pytest
import torch
from fizgig.web.optimizer_match import delta_sq, settings, START, optimizer_name

def test_compact_delta_matches_dense():
    g = torch.Generator().manual_seed(42)
    a = torch.randn(3, 7, generator=g)
    b = torch.randn(9, 3, generator=g)
    aa, bb = a + .001, b - .002
    dense = (bb.double() @ aa.double() - b.double() @ a.double()).square().sum().item()
    assert delta_sq(a,b,aa,bb) == pytest.approx(dense, rel=1e-9)
    assert delta_sq(a,b,a,b) == 0.

def test_all_candidates_fixed_recipe(tmp_path):
    rows = settings(tmp_path / "group")
    assert [r["OPTIMIZER_TYPE"] for r in rows] == [optimizer_name(case) for case in START]
    assert "max_precond_dim=512" in next(r["OPTIMIZER_ARGS"] for r in rows if r["__optimizer_match"] == "soap-512")
    assert all(r["MAX_TRAIN_EPOCHS"] == "1" and not r["ADAPTIVE_LR"]
               and not r["KREA2_SELECTIVE_COMPENSATION"] for r in rows)
    assert float(rows[0]["LEARNING_RATE"]) == .00025
