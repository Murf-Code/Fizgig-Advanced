import pytest
import torch
from fizgig.web.optimizer_match import delta_sq, settings, START, PILOT, optimizer_name, input_hash, require_recipe, require_holdout

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
    assert [r["OPTIMIZER_TYPE"] for r in rows] == [optimizer_name(case) for case in PILOT]
    assert "max_precond_dim=512" in next(r["OPTIMIZER_ARGS"] for r in rows if r["__optimizer_match"] == "soap-512")
    assert all(r["MAX_TRAIN_EPOCHS"] == "1" and not r["ADAPTIVE_LR"]
               and not r["KREA2_SELECTIVE_COMPENSATION"] for r in rows)
    assert float(rows[0]["LEARNING_RATE"]) == .00025


def test_input_hash():
    x = torch.arange(4, dtype=torch.bfloat16)
    assert input_hash(x, {"t": 1}) == input_hash(x.clone(), {"t": 1})
    assert input_hash(x, {"t": 1}) != input_hash(x, {"t": 2})


def test_research_guard_uses_actual_holdout():
    require_holdout({"paths": ["a.png", "b.png"]}, {"a": object(), "b": object()})
    for held, cached in [(None, {}), ({"paths": ["a", "a"]}, {"a": 1, "b": 2}),
                         ({"paths": ["a", "b"]}, {"a": 1})]:
        with pytest.raises(RuntimeError, match="two distinct actual held-out"):
            require_holdout(held, cached)


def test_recipe_guard_names_failed_fields():
    require_recipe({"seed": True, "epochs": True})
    with pytest.raises(RuntimeError, match="failed checks: seed, epochs"):
        require_recipe({"seed": False, "epochs": False, "rank": True})
