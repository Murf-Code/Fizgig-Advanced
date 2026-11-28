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


def test_soap_sweep(tmp_path):
    from fizgig.web.optimizer_match import soap_settings, expected_updates, SOAP_SWEEP
    rows = soap_settings(tmp_path / "group", "a test prompt")
    assert [r["__optimizer_match"] for r in rows] == list(SOAP_SWEEP)
    assert [float(r["LEARNING_RATE"]) for r in rows] == [.00025,.0004,.00025,.0005]
    assert all(expected_updates(r["__optimizer_match"]) == 65 for r in rows)
    assert all(r["SAMPLE_ENABLED"] and r["SAMPLE_AT_FIRST"] and r["SAMPLE_SEED"] == "1234"
               and r["MAX_TRAIN_EPOCHS"] == "5" and not r["ADAPTIVE_LR"]
               and not r["KREA2_SELECTIVE_COMPENSATION"] for r in rows)


def test_selective_compat_recipe(tmp_path):
    from fizgig.web.optimizer_match import selective_settings, SELECTIVE_RATES, selective_arm, expected_updates, optimizer_name
    rows = selective_settings(tmp_path / "group", "fixed prompt")
    assert len(rows) == 16
    for full, sel in zip(rows[::2], rows[1::2]):
        name = full["OPTIMIZER_TYPE"]
        assert name == sel["OPTIMIZER_TYPE"] == optimizer_name(sel["__optimizer_match"])
        assert float(full["LEARNING_RATE"]) == float(sel["LEARNING_RATE"]) == SELECTIVE_RATES[name]
        assert not selective_arm(full["__optimizer_match"]) and selective_arm(sel["__optimizer_match"])
        assert not full["KREA2_SELECTIVE_BLOCK_CHECK"] and sel["KREA2_SELECTIVE_BLOCK_CHECK"]
        assert not full["KREA2_SELECTIVE_COMPENSATION"] and not sel["KREA2_SELECTIVE_COMPENSATION"]
        assert expected_updates(sel["__optimizer_match"]) == 65
    assert len(selective_settings(tmp_path / "group", "fixed prompt", ["polora"])) == 2


def test_effective_selective_recipe(tmp_path):
    from fizgig.web.optimizer_match import effective_settings, selective_arm, expected_updates, optimizer_name
    rows = effective_settings(tmp_path / "group", "fixed prompt")
    assert [r["__optimizer_match"] for r in rows] == [
        "polora-geometry-full","polora-geometry-selective","polora-geometry-effective",
        "lora_tsd-geometry-full","lora_tsd-geometry-selective","lora_tsd-geometry-effective"]
    for i,row in enumerate(rows):
        case = row["__optimizer_match"]
        assert expected_updates(case)==78
        assert row["MAX_TRAIN_EPOCHS"]=="6"
        assert selective_arm(case)==(i%3!=0)
        assert row["KREA2_SELECTIVE_BLOCK_CHECK"]==selective_arm(case)
        assert not row["KREA2_SELECTIVE_COMPENSATION"]
        assert row["OPTIMIZER_TYPE"]==optimizer_name(case)


def test_effective_legacy_recipe_unchanged(tmp_path):
    from fizgig.web.optimizer_match import effective_settings, expected_updates
    rows = effective_settings(tmp_path / "legacy", "fixed prompt", epochs=5)
    assert [r["__optimizer_match"] for r in rows] == [
        "polora-full", "polora-selective", "polora-effective",
        "lora_tsd-full", "lora_tsd-selective", "lora_tsd-effective"]
    assert all(r["MAX_TRAIN_EPOCHS"] == "5" and expected_updates(r["__optimizer_match"]) == 65 for r in rows)


def test_pair_comp_recipe(tmp_path):
    from types import SimpleNamespace
    from fizgig.web.optimizer_match import pair_comp_settings, expected_updates, selective_arm, attach
    rows = pair_comp_settings(tmp_path / "pair", "fixed prompt")
    assert len(rows) == 8
    assert len({r["__optimizer_match"] for r in rows}) == 8
    for off,on in zip(rows[::2],rows[1::2]):
        assert off["__optimizer_match"].replace("-comp-off-", "-comp-on-") == on["__optimizer_match"]
        assert all(off[k]==on[k] for k in off if k not in ("__optimizer_match","LORA_NAME","LORA_OUTPUT_DIR"))
    for row in rows:
        assert row["MAX_TRAIN_EPOCHS"] == "6"
        assert expected_updates(row["__optimizer_match"]) == 78
        assert selective_arm(row["__optimizer_match"])
        assert not row["KREA2_SELECTIVE_COMPENSATION"]
        plan = SimpleNamespace(problems=[], stages=[SimpleNamespace(name="Training",cmd=[])])
        attach(plan,row,"krea2")
        assert plan.stages[0].cmd == ["--family_option", "optimizer_match="+row["__optimizer_match"]]
