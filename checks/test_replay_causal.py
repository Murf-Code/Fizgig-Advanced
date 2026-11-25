import json
import pytest
from fizgig.krea2 import replay_causal as c


def fixture_report(tmp_path):
    masks = [set(c.ALL) if i < 2 else c.ALL - {26, 25} for i in range(len(c.PROBES))]
    rows = [{"kind": "environment", "deterministic": True, "mode": "causal"},
            {"kind": "hash", "step": 0, "active_blocks": [], "sha256": "0"}]
    rows += [{"kind": "hash", "step": s, "active_blocks": sorted(c.expected_mask(masks, s)),
              "sha256": str(s)} for s in range(1, 146)]
    run = {"case": c.SOURCE, "status": "finished", "lr": "0.00045",
           "settings": {"SEED": "42"}, "epochs": [{"step": 145}],
           "research": rows, "probes": [{"step": s, "active": len(m), "total": 28,
           "strategy": "adaptive_gradient", "active_blocks": sorted(m)} for s, m in zip(c.PROBES, masks)]}
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"runs": [run]}))
    return path, run, masks


def test_source_and_controls(tmp_path):
    path, run, masks = fixture_report(tmp_path)
    assert c.source_schedule(path) == masks
    for i, m in enumerate(masks):
        assert c.intervention("exact", masks, i) == m
        assert len(c.intervention("random", masks, i)) == len(m)
        a, b = (c.intervention(route, masks, i) for route in ("block26", "block13"))
        assert len(a) == len(b)
        assert a == (c.ALL - {26} if 26 not in m else c.ALL)
        assert b == (c.ALL - {13} if 26 not in m else c.ALL)
    assert c.expected_mask(masks, 24) == c.ALL
    assert c.expected_mask(masks, 25) == c.ALL - {25, 26}
    assert c.expected_mask(masks, 145) == masks[-1]
    run["research"][-1]["active_blocks"] = list(range(28))
    path.write_text(json.dumps({"runs": [run]}))
    with pytest.raises(RuntimeError):
        c.source_schedule(path)


def test_prefix_diagnostic(tmp_path):
    _, run, _ = fixture_report(tmp_path)
    import copy
    exact = copy.deepcopy(run)
    exact["case"] = "causal-exact-lr450"
    assert c.compare_runs([run, exact])[exact["case"]]["identical_hashes"] == 146
    exact["research"][2]["sha256"] = "different"
    assert not c.compare_runs([run, exact])[exact["case"]]["pre_intervention_hash_match"]
