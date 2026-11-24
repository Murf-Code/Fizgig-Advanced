"""Calibration policy regressions: synthetic reports, no GPU required."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_stream_sync.py"
spec = importlib.util.spec_from_file_location("diagnose_stream_sync_calibration", path)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)
from fizgig.krea2.stream_calibration import recommend_stream_strategy


def _report():
    args = SimpleNamespace(suite="calibration", cpu=False, swap=22)
    planned = diag.cases(args)
    report = dict(configuration=dict(suite="calibration", cpu=False, steps=4,
                                     warmup=1, swap=22, model="test"),
                  hardware=dict(ram_total_gib=64, vram_total_gib=24),
                  planned_cases=planned, runs=[], errors=[], skipped=[])
    metric = dict(tensors=3, exact=3, finite=True, relative_l2=0.0, max_abs=0.0)
    for c in planned:
        # Ring 2 wins over classic. Ring 3 improves on it by 10%.
        # Ring 4 is barely faster than ring 3, and direct saves too little.
        speed = 10 if c["mode"] == "classic" else {2: 8, 3: 7.2, 4: 7.1}[c["ring"]]
        if c["pinned"]:
            speed -= 0.05
        rows = [dict(step=i, loss=1.0, validation=2.0, loss_abs_error=0.0,
                     validation_abs_error=0.0, gradients=dict(metric), updates=dict(metric),
                     seconds=100 if i == 1 else speed, transfers=dict(d2h_bytes=0))
                for i in range(1, 5)]
        report["runs"].append(dict(**c, status="completed", steps=rows,
                                   peak_allocated_gib=16, peak_reserved_gib=17,
                                   pinned_storage_gib=15 if c["pinned"] else 2,
                                   host_memory_plan=dict(direct=c["pinned"],
                                                         staging_slots=0 if c["pinned"] else c["ring"]),
                                   memory_samples=dict(system_available_min_gib=20)))
    return report


def _choose(report):
    return recommend_stream_strategy(report, diag.evaluate_exact_parity)


def _runs(report, pinned, ring):
    return [r for r in report["runs"] if r["mode"] == "fast" and
            r["pinned"] == pinned and r["ring"] == ring]


def test_repeated_suite_prefers_smallest_near_fastest_staged_ring():
    result = _choose(_report())
    assert result["selected"] == dict(mode="fast", pinned=False, ring=3)
    assert result["workload_scope"] == "real-checkpoint"
    assert not result["applied"]


def test_targeted_calibration_includes_both_repeats_and_classic():
    cases = diag.cases(SimpleNamespace(suite="calibration", cpu=False, swap=22))
    selected = diag.select_cases(cases, ["CAL_staged_ring3_1"])
    assert [c["label"] for c in selected] == [
        "A_classic_pageable", "CAL_staged_ring2_1", "CAL_staged_ring3_1",
        "CAL_staged_ring3_2", "CAL_staged_ring2_2", "D_classic_repeat"]


def test_tight_vram_only_schedules_fitting_rings():
    cases = diag.cases(SimpleNamespace(suite="calibration", cpu=False, swap=26))
    assert all(c["ring"] == 2 for c in cases)
    assert len(cases) == 6


@pytest.mark.parametrize("defect", ["parity", "ram", "vram", "missing_repeat", "spread",
                                   "d2h", "fallback", "mislabeled", "too_small_gain",
                                   "missing_memory", "nan_timing"])
def test_reject_bad_candidate(defect):
    report = _report()
    runs = _runs(report, False, 3)
    if defect == "parity":
        runs[0]["steps"][1]["gradients"]["exact"] = 2
    elif defect == "ram":
        runs[0]["memory_samples"]["system_available_min_gib"] = 5
    elif defect == "vram":
        runs[0]["peak_reserved_gib"] = 18
    elif defect == "missing_repeat":
        report["runs"].remove(runs[0])
    elif defect == "spread":
        for row in runs[0]["steps"]:
            row["seconds"] = 5
    elif defect == "d2h":
        runs[0]["steps"][1]["transfers"]["d2h_bytes"] = 1
    elif defect == "fallback":
        runs[0]["host_memory_plan"]["staging_slots"] = 1
    elif defect == "mislabeled":
        runs[0]["ring"] = 2
    elif defect == "too_small_gain":
        for run in runs:
            for row in run["steps"]:
                row["seconds"] = 7.9
    elif defect == "missing_memory":
        runs[0]["memory_samples"] = {}
    elif defect == "nan_timing":
        runs[0]["steps"][2]["seconds"] = float("nan")
    result = _choose(report)
    entry = next(c for c in result["candidates"] if not c["pinned"] and c["ring"] == 3)
    assert "reason" in entry and not entry.get("eligible")
    assert result["selected"] != dict(mode="fast", pinned=False, ring=3)


@pytest.mark.parametrize("defect", ["cpu", "classic_drift", "worker_error", "short",
                                   "duplicate", "missing_hardware"])
def test_fail_closed_without_trustworthy_baseline(defect):
    report = _report()
    if defect == "cpu":
        report["configuration"]["cpu"] = True
    elif defect == "classic_drift":
        report["runs"][-1]["steps"][1]["updates"]["exact"] = 2
    elif defect == "worker_error":
        report["errors"].append({"error": "worker failed"})
    elif defect == "short":
        report["configuration"]["warmup"] = 2
    elif defect == "duplicate":
        report["runs"].append(report["runs"][0])
    elif defect == "missing_hardware":
        report["hardware"] = {}
    result = _choose(report)
    assert result["status"] == "inconclusive"
    assert result["selected"] is None


def test_no_qualified_speedup_is_not_a_recommendation():
    report = _report()
    for run in report["runs"]:
        if run["mode"] == "fast":
            for row in run["steps"]:
                row["seconds"] = 9.9
    result = _choose(report)
    assert result["status"] == "no_qualified_fast_strategy"
    assert result["selected"] is None


def test_aggressive_mode_requires_valid_staged_ring2_comparison():
    report = _report()
    for run in _runs(report, False, 2):
        report["runs"].remove(run)
    result = _choose(report)
    assert result["selected"] is None
    assert all(not c.get("eligible") for c in result["candidates"])


def test_larger_ring_must_beat_staged_ring2_even_if_ring2_misses_classic_threshold():
    report = _report()
    for run in report["runs"]:
        if run["mode"] == "fast":
            for row in run["steps"]:
                row["seconds"] = 9.6 if run["ring"] == 2 else 9.4
    result = _choose(report)
    assert result["selected"] is None
    assert result["status"] == "no_qualified_fast_strategy"
