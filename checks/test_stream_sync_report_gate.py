"""Regression tests for the report gate without requiring CUDA or a model."""
import importlib.util
from pathlib import Path
import copy

import pytest


path = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_stream_sync.py"
spec = importlib.util.spec_from_file_location("diagnose_stream_sync", path)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def _fixture():
    planned = diag.select_cases(diag.cases(
        type("Args", (), {"suite": "expanded", "cpu": False, "model": "checkpoint"})()),
        ["J_fast_staged_ring3"])
    metric = dict(tensors=3, exact=3, finite=True, relative_l2=0.0, max_abs=0.0)
    report = dict(configuration=dict(steps=2, cpu=False), errors=[], runs=[])
    for case in planned:
        rows = [dict(step=i, loss=1.0, validation=2.0, loss_abs_error=0.0,
                     validation_abs_error=0.0, gradients=dict(metric), updates=dict(metric))
                for i in (1, 2)]
        report["runs"].append(dict(**case, status="completed", steps=rows))
    return report, planned


def test_selection_includes_reference_and_repeat():
    _, planned = _fixture()
    assert [c["label"] for c in planned] == [
        "A_classic_pageable", "D_classic_repeat", "J_fast_staged_ring3"]


def test_math_selection_uses_its_own_reference():
    args = type("Args", (), {"suite": "expanded", "cpu": False, "model": "checkpoint"})()
    selected = diag.select_cases(diag.cases(args), ["N_fast_math"])
    assert [c["label"] for c in selected] == [
        "L_classic_math", "M_classic_math_repeat", "N_fast_math"]
    with pytest.raises(ValueError):
        diag.select_cases(diag.cases(args), ["unknown"])


def test_exact_parity_and_cpu_scope():
    report, planned = _fixture()
    assert diag.evaluate_exact_parity(report, planned)["passed"]
    report["configuration"]["cpu"] = True
    assert diag.evaluate_exact_parity(report, planned)["scope"] == "cpu-smoke-only"


@pytest.mark.parametrize("defect", ["missing_case", "missing_step", "gradient_drift",
                                   "update_drift", "validation_drift", "loss_drift",
                                   "nonfinite", "zero_tensors", "worker_failure",
                                   "classic_repeat_drift", "duplicate"])
def test_gate_rejects_invalid_evidence(defect):
    report, planned = _fixture()
    row = report["runs"][-1]["steps"][0]
    if defect == "missing_case":
        report["runs"].pop()
    elif defect == "missing_step":
        report["runs"][-1]["steps"].pop()
    elif defect == "gradient_drift":
        row["gradients"]["exact"] = 2
    elif defect == "update_drift":
        row["updates"]["relative_l2"] = 0.001
    elif defect == "validation_drift":
        row["validation_abs_error"] = 0.001
    elif defect == "loss_drift":
        row["loss_abs_error"] = 0.001
    elif defect == "nonfinite":
        row["loss"] = float("nan")
    elif defect == "zero_tensors":
        row["gradients"].update(tensors=0, exact=0)
    elif defect == "worker_failure":
        report["errors"].append({"error": "worker died"})
    elif defect == "classic_repeat_drift":
        report["runs"][1]["steps"][0]["gradients"]["exact"] = 1
    elif defect == "duplicate":
        report["runs"].append(copy.deepcopy(report["runs"][-1]))
    result = diag.evaluate_exact_parity(report, planned)
    assert not result["passed"]
    assert result["issues"]
