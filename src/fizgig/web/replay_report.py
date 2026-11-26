"""Persistent comparison status and consolidated logs. No GPU or web imports."""
import ast
import json
import logging
import math
import re
import shutil
import threading
import time
import zipfile
from pathlib import Path

LOG = logging.getLogger(__name__)
_LOCK = threading.RLock()
REPORT = "global-replay-report.json"
COMBINED = "global-replay-comparison.log"
BUNDLE = "global-replay-results.zip"
TERMINAL = {"finished", "failed", "stopped", "removed"}
NUM = r"([0-9.eE+-]+)"


def _atomic(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save(root, report):
    statuses = [r["status"] for r in report["runs"]]
    report["status"] = ("complete" if all(s == "finished" for s in statuses)
                        else ("partial" if any(s == "finished" for s in statuses) else "failed")
                        if all(s in TERMINAL for s in statuses)
                        else "partial" if any(s in ("blocked", "stopped", "paused", "failed", "removed") for s in statuses)
                        else "in_progress")
    if any(r["case"].startswith("causal-") for r in report["runs"]):
        from fizgig.krea2.replay_causal import compare_runs
        report["causal_comparisons"] = compare_runs(report["runs"])
        from fizgig.krea2.replay_causal import compare_decay
        report["decay_comparisons"] = compare_decay(report["runs"])
    if any(r["case"].startswith("calibration-") for r in report["runs"]):
        report["experiment"] = "5C"
        report["calibration_summary"] = calibration_summary(report["runs"])
    multi = [r for r in report["runs"] if r["case"].startswith("movement-multi-")]
    if multi:
        report["experiment"] = "6C.2k-D"
        panels = {q["panel_hash"] for r in multi for q in r.get("movement_multi", {}).get("probes", [])}
        report["movement_multi_summary"] = {"completed_probes": sum(
            len(r.get("movement_multi", {}).get("probes", [])) for r in multi),
            "same_panel_across_seeds": len(panels) == 1 if all(r["status"] == "finished" for r in multi) else None,
            "scope": "development_target_only"}
    history = [r for r in report["runs"] if r["case"].startswith("history-")]
    if history:
        report["experiment"] = "6F.2"
        panels = {r["historical_reversion"]["panel_hash"] for r in history if "historical_reversion" in r}
        report["history_summary"] = {"completed_runs": sum("historical_reversion" in r for r in history),
            "same_panel_across_seeds": len(panels) == 1 if all(r["status"] == "finished" for r in history) else None,
            "scope": "development_target_only"}
    predictors = [r for r in report["runs"] if r["case"].startswith("predictors-")]
    if predictors:
        report["experiment"] = "6F.3"
        panels = {p["panel_hash"] for r in predictors for p in r.get("predictors", [])}
        report["predictor_summary"] = {"completed_probes": sum(len(r.get("predictors", [])) for r in predictors),
            "same_panel": len(panels) == 1 if all(r["status"] == "finished" for r in predictors) else None,
            "scope": "development_target_only"}
    allocation = [r for r in report["runs"] if r["case"].startswith("allocation-")]
    if allocation:
        report["experiment"] = "7C/7D"
        report["allocation_summary"] = {"completed_runs": sum("allocation" in r for r in allocation),
            "scope": "development_target_only", "momentum_policy": "preserve",
            "matching_scope": "local_same_parent_proposal"}
    report["updated_at"] = time.time()
    _atomic(root / REPORT, json.dumps(report, indent=2, allow_nan=False))
    lines = ["Fizgig Global Replay comparison", "Status: " + report["status"],
             "Dataset: " + report["dataset"], "", "Case | LR | Status | step ms | final validation | GPU peak GiB | RAM peak GiB"]
    for r in report["runs"]:
        perf = r.get("performance", {})
        val = r.get("validation", [])
        lines.append(" | ".join(map(str, [r["case"], r["lr"], r["status"], perf.get("step_ms", "-"),
                     val[-1]["loss"] if val else "-", perf.get("vram_peak_gb", "-"), perf.get("system_ram_peak_gib", "-")])))
    for r in report["runs"]:
        lines.extend(["", "=" * 78, r["case"] + " | " + r["status"], "=" * 78])
        if r.get("reason"):
            lines.append(r["reason"])
        for f in r.get("logs", []):
            try:
                lines.append((root / f).read_text(encoding="utf-8", errors="replace"))
            except OSError:
                lines.append("Log unavailable: " + f)
    _atomic(root / COMBINED, "\n".join(lines) + "\n")
    if all(s in TERMINAL for s in statuses):
        tmp = root / (BUNDLE + ".tmp")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
            for name in (REPORT, COMBINED):
                bundle.write(root / name, name)
            for run in report["runs"]:
                diagnostics = [f"{run['output']}/{name}" for name in
                               ("movement-pilot.json", "movement-isolation.json", "movement-dual.json",
                                "movement-dual-step04.json", "movement-dual-step08.json", "movement-dual-step12.json", "historical-reversion.json",
                                "predictors-step04.json", "predictors-step08.json", "predictors-step12.json")]
                for name in dict.fromkeys(run.get("samples", []) + run.get("logs", []) + diagnostics):
                    path = root / name
                    if path.is_file():
                        bundle.write(path, name)
        tmp.replace(root / BUNDLE)


def calibration_summary(runs):
    """Full precision scores, participation and common initialization checks."""
    out = {}
    initial = None
    for run in runs:
        if not run["case"].startswith("calibration-") or run["status"] != "finished":
            continue
        rows = run.get("research", [])
        hashes = [x for x in rows if x.get("kind") == "hash"]
        if not hashes:
            continue
        if initial is None:
            initial = hashes[0]["sha256"]
        row = {"lr": float(run["lr"]), "initial_hash_match": hashes[0]["sha256"] == initial,
               "block_updates": sum(len(x["active_blocks"]) for x in hashes[1:])}
        for kind in ("raw_validation", "ema_validation"):
            vals = [x for x in rows if x.get("kind") == kind]
            if vals:
                best = min(vals, key=lambda x: x["loss"])
                row[kind] = {"final": vals[-1]["loss"], "best": best["loss"], "best_step": best["step"]}
        out[run["case"]] = row
    return out


def create(root, dataset, configs):
    root = Path(root)
    with _LOCK:
        root.mkdir(parents=True, exist_ok=False)
        runs = [{"case": v["__replay_compare"], "name": v["LORA_NAME"], "lr": v["LEARNING_RATE"],
                 "output": v["__replay_compare"], "status": "queued", "logs": [],
                 "settings": {k: x for k, x in v.items() if not k.startswith("__")}} for v in configs]
        _save(root, {"version": 1, "dataset": str(dataset), "created_at": time.time(), "runs": runs})


def parse_log(text):
    epochs, validation, probes, rotations, gradient_updates = [], [], [], [], []
    research = []
    for line in text.splitlines():
        if "[replay-research] " in line:
            try:
                research.append(json.loads(line.split("[replay-research] ", 1)[1]))
            except ValueError:
                pass
        m = re.search(r"epoch (\d+)/(\d+)\s+avr_loss=" + NUM + r"\s+step=(\d+)\s+" + NUM + r"s/step", line)
        if m:
            epochs.append({"epoch": int(m[1]), "step": int(m[4]), "loss": float(m[3]), "seconds_per_step": float(m[5])})
        m = re.search(r"\[validation\] step (\d+) \(([^)]+)\): loss " + NUM, line)
        if m:
            validation.append({"step": int(m[1]), "phase": m[2], "loss": float(m[3])})
        m = re.search(r"\[selective-replay\] probe (\d+): (\d+)/(\d+) blocks active", line)
        if m:
            probes.append({"step": int(m[1]), "active": int(m[2]), "total": int(m[3])})
        m = re.search(r"\[selective-replay\] active_blocks=(\[.*?\]) scores=(\[.*?\]) strategy=(\w+)(?: step=(\d+))?", line)
        if m:
            try:
                active = ast.literal_eval(m[1])
                scores = ast.literal_eval(m[2])
                step = int(m[4]) if m[4] else None
                if step is not None:
                    record = next((p for p in probes if p["step"] == step), None)
                    if record is None:
                        record = {"step": step, "active": len(active), "total": len(scores)}
                        probes.append(record)
                elif probes and "active_blocks" not in probes[-1]:
                    record = probes[-1]
                else:
                    record = None
                if record is not None:
                    record.update(active_blocks=active, scores=scores, strategy=m[3])
            except (ValueError, SyntaxError):
                pass
        m = re.search(r"\[selective-replay\] rotation next_step=(\d+) active_blocks=(\[.*?\]) strategy=(\w+)", line)
        if m:
            try:
                rotations.append({"next_step": int(m[1]), "active_blocks": ast.literal_eval(m[2]), "strategy": m[3]})
            except (ValueError, SyntaxError):
                pass
        m = re.search(r"\[selective-replay\] gradient_updates step=(\d+) counts=(\[.*?\])", line)
        if m:
            try:
                gradient_updates.append({"step": int(m[1]), "counts": ast.literal_eval(m[2])})
            except (ValueError, SyntaxError):
                pass
    out = {"epochs": epochs, "validation": validation, "probes": sorted(probes, key=lambda p: p["step"]),
           "rotations": rotations, "gradient_updates": gradient_updates}
    if research:
        out["research"] = research
    return out


def update(output, case, status, *, log_path=None, reason="", wall_seconds=None):
    """Best effort for older queues without a manifest; never break training to write a report."""
    if not case or not output:
        return
    output = Path(output)
    root = output.parent
    if not (root / REPORT).is_file():
        return
    try:
        with _LOCK:
            report = _read_json(root / REPORT)
            run = next(r for r in report["runs"] if r["case"] == case)
            run.update(status=status, reason=reason)
            if status == "running":
                run["started_at"] = time.time()
            if wall_seconds is not None:
                run["wall_seconds"] = round(wall_seconds, 2)
            if status in TERMINAL or status == "paused":
                run["ended_at"] = time.time()
            if log_path and Path(log_path).is_file():
                output.mkdir(parents=True, exist_ok=True)
                dest = output / ("comparison-run-%02d.log" % (len(run["logs"]) + 1))
                shutil.copyfile(log_path, dest)
                run["logs"].append(dest.relative_to(root).as_posix())
                raw = "\n".join((root / f).read_text(encoding="utf-8", errors="replace") for f in run["logs"])
                run.update(parse_log(raw))
                if status == "finished" and case.startswith("repro-"):
                    rows = run.get("research", [])
                    hashes = [r.get("step") for r in rows if r.get("kind") == "hash"]
                    checks = [r.get("step") for r in rows if r.get("kind") == "raw_validation"]
                    ema_checks = [r.get("step") for r in rows if r.get("kind") == "ema_validation"]
                    if (hashes != list(range(59)) or checks != [29, 58]
                            or not any(r.get("kind") == "complete" and r.get("step") == 58 for r in rows)
                            or (case == "repro-strict-telemetry-lr450" and ema_checks != [29, 58])):
                        run.update(status="failed", reason="Incomplete Experiment 5A diagnostics; do not compare this run.")
                if status == "finished" and (case == "movement-dual-lr450" or case in
                        ("movement-multi-seed42-lr450", "movement-multi-seed43-lr450")):
                    try:
                        multi = case.startswith("movement-multi-")
                        probe_steps = (4, 8, 12) if multi else (8,)
                        rows = run.get("research", [])
                        hashes = [r["step"] for r in rows if r.get("kind") == "hash"]
                        completed = [r for r in rows if r.get("kind") == "complete"]
                        markers = [r for r in rows if r.get("kind") == "movement_dual"]
                        probes = []
                        for probe_step in probe_steps:
                            filename = f"movement-dual-step{probe_step:02d}.json" if multi else "movement-dual.json"
                            pilot = _read_json(output / filename)
                            expected = {"uniform", "static26", "static13", "adaptive_low_rms", "adaptive_movement"}
                            matching = pilot["matching"]
                            aliases = pilot["aliases"]
                            geometry = pilot["endpoint_geometry"]
                            names = set(aliases.values())
                            count = 4 * len(pilot["probe_units"])
                            if (pilot["status"] != "completed" or pilot["step"] != probe_step
                                    or pilot["experiment"] != "6C.2k-C" or not pilot["integrity_passed"]
                                    or pilot["scope"] != "development_target_only"
                                    or pilot["parent_rollback"] != "exact_registered_state"
                                    or not matching["matched"] or set(matching["arms"]) != expected
                                    or set(aliases) != expected | {"decay", "quarter", "half", "full"}
                                    or set(geometry) != names or not count
                                    or set(pilot["evaluators"]) != {"native", "weight_only"}
                                    or any(not v["decay_repeat_exact"] or set(v["restored"]) != names
                                           or set(v["arms"]) != names
                                           or any(len(a["rows"]) != count for a in v["arms"].values())
                                           for v in pilot["evaluators"].values())
                                    or max(geometry[aliases[n]]["fp32"] for n in expected)
                                       - min(geometry[aliases[n]]["fp32"] for n in expected) > matching["tolerance"]
                                    or len(completed) != 1 or not max(probe_steps) <= completed[0]["step"] <= 40
                                    or hashes != list(range(completed[0]["step"] + 1))
                                    or [m["step"] for m in markers] != list(probe_steps)
                                    or any(m.get("status") != "completed" or not m.get("integrity")
                                           or not m.get("matched") or m.get("rollback") != "exact_registered_state" for m in markers)):
                                raise ValueError("incomplete dual diagnostics")
                            probes.append({"report": filename, "step": probe_step,
                                "panel_hash": pilot["panel_hash"], "selected_blocks": pilot["selected_blocks"],
                                "unique_endpoints": len(names), "integrity": True, "matched": True})
                        if len({q["panel_hash"] for q in probes}) != 1:
                            raise ValueError("evaluation panel changed between checkpoints")
                        if multi:
                            run["movement_multi"] = {"probes": probes, "integrity": True, "matched": True}
                        else:
                            run["movement_dual"] = {"report": "movement-dual.json",
                                "integrity": True, "matched": True, "unique_endpoints": len(names)}
                    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                        run.update(status="failed", reason="Invalid movement dual diagnostics: " + str(exc))
                if status == "finished" and case.startswith("allocation-"):
                    try:
                        rows = run.get("research", [])
                        allocations = [r for r in rows if r.get("kind") == "allocation"]
                        hashes = [r["step"] for r in rows if r.get("kind") == "hash"]
                        complete = [r["step"] for r in rows if r.get("kind") == "complete"]
                        mode = case.split("-")[1]
                        if ([a["step"] for a in allocations] != list(range(1,14))
                                or hashes != list(range(14)) or complete != [13]):
                            raise ValueError("incomplete allocation trajectory")
                        for a in allocations:
                            ms = a["multipliers"]
                            target, actual, native = (a[k] for k in ("target_movement","actual_movement","native_movement"))
                            attenuated = set(range((a["step"]-1)%2,28,2))
                            if (a["mode"] != mode or a["momentum_policy"] != "preserve"
                                    or a["gradient_blocks"] != list(range(28))
                                    or set(a["attenuated_blocks"]) != attenuated
                                    or set(ms) != {str(b) for b in range(28)}
                                    or not all(math.isfinite(x) and x > 0 for x in (target,actual,native))
                                    or abs(actual-target) > max(1e-12,.001*target)
                                    or any(not 0 < v <= 1.5 for v in ms.values())):
                                raise ValueError("invalid allocation movement or coverage")
                            c = a["common_scale"]
                            expected = {str(b): (c*(.5 if b in attenuated else 1.)
                                        if mode in ("rotate","compensated") else c) for b in range(28)}
                            if (any(abs(ms[b]-v)>1e-10 for b,v in expected.items())
                                    or (mode in ("rotate","full") and c != 1.)
                                    or (mode in ("full","compensated") and target != native)):
                                raise ValueError("allocation multiplier or target mismatch")
                        run["allocation"] = {
                            "updates":len(allocations),
                            "mean_local_movement_ratio":sum(a["movement_ratio"] for a in allocations)/13,
                            "max_relative_gap":max(a["relative_gap"] for a in allocations),
                            "mean_geometry_step_seconds":sum(a["seconds"] for a in allocations)/13,
                            "cumulative_movement":sum(a["actual_movement"] for a in allocations)}
                    except (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as exc:
                        run.update(status="failed", reason="Invalid allocation diagnostics: " + str(exc))
                if status == "finished" and case.startswith("predictors-"):
                    try:
                        rows = run.get("research", [])
                        hashes = {r["step"]: r["sha256"] for r in rows if r.get("kind") == "hash"}
                        markers = [r for r in rows if r.get("kind") == "predictors"]
                        completed = [r for r in rows if r.get("kind") == "complete"]
                        expected_steps = (4, 8, 12)
                        if ([m["step"] for m in markers] != list(expected_steps)
                                or len(completed) != 1 or sorted(hashes) != list(range(completed[0]["step"]+1))):
                            raise ValueError("missing predictor checkpoint markers")
                        summaries = []
                        blocks = {str(b) for b in range(28)}
                        for step, marker in zip(expected_steps, markers):
                            filename = f"predictors-step{step:02d}.json"
                            p = _read_json(output / filename)
                            modes = p["activations"]
                            count = 4*len(p["probe_units"])
                            if (p["experiment"] != "6F.3" or p["status"] != "completed"
                                    or p["parent_rollback"] != "exact_registered_state" or p["step"] != step
                                    or p["panel_hash"] != marker["panel_hash"]
                                    or p["parent_fingerprint"] != hashes[step-1]
                                    or p["post_step_fingerprint"] != hashes[step]
                                    or set(p["update"]) != blocks or set(modes) != {"native", "weight_only"}
                                    or any(len(rs) != count or any(set(r["blocks"]) != blocks for r in rs)
                                           for rs in modes.values())
                                    or [r["input_hash"] for r in modes["native"]] != [r["input_hash"] for r in modes["weight_only"]]):
                                raise ValueError("incomplete predictor data or trajectory mismatch")
                            summaries.append({"step": step, "report": filename, "panel_hash": p["panel_hash"],
                                              "parent_fingerprint": p["parent_fingerprint"],
                                              "post_step_fingerprint": p["post_step_fingerprint"]})
                        run["predictors"] = summaries
                    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                        run.update(status="failed", reason="Invalid predictor diagnostics: " + str(exc))
                if status == "finished" and case in ("history-seed42-lr450", "history-seed43-lr450"):
                    try:
                        h = _read_json(output / "historical-reversion.json")
                        rows = run.get("research", [])
                        hashes = [r["step"] for r in rows if r.get("kind") == "hash"]
                        completed = [r for r in rows if r.get("kind") == "complete"]
                        captures = [r for r in rows if r.get("kind") == "historical_capture"]
                        markers = [r for r in rows if r.get("kind") == "historical_reversion"]
                        primary = {f"block{b:02d}" for b in range(28)} | {"all_early"} | {f"half{b:02d}" for b in (13, 17, 26)}
                        matched_names = {f"uniform_{n}" for n in ("block13", "block17", "block26", "half13", "half17", "half26")}
                        expected = primary | matched_names
                        count = 4 * len(h["probe_units"])
                        if (h["experiment"] != "6F.2" or h["status"] != "completed" or h["step"] != 12
                                or h["source_step"] != 3 or h["parent_step"] != 11 or not h["integrity_passed"]
                                or h["scope"] != "development_target_only" or h["semantic_preservation"]
                                or h["parent_rollback"] != "exact_registered_state"
                                or set(h["geometry"]) != expected or set(h["matching"]) != matched_names
                                or not all(m["matched"] for m in h["matching"].values()) or not count
                                or set(h["evaluators"]) != {"native", "weight_only"}
                                or any(set(v["arms"]) != expected or set(v["restored"]) != expected
                                       or not v["repeat_exact"] or not v["reversion_repeat_exact"]
                                       or any(len(a["rows"]) != count for a in v["arms"].values())
                                       for v in h["evaluators"].values())
                                or len(captures) != 1 or captures[0]["step"] != 3
                                or captures[0]["sha256"] != h["source_hash"]
                                or len(markers) != 1 or markers[0]["step"] != 12
                                or markers[0]["source_hash"] != h["source_hash"]
                                or not markers[0]["integrity"] or markers[0]["status"] != "completed"
                                or markers[0]["rollback"] != "exact_registered_state"
                                or len(completed) != 1 or not 12 <= completed[0]["step"] <= 40
                                or hashes != list(range(completed[0]["step"] + 1))):
                            raise ValueError("incomplete historical reversion diagnostics")
                        run["historical_reversion"] = {"report": "historical-reversion.json",
                            "panel_hash": h["panel_hash"], "integrity": True, "arms": len(expected)}
                    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                        run.update(status="failed", reason="Invalid historical reversion diagnostics: " + str(exc))
                if status == "finished" and case == "movement-isolation-lr450":
                    try:
                        pilot = _read_json(output / "movement-isolation.json")
                        rows = run.get("research", [])
                        hashes = [r["step"] for r in rows if r.get("kind") == "hash"]
                        completed = [r for r in rows if r.get("kind") == "complete"]
                        markers = [r for r in rows if r.get("kind") == "movement_isolation"]
                        required = {"repeat", "exact_rewrite", "after_decay", "after_base",
                                    "repeat_decay_reversed", "final_native_parent", "weight_only_restored",
                                    "fp32_lora_restored", "combined_restored", "after_precision_modes",
                                    "after_strength", "after_trace"}
                        if (pilot["status"] != "completed" or pilot["step"] != 8
                                or pilot["experiment"] != "6C.2k-B" or not pilot["integrity_passed"]
                                or pilot["parent_rollback"] != "exact_registered_state"
                                or not required <= set(pilot["integrity"])
                                or not all(pilot["integrity"][k] and all(r["equal"] for r in pilot["integrity"][k]) for k in required)
                                or len(pilot["factor_rounding"]) != 224
                                or set(pilot["precision"]) != {"weight_only", "fp32_lora", "combined"}
                                or not pilot["trace"] or len(completed) != 1 or not 8 <= completed[0]["step"] <= 40
                                or hashes != list(range(completed[0]["step"] + 1))
                                or len(markers) != 1 or markers[0]["step"] != 8):
                            raise ValueError("incomplete isolation diagnostics")
                        run["movement_isolation"] = {"report": "movement-isolation.json",
                            "integrity": True, "trace_valid": all(x["matches_uninstrumented"] for x in pilot["trace"])}
                    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                        run.update(status="failed", reason="Invalid movement isolation diagnostics: " + str(exc))
                if status == "finished" and case == "movement-sensitivity-lr450":
                    try:
                        pilot = _read_json(output / "movement-pilot.json")
                        rows = run.get("research", [])
                        hashes = [r["step"] for r in rows if r.get("kind") == "hash"]
                        completed = [r for r in rows if r.get("kind") == "complete"]
                        markers = [r for r in rows if r.get("kind") == "movement_pilot"]
                        if (pilot["status"] != "completed" or pilot["step"] != 8
                                or pilot["scope"] != "development_target_only"
                                or pilot["parent_rollback"] != "exact_registered_state"
                                or set(pilot["sweep"]) != {"0.0", "0.25", "0.5", "1.0"}
                                or len(completed) != 1 or not 8 <= completed[0]["step"] <= 40
                                or hashes != list(range(completed[0]["step"] + 1))
                                or len(markers) != 1 or markers[0]["step"] != 8):
                            raise ValueError("incomplete pilot diagnostics")
                        run["movement_pilot"] = {"report": "movement-pilot.json",
                            "matched": pilot["matching"]["matched"], "repeat_exact": pilot["repeat_exact"]}
                    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                        run.update(status="failed", reason="Invalid movement pilot diagnostics: " + str(exc))
                if status == "finished" and case.startswith(("causal-", "calibration-")):
                    from fizgig.krea2.replay_causal import (ALL, PROBES, SOURCE, checked_mask,
                                                            expected_mask, intervention, source_schedule)
                    from fizgig.web.replay_benchmark import (CAUSAL_ROUTES, CAUSAL_DECAYS, UNRESTRICTED_CASES,
                                                             CALIBRATION_CASES)
                    from fizgig.krea2.selective_policy import select_blocks
                    try:
                        calibration = case in CALIBRATION_CASES
                        expected_lr = float(CALIBRATION_CASES[case][0]) if calibration else 0.00045
                        strategy = CALIBRATION_CASES[case][1] if calibration else "adaptive_gradient"
                        rows = run.get("research", [])
                        hashes = [r for r in rows if r.get("kind") == "hash"]
                        if [r["step"] for r in hashes] != list(range(146)):
                            raise ValueError("missing fingerprints")
                        for kind in ("raw_validation", "ema_validation"):
                            if [r["step"] for r in rows if r.get("kind") == kind] != [29, 58, 87, 116, 145]:
                                raise ValueError("missing " + kind)
                        telemetry = [r for r in rows if r.get("kind") == "probe_telemetry"]
                        if [r["step"] for r in telemetry] != list(PROBES):
                            raise ValueError("missing probe telemetry")
                        for probe in telemetry:
                            factors = probe["factors"]
                            if (len(factors) != 56 or {(f["block"], f["factor"]) for f in factors}
                                    != {(b, k) for b in ALL for k in ("A", "B")}
                                    or any(not {"decay_update_rms", "muon_update_rms_reconstructed",
                                                "gradient_momentum_cosine", "momentum_agreement_matrices",
                                                "gradient_bearing_matrices"} <= f.keys() for f in factors)):
                                raise ValueError("incomplete expanded factor telemetry")
                        if not any(r.get("kind") == "complete" and r.get("step") == 145 for r in rows):
                            raise ValueError("missing completion marker")
                        env = [r for r in rows if r.get("kind") == "environment"]
                        if len(env) != 1 or env[0].get("mode") != ("calibration" if calibration else "causal") or not env[0].get("deterministic"):
                            raise ValueError("not strict deterministic causal execution")
                        opt = [r for r in rows if r.get("kind") == "optimizer"]
                        decay = 0.01 if calibration else CAUSAL_DECAYS[case]
                        if (len(opt) != 1 or not opt[0].get("groups")
                                or any(g.get("weight_decay") != decay or g.get("lr") != expected_lr
                                       or g.get("momentum") != 0.95 or g.get("nesterov") is not True
                                       or g.get("ns_steps") != 5 or g.get("adjust_lr_fn") != "match_rms_adamw"
                                       for g in opt[0]["groups"])):
                            raise ValueError("incorrect or missing optimizer decay")
                        for row in hashes[1:]:
                            rates = row.get("optimizer_rates", [])
                            if not rates or any(g.get("weight_decay") != decay or g.get("lr") != expected_lr for g in rates):
                                raise ValueError("incorrect per-step decay")
                        run["weight_decay"] = decay
                        if case in UNRESTRICTED_CASES or (calibration and strategy == "replay"):
                            masks = [set(ALL) for _ in PROBES]
                        elif case == SOURCE or calibration:
                            probes = run["probes"]
                            if [p["step"] for p in probes] != list(PROBES):
                                raise ValueError("missing source probes")
                            masks = [checked_mask(p["active_blocks"]) for p in probes]
                            if any(p["active"] != len(mask) or p["total"] != 28
                                   or p["strategy"] != strategy for p, mask in zip(probes, masks)):
                                raise ValueError("invalid source probes")
                            if calibration and strategy == "random":
                                for i, mask in enumerate(masks):
                                    if mask != select_blocks("random", [None] * 28, 14, i):
                                        raise ValueError("Random mask differs from seeded fixed-budget schedule")
                        else:
                            source = source_schedule(root / REPORT)
                            masks = [intervention(CAUSAL_ROUTES[case], source, i) for i in range(len(PROBES))]
                        for row in hashes[1:]:
                            if checked_mask(row["active_blocks"]) != expected_mask(masks, row["step"]):
                                raise ValueError(f"unexpected actual routing at step {row['step']}")
                    except (RuntimeError, ValueError, KeyError, TypeError, IndexError) as exc:
                        run.update(status="failed", reason="Invalid replay research diagnostics: " + str(exc))
            perf = output / (".perf_" + run["name"] + ".json")
            if perf.is_file():
                try:
                    run["performance"] = _read_json(perf).get("summary", {})
                except (OSError, ValueError):
                    pass
            run["samples"] = sorted(f.relative_to(root).as_posix() for f in (output / "sample").rglob("*")
                                    if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
            val = run.get("validation", [])
            if len(val) > 1 and run.get("wall_seconds", 0) > 0:
                run["validation_loss_reduction_per_wall_minute"] = (val[0]["loss"] - val[-1]["loss"]) * 60 / run["wall_seconds"]
            _save(root, report)
    except (OSError, ValueError, KeyError, StopIteration, TypeError):
        LOG.exception("Could not update replay comparison report for %s", case)
