"""Persistent comparison status and consolidated logs. No GPU or web imports."""
import ast
import json
import logging
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
                               ("movement-pilot.json", "movement-isolation.json")]
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
