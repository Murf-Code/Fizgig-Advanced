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
                for name in run.get("samples", []) + run.get("logs", []):
                    path = root / name
                    if path.is_file():
                        bundle.write(path, name)
        tmp.replace(root / BUNDLE)


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
    for line in text.splitlines():
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
    return {"epochs": epochs, "validation": validation, "probes": sorted(probes, key=lambda p: p["step"]),
            "rotations": rotations, "gradient_updates": gradient_updates}


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
