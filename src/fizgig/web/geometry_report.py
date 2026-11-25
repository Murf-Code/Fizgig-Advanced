"""Persistent comparison status and consolidated logs. No GPU or web imports."""
import json
import logging
import re
import shutil
import threading
import time
import zipfile
from fizgig.web.geometry_benchmark import is_feasibility
from pathlib import Path

LOG = logging.getLogger(__name__)
_LOCK = threading.RLock()
REPORT = "matrix-geometry-report.json"
COMBINED = "matrix-geometry-comparison.log"
BUNDLE = "matrix-geometry-results.zip"
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
    lines = ["Fizgig Matrix Geometry optimizer pilot", "Status: " + report["status"],
             "Dataset: " + report["dataset"], "", "Case | Rank | MP | Status | updates | step ms | CUDA allocated GiB | CUDA reserved GiB | GPU peak GiB | RAM peak GiB | process shared GPU GiB | post-training validation"]
    for r in report["runs"]:
        perf = r.get("performance", {})
        val = r.get("validation", [])
        fit = r.get("feasibility") or {}
        config = r.get("settings", {})
        gib = lambda key: round(fit[key] / 2**30, 3) if key in fit else "-"
        lines.append(" | ".join(map(str, [r["case"], config.get("NETWORK_DIM", "-"),
                     config.get("DATASET_MEGAPIXELS", "-"), r["status"], fit.get("updates", perf.get("steps", "-")),
                     perf.get("step_ms", "-"), gib("cuda_peak_allocated_bytes"), gib("cuda_peak_reserved_bytes"),
                     perf.get("vram_peak_gb", "-"), perf.get("system_ram_peak_gib", "-"),
                     perf.get("process_shared_gpu_peak_gib", "-"),
                     val[-1]["loss"] if len(val) > 1 else "-"])))
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
        runs = [{"case": v["__geometry_compare"], "name": v["LORA_NAME"], "lr": v["LEARNING_RATE"],
                 "output": v["__geometry_compare"], "status": "queued", "logs": [],
                 "settings": {k: x for k, x in v.items() if not k.startswith("__")}} for v in configs]
        _save(root, {"version": 1, "dataset": str(dataset), "created_at": time.time(), "scope": "one-seed feasibility pilot; provisional optimizer rates", "runs": runs})


def parse_log(text):
    epochs, validation, feasibility = [], [], None
    buckets = []
    failure = None
    for line in text.splitlines():
        bucket = re.search(r"bucket: \((\d+), (\d+)\), count: (\d+)", line)
        if bucket:
            buckets.append({"width": int(bucket[1]), "height": int(bucket[2]), "count": int(bucket[3])})
        if re.search(r"CUDA out of memory|torch\.(?:cuda\.)?OutOfMemoryError|CUDA error: out of memory", line, re.I):
            failure = "cuda_oom"
        m = re.search(r"epoch (\d+)/(\d+)\s+avr_loss=" + NUM + r"\s+step=(\d+)\s+" + NUM + r"s/step", line)
        if m:
            epochs.append({"epoch": int(m[1]), "step": int(m[4]), "loss": float(m[3]), "seconds_per_step": float(m[5])})
        m = re.search(r"\[validation\] step (\d+) \(([^)]+)\): loss " + NUM, line)
        if m:
            validation.append({"step": int(m[1]), "phase": m[2], "loss": float(m[3])})
        m = re.search(r"\[geometry-fit\] complete updates=(\d+) allocated_bytes=(\d+) reserved_bytes=(\d+) wall_seconds=" + NUM, line)
        if m:
            feasibility = {"updates": int(m[1]), "cuda_peak_allocated_bytes": int(m[2]),
                           "cuda_peak_reserved_bytes": int(m[3]), "training_wall_seconds": float(m[4])}
    return {"epochs": epochs, "validation": validation, "feasibility": feasibility,
            "reported_buckets": buckets, "failure_kind": failure}


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
            if is_feasibility(case) and status == "finished" and not run.get("feasibility"):
                run.update(status="failed", reason="Missing feasibility completion telemetry; do not infer GPU fit.")
            if status == "failed" and run.get("failure_kind") == "cuda_oom":
                run["reason"] = "CUDA out of memory (fixed settings; no automatic reduction)."
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
        LOG.exception("Could not update matrix geometry comparison report for %s", case)
