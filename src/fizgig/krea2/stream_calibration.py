"""Conservative, offline selection from repeated real-GPU training diagnostics.

This module never allocates GPU memory, starts training, or changes preferences.
A recommendation is specific to the measured workload and is not a production
hardware profile. The caller must recheck live memory before applying it.
"""
import math
from statistics import median


def _number(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def _signature(case):
    return case["mode"], case["pinned"], case["ring"]


def recommend_stream_strategy(report, parity_check, *, min_gain=0.05,
                              max_spread=0.10, vram_slack_gib=0.25):
    """Choose only a repeatable, exact, memory-safe improvement.

    Compare worst candidate repeat against best reference repeat. A larger ring
    or fully pinned strategy must also beat the simpler eligible staged ring-2
    strategy by the same margin. CPU results cannot authorize a recommendation.
    Skipped alternatives do not invalidate completed pairs; worker errors do.
    """
    result = {"status": "inconclusive", "selected": None, "candidates": [],
              "min_gain": min_gain, "max_repeat_spread": max_spread,
              "vram_slack_gib": vram_slack_gib, "applied": False}

    def stop(reason):
        result["reason"] = reason
        return result

    cfg = report.get("configuration", {})
    if cfg.get("cpu") is not False:
        return stop("Real CUDA measurements required")
    if cfg.get("suite") != "calibration":
        return stop("Use the repeated calibration suite")
    steps, warmup = cfg.get("steps", 0), cfg.get("warmup", 0)
    if type(steps) is not int or type(warmup) is not int or warmup < 1 or steps - warmup < 3:
        return stop("Require at least one warmup and three measured steps per repeat")
    if report.get("errors"):
        return stop("Worker failures must be investigated before recommending a strategy")
    hardware = report.get("hardware", {})
    total_ram = hardware.get("ram_total_gib")
    total_vram = hardware.get("vram_total_gib")
    if not _number(total_ram, positive=True) or not _number(total_vram, positive=True):
        return stop("Missing hardware memory capacity")
    reserve = max(6.0, total_ram * 0.15)
    planned = report.get("planned_cases", [])
    runs = report.get("runs", [])
    by_label = {r["label"]: r for r in runs}
    if len(by_label) != len(runs):
        return stop("Duplicate diagnostic results")
    if len({c["label"] for c in planned}) != len(planned):
        return stop("Duplicate planned cases")
    groups = {}
    for case in planned:
        if case.get("attention") == "auto" and case.get("reference") == "auto" and case.get("mode") in ("classic", "fast"):
            groups.setdefault(_signature(case), []).append(case)
    baseline_key = ("classic", False, 2)
    baseline = groups.get(baseline_key, [])
    if len(baseline) != 2 or sum(bool(c["record"]) for c in baseline) != 1:
        return stop("Two classic pageable reference runs required")

    def evaluate(cases, *, reference=False):
        if len(cases) != 2:
            return None, "Two independent repeats required"
        pair = [by_label.get(c["label"]) for c in cases]
        if any(r is None for r in pair):
            return None, "Missing or skipped repeat"
        for case, run in zip(cases, pair):
            if any(run.get(k) != case.get(k) for k in ("mode", "pinned", "ring", "attention", "reference", "record")):
                return None, "Measured configuration differs from planned case"
        if not parity_check(report, baseline if reference else baseline + cases)["passed"]:
            return None, "Exact numerical parity not established"
        times, allocated, reserved, pinned, available = [], [], [], [], []
        for run in pair:
            rows = run["steps"][warmup:]
            if any(not _number(row.get("seconds"), positive=True) for row in rows):
                return None, "Invalid step timing"
            times.append(median(row["seconds"] for row in rows))
            metrics = [run.get("peak_allocated_gib"), run.get("peak_reserved_gib"),
                       run.get("pinned_storage_gib"),
                       run.get("memory_samples", {}).get("system_available_min_gib")]
            if any(not _number(x) for x in metrics):
                return None, "Missing or invalid memory measurement"
            a, r, p, m = metrics
            if a > r or r > total_vram or m < reserve:
                return None, "Memory reserve or GPU capacity exceeded"
            allocated.append(a)
            reserved.append(r)
            pinned.append(p)
            available.append(m)
            if not reference:
                if any(row.get("transfers", {}).get("d2h_bytes") != 0 for row in run["steps"]):
                    return None, "Send-only transfer recorded missing or nonzero D2H bytes"
                host = run.get("host_memory_plan") or {}
                if host.get("direct") is not run["pinned"] or (
                        not run["pinned"] and host.get("staging_slots") != run["ring"]):
                    return None, "Host allocation fallback or missing effective strategy"
        if max(times) / min(times) - 1 > max_spread:
            return None, "Repeat timing spread exceeds limit"
        return {"mode": cases[0]["mode"], "pinned": cases[0]["pinned"],
                "ring": cases[0]["ring"], "repeat_medians": times,
                "peak_allocated_gib": max(allocated), "peak_reserved_gib": max(reserved),
                "pinned_storage_gib": max(pinned), "min_available_ram_gib": min(available)}, None

    base, reason = evaluate(baseline, reference=True)
    if base is None:
        return stop("Classic reference invalid: " + reason)
    result["baseline"] = base
    eligible = []
    for key, cases in groups.items():
        if key == baseline_key or key[0] != "fast":
            continue
        mode, pinned, ring = key
        entry = {"mode": mode, "pinned": pinned, "ring": ring}
        result["candidates"].append(entry)
        if type(ring) is not int or not 2 <= ring <= 4 or ring > 28 - cfg.get("swap", 28):
            entry["reason"] = "Ring exceeds supported block-weight budget"
            continue
        measured, reason = evaluate(cases)
        if measured is None:
            entry["reason"] = reason
            continue
        entry.update(measured)
        if any(entry[k] > base[k] + vram_slack_gib for k in ("peak_allocated_gib", "peak_reserved_gib")):
            entry["reason"] = "Peak GPU memory exceeds classic allowance"
            continue
        gain = 1 - max(entry["repeat_medians"]) / min(base["repeat_medians"])
        entry["conservative_gain"] = gain
        if gain < min_gain:
            entry["reason"] = "Insufficient repeatable improvement over classic"
            continue
        eligible.append(entry)

    simple = next((e for e in eligible if not e["pinned"] and e["ring"] == 2), None)
    accepted = []
    for entry in eligible:
        if simple is not None and entry is not simple:
            gain = 1 - max(entry["repeat_medians"]) / min(simple["repeat_medians"])
            entry["gain_over_staged_ring2"] = gain
            if gain < min_gain:
                entry["reason"] = "Insufficient improvement over staged ring 2"
                continue
        entry["eligible"] = True
        accepted.append(entry)
    if not accepted:
        if not result["candidates"]:
            return stop("No send-only candidate was measured")
        result.update(status="no_qualified_fast_strategy",
                      reason="No repeatable, exact and memory-safe speedup established")
        return result
    # When candidates are within the margin, prefer less pinning and smaller rings.
    fastest = min(max(e["repeat_medians"]) for e in accepted)
    near = [e for e in accepted if max(e["repeat_medians"]) <= fastest / (1 - min_gain)]
    best = min(near, key=lambda e: (e["pinned"], e["ring"], max(e["repeat_medians"])))
    result.update(status="recommended", selected={k: best[k] for k in ("mode", "pinned", "ring")},
                  reason="Repeatable improvement with exact parity and measured memory limits",
                  workload_scope="real-checkpoint" if cfg.get("model") else "synthetic-only",
                  hardware=hardware, git_commit=report.get("git_commit"))
    return result
