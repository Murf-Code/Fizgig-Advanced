"""Experiment 5B: report-derived routing interventions; no training RNG."""
import json
import random
from pathlib import Path

SOURCE = "causal-gradient-lr450"
ROUTES = ("random", "exact", "block26", "block13")
STEPS = 145
N = 28
PROBES = tuple(range(8, STEPS + 1, 8))
ALL = set(range(N))


def checked_mask(values):
    if (not isinstance(values, list) or not values
            or any(type(i) is not int or i not in ALL for i in values)
            or len(set(values)) != len(values)):
        raise ValueError("invalid block mask")
    return set(values)


def expected_mask(masks, step):
    if not 1 <= step <= STEPS:
        raise ValueError("invalid optimizer step")
    return set(ALL) if step <= 8 or step % 8 == 0 else set(masks[(step - 1)//8 - 1])


def source_schedule(report_path):
    """Validate the completed source, including every actual gradient-bearing mask."""
    try:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        run = next(r for r in report["runs"] if r["case"] == SOURCE)
        if run["status"] != "finished" or run["epochs"][-1]["step"] != STEPS:
            raise ValueError("source is not a finished 145-update run")
        if run["settings"]["SEED"] != "42" or run["lr"] != "0.00045":
            raise ValueError("source settings differ")
        rows = run["research"]
        env = [r for r in rows if r.get("kind") == "environment"]
        if len(env) != 1 or not env[0].get("deterministic") or env[0].get("mode") != "causal":
            raise ValueError("source did not use strict causal mode")
        hashes = [r for r in rows if r.get("kind") == "hash"]
        if [r["step"] for r in hashes] != list(range(STEPS + 1)):
            raise ValueError("source fingerprints incomplete")
        probes = run["probes"]
        if [p["step"] for p in probes] != list(PROBES):
            raise ValueError("source probes incomplete")
        masks = []
        for p in probes:
            mask = checked_mask(p["active_blocks"])
            if p["active"] != len(mask) or p["total"] != N or p["strategy"] != "adaptive_gradient":
                raise ValueError("invalid source probe")
            masks.append(mask)
        for row in hashes[1:]:
            if checked_mask(row["active_blocks"]) != expected_mask(masks, row["step"]):
                raise ValueError(f"source actual mask differs from probe schedule at step {row['step']}")
        return masks
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration) as exc:
        raise RuntimeError("Experiment 5B requires the completed current Adaptive Gradient source: " + str(exc)) from exc


def intervention(route, masks, index):
    if route not in ROUTES or not 0 <= index < len(masks):
        raise ValueError("invalid causal intervention or probe index")
    source = masks[index]
    if route == "exact":
        return set(source)
    if route == "random":
        return set(random.Random(42 + index).sample(range(N), len(source)))
    block = 26 if route == "block26" else 13
    return ALL - {block} if 26 not in source else set(ALL)


def compare_runs(runs):
    """Compare complete runs; flag divergence before the intended intervention."""
    source = next((r for r in runs if r["case"] == SOURCE and r["status"] == "finished"), None)
    if source is None:
        return {}
    def hashes(run):
        return {r["step"]: r for r in run.get("research", []) if r.get("kind") == "hash"}
    src = hashes(source)
    out = {}
    for run in runs:
        if (run is source or run["status"] != "finished" or not run["case"].startswith("causal-")
                or run["case"].startswith("causal-decay")):
            continue
        other = hashes(run)
        if sorted(src) != list(range(STEPS + 1)) or sorted(other) != sorted(src):
            continue
        first_mask = next((s for s in range(1, STEPS + 1)
                           if src[s]["active_blocks"] != other[s]["active_blocks"]), None)
        first_hash = next((s for s in range(STEPS + 1) if src[s]["sha256"] != other[s]["sha256"]), None)
        out[run["case"]] = {
            "first_different_actual_mask_step": first_mask,
            "first_different_weight_hash_step": first_hash,
            "identical_hashes": sum(src[s]["sha256"] == other[s]["sha256"] for s in src),
            "different_block_update_slots": sum(len(set(src[s]["active_blocks"]) ^ set(other[s]["active_blocks"]))
                                                for s in range(1, STEPS + 1)),
            "source_block_updates": sum(len(src[s]["active_blocks"]) for s in range(1, STEPS + 1)),
            "control_block_updates": sum(len(other[s]["active_blocks"]) for s in range(1, STEPS + 1)),
            "pre_intervention_hash_match": first_hash is None or (first_mask is not None and first_hash >= first_mask),
        }
    return out


def compare_decay(runs):
    """Weight decay is an intervention at step 1, not a routing intervention."""
    baseline = next((r for r in runs if r["case"] == "causal-unrestricted-lr450"
                     and r["status"] == "finished"), None)
    if baseline is None:
        return {}
    def hashes(run):
        return {r["step"]: r for r in run.get("research", []) if r.get("kind") == "hash"}
    def validations(run, kind):
        return {r["step"]: r["loss"] for r in run.get("research", []) if r.get("kind") == kind}
    base = hashes(baseline)
    out = {}
    for run in runs:
        if not run["case"].startswith("causal-decay") or run["status"] != "finished":
            continue
        other = hashes(run)
        if sorted(base) != list(range(STEPS + 1)) or sorted(other) != sorted(base):
            continue
        out[run["case"]] = {
            "weight_decay": run.get("weight_decay"),
            "initial_hash_match": base[0]["sha256"] == other[0]["sha256"],
            "first_different_weight_hash_step": next((s for s in base if base[s]["sha256"] != other[s]["sha256"]), None),
            "identical_actual_masks": all(base[s]["active_blocks"] == other[s]["active_blocks"] for s in base),
            "raw_loss_difference_vs_unrestricted": {s: v - validations(baseline, "raw_validation")[s]
                for s, v in validations(run, "raw_validation").items()},
            "ema_loss_difference_vs_unrestricted": {s: v - validations(baseline, "ema_validation")[s]
                for s, v in validations(run, "ema_validation").items()},
        }
    return out
