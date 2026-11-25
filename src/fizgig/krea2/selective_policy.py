"""Budget-matched experimental block routing. No training RNG is consumed."""
import math
import random

STRATEGIES = ("downhill", "uphill", "high_gradient", "low_gradient", "random", "rotational", "static_early", "walking_coverage",
              "balanced_random", "static_first_half", "hybrid_coverage")


def select_blocks(strategy, scores, budget, probe_index, seed=42, *, walk_phase=0, update_counts=None):
    n = len(scores)
    if strategy not in STRATEGIES or not 1 <= budget <= n or probe_index < 0 or walk_phase < 0:
        raise ValueError("Invalid selective benchmark strategy, budget or probe index.")
    if strategy == "random":
        return set(random.Random(seed + probe_index).sample(range(n), budget))
    if strategy == "balanced_random":
        # At half budget, each shuffled pair covers all blocks exactly once.
        cycle, offset = divmod(probe_index * budget, n)
        order = list(range(n))
        random.Random(seed + cycle).shuffle(order)
        return {order[(offset + i) % n] for i in range(budget)}
    if strategy == "static_first_half":
        return set(range(budget))
    if strategy == "static_early":
        return set(range(budget - 1)) | {n - 1}
    if strategy == "rotational":
        order = list(range(0, n, 2)) + list(range(1, n, 2))
        return {order[(probe_index * budget + i) % n] for i in range(budget)}
    order = list(range(n))
    random.Random(seed + probe_index).shuffle(order)
    def value(i):
        v = scores[i]
        return float(v) if v is not None and math.isfinite(v) else 0.0
    order.sort(key=value)
    if strategy in ("walking_coverage", "hybrid_coverage"):
        # Keep half of the budget for gradient priorities and walk the other
        # slots through the remaining blocks in ascending order.
        priority_count = budget // 2
        priority = set(order[-priority_count:]) if priority_count else set()
        remaining = sorted(set(range(n)) - priority)
        coverage_count = budget - priority_count
        if strategy == "hybrid_coverage":
            counts = [0] * n if update_counts is None else update_counts
            if len(counts) != n:
                raise ValueError("Invalid block update counts.")
            random.Random(seed + probe_index).shuffle(remaining)
            remaining.sort(key=lambda i: counts[i])
            return priority | set(remaining[:coverage_count])
        return priority | {
            remaining[(i * len(remaining) // coverage_count + walk_phase) % len(remaining)]
            for i in range(coverage_count)
        }
    return set(order[-budget:] if strategy in ("downhill", "high_gradient") else order[:budget])



# Separate from fixed-budget STRATEGIES: these policies determine their own count.
ADAPTIVE_STRATEGIES = ("adaptive_descent", "adaptive_inverse_descent",
                       "adaptive_gradient", "adaptive_inverse_gradient", "adaptive_random")
GRADIENT_THRESHOLD_RATIO = 0.5


def adaptive_blocks(strategy, scores, probe_index, *, matched_count=None, seed=42):
    n = len(scores)
    if strategy not in ADAPTIVE_STRATEGIES or not n or probe_index < 0:
        raise ValueError("Invalid adaptive selective strategy.")
    if strategy == "adaptive_random":
        if type(matched_count) is not int or not 1 <= matched_count <= n:
            raise ValueError("Missing or invalid matched adaptive block count.")
        return set(random.Random(seed + probe_index).sample(range(n), matched_count))
    known = {i: float(v) for i, v in enumerate(scores) if v is not None and math.isfinite(v)}
    unknown = set(range(n)) - set(known)
    if strategy in ("adaptive_descent", "adaptive_inverse_descent"):
        # Original policy includes unknowns and falls back to the best score.
        inverse = strategy == "adaptive_inverse_descent"
        active = unknown | {i for i, v in known.items() if (v <= 0 if inverse else v > 0)}
        if active:
            return active
        return {min(known, key=lambda i: (known[i], i)) if inverse else
                max(known, key=lambda i: (known[i], i))}
    # A relative threshold, not a quantile: all 28 can qualify.
    if not known:
        return set(range(n))  # no usable probe evidence: retry on the next probe
    threshold = GRADIENT_THRESHOLD_RATIO * sum(known.values()) / len(known)
    high = {i for i, v in known.items() if v > 0 and v >= threshold}
    inverse = strategy == "adaptive_inverse_gradient"
    active = (set(known) - high) if inverse else high
    # Keep at least one block active to avoid empty optimizer/adapter windows.
    if not active:
        active = {min(known, key=lambda i: (known[i], i)) if inverse else
                  max(known, key=lambda i: (known[i], i))}
    return active


def load_matched_counts(report_path, n, warmup, interval):
    """Read the completed source case; never silently substitute a fixed budget."""
    import json
    from pathlib import Path
    try:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        run = next(r for r in report["runs"] if r["case"] == "adaptive_gradient-lr450")
        if run["status"] != "finished":
            raise ValueError("source run has not finished")
        last_step = run["epochs"][-1]["step"]
        if run["epochs"][-1]["epoch"] != 5 or type(last_step) is not int or last_step < warmup:
            raise ValueError("source run lacks its final epoch")
        expected = list(range(warmup, last_step + 1, interval))
        counts = {}
        for p in run["probes"]:
            step, count = p["step"], p["active"]
            if (type(step) is not int or step in counts or type(count) is not int or not 1 <= count <= n
                    or p["total"] != n or p.get("strategy") != "adaptive_gradient"
                    or len(set(p["active_blocks"])) != count
                    or any(type(i) is not int or not 0 <= i < n for i in p["active_blocks"])):
                raise ValueError("invalid or duplicate source probe")
            counts[step] = count
        if sorted(counts) != expected:
            raise ValueError("incomplete source probe schedule")
        return [counts[step] for step in expected]
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration) as exc:
        raise RuntimeError("Adaptive Random requires a completed gradient-threshold run in this comparison: "
                           + str(exc)) from exc
