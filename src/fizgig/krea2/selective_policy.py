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
