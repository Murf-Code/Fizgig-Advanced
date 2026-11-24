"""Budget-matched experimental block routing. No training RNG is consumed."""
import math
import random

STRATEGIES = ("downhill", "uphill", "high_gradient", "low_gradient", "random", "rotational")


def select_blocks(strategy, scores, budget, probe_index, seed=42):
    n = len(scores)
    if strategy not in STRATEGIES or not 1 <= budget <= n or probe_index < 0:
        raise ValueError("Invalid selective benchmark strategy, budget or probe index.")
    if strategy == "random":
        return set(random.Random(seed + probe_index).sample(range(n), budget))
    if strategy == "rotational":
        order = list(range(0, n, 2)) + list(range(1, n, 2))
        return {order[(probe_index * budget + i) % n] for i in range(budget)}
    order = list(range(n))
    random.Random(seed + probe_index).shuffle(order)
    def value(i):
        v = scores[i]
        return float(v) if v is not None and math.isfinite(v) else 0.0
    order.sort(key=value)
    return set(order[-budget:] if strategy in ("downhill", "high_gradient") else order[:budget])
