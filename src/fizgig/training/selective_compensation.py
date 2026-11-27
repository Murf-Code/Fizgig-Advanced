"""Bounded nominal active-count scaling validated in Krea 2 research 7G.1."""
import math

CAP = 1.5


def active_count_multiplier(active: int, total: int) -> float:
    if not 1 <= active <= total:
        raise ValueError("Selective compensation requires at least one active block")
    return min(CAP, math.sqrt(total / active))
