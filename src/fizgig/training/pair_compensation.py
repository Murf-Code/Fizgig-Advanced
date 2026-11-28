"""Research-only native LR scaling for paired LoRA optimizers."""
import math

CAPS = {"polora": 1.5, "lora_tsd": 1.25}


def pair_multiplier(name, active, total, *, enabled):
    if name not in CAPS or not 1 <= active <= total or total != 28:
        raise ValueError("Unsupported pair optimizer or Selective block count")
    return min(CAPS[name], math.sqrt(total / active)) if enabled else 1.


def pair_step(optimizer, name, multiplier):
    """One native step; do not modify gradients, momentum, magnitude or caps.

    The comparison harness verifies the exact optimizer class and pair coverage.
    Restore native group LRs even if the step raises; weights are not rolled back.
    """
    scale = float(multiplier)
    if name not in CAPS or not 1. <= scale <= CAPS[name]:
        raise ValueError("Invalid pair compensation multiplier")
    saved = [(g, g["lr"]) for g in optimizer.param_groups]
    if not saved or any(type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0 for _, lr in saved):
        raise RuntimeError("Pair compensation requires positive finite scalar native LRs")
    caps = [g.get("max_delta_norm") for g,_ in saved]
    magnitude = optimizer.magnitude_scale
    applied = [lr * scale for _,lr in saved]
    try:
        for (g,_),lr in zip(saved,applied):
            g["lr"] = lr
        optimizer.step()
    finally:
        for g,lr in saved:
            g["lr"] = lr
    restored = all(g["lr"] == lr for g,lr in saved)
    if not restored or optimizer.magnitude_scale != magnitude or [g.get("max_delta_norm") for g,_ in saved] != caps:
        raise RuntimeError("Native group LR, magnitude or cap changed")
    return {"pair_compensation_multiplier": scale, "pair_compensation_lr_restored": restored,
            "pair_compensation_applied_lrs": applied}
