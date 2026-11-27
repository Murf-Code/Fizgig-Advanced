"""Nominal active-count learning-term compensation, isolated from scheduler state."""
import math

CAP = 1.5
SUPPORTED = ("muon", "adamw", "adamw8bit", "pagedadamw8bit")


def active_count_multiplier(active: int, total: int) -> float:
    if not 1 <= active <= total:
        raise ValueError("Selective compensation requires at least one active block")
    return min(CAP, math.sqrt(total / active))


def validate_optimizer(optimizer, name):
    """Fail closed: don't mistake a fallback or external optimizer for a supported one."""
    import torch
    from fizgig.training.muon import Muon, BatchedMuon

    name = str(name).strip().lower()
    if name == "muon":
        if type(optimizer) not in (Muon, BatchedMuon) or any(
                g.get("adjust_lr_fn") != "match_rms_adamw" for g in optimizer.param_groups):
            raise RuntimeError("Selective compensation requires built-in Muon with match_rms_adamw")
    elif name == "adamw":
        if type(optimizer) is not torch.optim.AdamW or any(
                g.get("differentiable", False) or not g.get("decoupled_weight_decay", True)
                for g in optimizer.param_groups):
            raise RuntimeError("Selective compensation requires native, non-differentiable AdamW")
    elif name in ("adamw8bit", "pagedadamw8bit"):
        import bitsandbytes as bnb
        expected = bnb.optim.AdamW8bit if name == "adamw8bit" else bnb.optim.PagedAdamW8bit
        if (type(optimizer) is not expected or getattr(optimizer, "optimizer_name", None) != "adam"
                or not callable(getattr(optimizer, "get_config", None))):
            raise RuntimeError("Selective compensation requires the requested native bitsandbytes AdamW")
    else:
        raise RuntimeError("Unsupported optimizer for Selective compensation: " + name)
    if not optimizer.param_groups:
        raise RuntimeError("Selective compensation requires optimizer parameter groups")


def _adamw_group_values(group):
    lr, wd = group["lr"], group["weight_decay"]
    if type(lr) not in (int, float) or type(wd) not in (int, float):
        raise RuntimeError("Selective compensation requires scalar LR and weight decay")
    if not (math.isfinite(lr) and math.isfinite(wd) and lr >= 0 and wd >= 0):
        raise RuntimeError("Invalid AdamW LR or weight decay")
    return lr, wd


def _check_bnb_config(optimizer):
    # The all-active warmup must initialize BNB's internal override manager first.
    if not getattr(optimizer, "initialized", False):
        raise RuntimeError("Selective compensation requires an initialized 8-bit optimizer")
    for gi, group in enumerate(optimizer.param_groups):
        for pi, p in enumerate(group["params"]):
            if p.grad is None:
                continue
            cfg = optimizer.get_config(gi, pi, group)
            if (cfg.get("lr") != group["lr"] or cfg.get("weight_decay") != group["weight_decay"]
                    or cfg.get("max_unorm", 0) != 0 or cfg.get("skip_zeros", False)):
                raise RuntimeError("Selective compensation does not support bitsandbytes per-parameter "
                                   "LR/decay overrides, update-norm limiting or skip_zeros")


def compensated_step(optimizer, name, multiplier):
    """One native optimizer step. Always restore group LR/decay before scheduler/EMA/save.

    AdamW's decoupled shrinkage is 1-lr*wd. Scaling lr and inversely scaling wd
    preserves that product in real arithmetic, while scaling the adaptive update.
    Floating-point kernels can differ by rounding. Neither moments nor gradients
    are manually modified. A failed step is NOT a rollback of optimizer weights.
    """
    scale = float(multiplier)
    if not 1.0 <= scale <= CAP:
        raise ValueError("Selective learning multiplier must be in [1, 1.5]")
    if name not in SUPPORTED:
        raise RuntimeError("Unsupported optimizer for Selective compensation: " + str(name))
    if name == "muon":
        return optimizer.step(learning_multiplier=scale)
    if scale == 1.:
        return optimizer.step()
    saved = [(g, *_adamw_group_values(g)) for g in optimizer.param_groups]
    is_bnb = name in ("adamw8bit", "pagedadamw8bit")
    if is_bnb:
        _check_bnb_config(optimizer)
    try:
        for g, lr, wd in saved:
            g["lr"] = lr * scale
            g["weight_decay"] = wd / scale
        if is_bnb:
            # Catch absolute overrides that happened to agree before scaling.
            _check_bnb_config(optimizer)
        return optimizer.step()
    finally:
        for g, lr, wd in saved:
            g["lr"], g["weight_decay"] = lr, wd
