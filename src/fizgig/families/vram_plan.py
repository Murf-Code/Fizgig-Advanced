"""VRAM planning that knows the LoRA's rank and the optimizer, and learns from the peaks past runs measured.

The Auto plan (quant.plan) reads each family's measured training peaks (description.train_memory), taken with a
modest LoRA (TRAIN_MEMORY_RECIPE: rank 16 on AdamW8bit unless a family says otherwise). A run past that recipe holds
more: every LoRA weight keeps its value and gradient, and the optimizer its own state per weight - AdamW two fp32
moments, Muon one, SOAP two moments plus two square matrices per side of every weight. adapter_gb() sizes that from
the model file's own Linear shapes (its header, nothing loaded) and extra_gb() adds what this run holds beyond the
recipe (never less - a smaller LoRA's saving is too small to plan swap on).

Past runs correct the figures: at each epoch end the trainer records the peak it measured against the peak the plan
predicted for the precision, swap and size it ran at (record()); correction() gives the largest recent difference for
the same model, precision, network and optimizer (else the same model and precision), at most CORRECTION_DOWN_GB less.
Kept in user_data/vram_history.json. The safety margin (Preferences > Training memory, default 1.5 GB) is what the
plan leaves free beyond all of it.
"""
from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

DEFAULT_MARGIN_GB = 1.5
TRAIN_MEMORY_RECIPE = (16, "adamw8bit")      # (rank, optimizer) the train_memory figures were measured with
CORRECTION_DOWN_GB = 1.0                     # a learned correction lowers the figures by at most this
KEEP_RUNS = 5                                # recent runs remembered per key
# optimizer state per trained weight, bytes, beyond its own value and gradient (fp32 LoRA weights: 4 + 4)
STATE_BYTES = {"adamw": 8, "adamw8bit": 2, "pagedadamw8bit": 2, "ademamix8bit": 3, "pagedademamix8bit": 3,
               "lion8bit": 1, "prodigy": 16, "muon": 4, "polora": 8, "automagic3": 6, "automagic4": 8, "soap": 8,
               "lora_tsd": 4, "cmuon": 4}
UNKNOWN_STATE_BYTES = 8                      # a module.path.ClassName optimizer: planned like AdamW
SOAP_MAX_PRECOND = 10000                     # SOAP's max_precond_dim: wider sides get no matrix


def _history_path():
    from fizgig.user_data import USER_DATA
    return os.path.join(USER_DATA, "vram_history.json")


def linear_shapes(path) -> list:
    """(out, in) of every 2D weight in a .safetensors model file, read from its header; [] when it can't be read."""
    try:
        from fizgig.utils.st_meta import _header
        with open(path, "rb") as f:
            _n, head = _header(f)
    except Exception:
        return []
    out = []
    for k, info in head.items():
        shape = (info or {}).get("shape") if isinstance(info, dict) else None
        if k.endswith(".weight") and shape and len(shape) == 2 and min(shape) >= 16:
            out.append((int(shape[0]), int(shape[1])))
    return out


def state_bytes(optimizer) -> int:
    return STATE_BYTES.get(str(optimizer or "").strip().lower(), UNKNOWN_STATE_BYTES)


def adapter_gb(shapes, rank, optimizer) -> float:
    """GB a rank-`rank` LoRA on every Linear of `shapes` holds with `optimizer`: weights, gradients, its state."""
    rank, per = max(1, int(rank)), 8 + state_bytes(optimizer)
    total = 0.0
    soap = str(optimizer or "").strip().lower() == "soap"
    for o, i in shapes:
        n = rank * (o + i)
        total += n * per
        if soap:                                   # GG and Q per side (fp32) of the down (r, i) and up (o, r)
            total += 8 * sum(d * d for d in (rank, i, o, rank) if d <= SOAP_MAX_PRECOND)
    return total / 1024 ** 3


def extra_gb(desc, dit_path, network_type, rank, optimizer) -> tuple:
    """(GB, why) this run's LoRA and optimizer hold beyond the recipe the family's figures were measured with. 0 for
    a network that isn't a plain LoRA (LoKr's factors are far smaller) or a model file whose header can't be read."""
    if str(network_type or "lora").lower() != "lora":
        return 0.0, ""
    shapes = linear_shapes(dit_path) if dit_path else []
    if not shapes:
        return 0.0, ""
    ref_rank, ref_opt = getattr(desc, "train_memory_recipe", None) or TRAIN_MEMORY_RECIPE
    have = adapter_gb(shapes, rank, optimizer)
    ref = adapter_gb(shapes, ref_rank, ref_opt)
    gb = max(0.0, have - ref)
    why = (f"rank {int(rank)} with {optimizer}: ~{have:.2f} GB of LoRA and optimizer state, {gb:.2f} GB over the "
           f"figures' rank {ref_rank} {ref_opt}") if gb >= 0.01 else ""
    return gb, why


# what a run holds beside its own LoRA and optimizer state, which the family's figures never had (fp32 / bf16 bytes)
EMA_BYTES = 4                                # the EMA's fp32 shadow, per trained weight
PAIR_STEP_COPIES = {"polora": 6, "lora_tsd": 4}   # fp32 copies a pair optimizer's step stacks per shape group


def file_weights_gb(path, bytes_per=2) -> float:
    """GB a frozen LoRA file takes on the GPU, loaded in bf16 (every tensor in its header); 0 when it can't be read."""
    try:
        from fizgig.utils.st_meta import _header
        with open(path, "rb") as f:
            _n, head = _header(f)
    except Exception:
        return 0.0
    n = 0
    for k, info in head.items():
        shape = (info or {}).get("shape") if isinstance(info, dict) and k != "__metadata__" else None
        if shape:
            m = 1
            for d in shape:
                m *= int(d)
            n += m
    return n * bytes_per / 1024 ** 3


def run_extras_gb(dit_path, network_type, rank, optimizer, ema=False, frozen_files=()) -> tuple:
    """(GB, why) a run holds on the GPU that the family's figures don't count: the EMA's fp32 shadow of the LoRA, a pair
    optimizer's per-step stacks (PoLoRA stacks each shape group's gradients, weights and updates in fp32 - the largest
    group is the peak), and frozen LoRA files (a context LoRA, a training adapter) in bf16 on every Linear they patch.
    On a 12 GB card these together are the half to one GB that tipped a swapped INT8 Krea 2 run into Windows' shared
    GPU memory (Nov 2026: 66 s a step, where the same run with more blocks swapped takes about 5)."""
    gb, parts = 0.0, []
    plain = str(network_type or "lora").lower() == "lora"
    shapes = linear_shapes(dit_path) if (dit_path and plain and (ema or str(optimizer or "").lower() in PAIR_STEP_COPIES)) else []
    r = max(1, int(rank or 1))
    if ema and shapes:
        e = sum(r * (o + i) for o, i in shapes) * EMA_BYTES / 1024 ** 3
        gb += e
        parts.append(f"EMA {e:.2f}")
    copies = PAIR_STEP_COPIES.get(str(optimizer or "").strip().lower(), 0)
    if copies and shapes:
        groups = {}
        for o, i in shapes:
            groups[(o, i)] = groups.get((o, i), 0) + r * (o + i)
        s = max(groups.values()) * 4 * copies / 1024 ** 3
        gb += s
        parts.append(f"{optimizer} step {s:.2f}")
    for f in frozen_files or ():
        if f:
            w = file_weights_gb(f)
            if w:
                gb += w
                parts.append(f"{os.path.basename(str(f))} {w:.2f}")
    why = (f"~{gb:.2f} GB beside the LoRA: " + ", ".join(parts)) if parts else ""
    return gb, why


# ---- learned corrections --------------------------------------------------------------------------------------
def key(desc_key, precision, network_type, optimizer) -> str:
    return f"{desc_key}|{precision}|{str(network_type or 'lora').lower()}|{str(optimizer or '').lower()}"


def _read(path=None):
    try:
        with open(path or _history_path(), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def record(k, run, predicted_gb, measured_gb, *, path=None, info=None) -> None:
    """This run's measured peak against the plan's prediction (one entry per run, the latest kept)."""
    path = path or _history_path()
    d = _read(path)
    runs = d.setdefault(k, {})
    runs[str(run)] = {"predicted": round(float(predicted_gb), 3), "measured": round(float(measured_gb), 3),
                      "t": int(time.time()), **(info or {})}
    for old in sorted(runs, key=lambda r: runs[r].get("t", 0))[:-KEEP_RUNS]:
        runs.pop(old, None)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
        os.replace(path + ".tmp", path)
    except OSError:
        logger.warning("[vram] couldn't save this run's measured peak", exc_info=True)


def correction(k, *, path=None) -> tuple:
    """(GB, why) to add to the figures for runs like `k`: the largest of its recent runs' (measured - predicted),
    else of any run of the same model and precision; never below -CORRECTION_DOWN_GB. (0, "") with none."""
    d = _read(path)
    runs = list((d.get(k) or {}).values())
    where = "past runs like this one"
    if not runs:
        pre = "|".join(k.split("|")[:2]) + "|"
        runs = [r for kk, v in d.items() if kk.startswith(pre) for r in v.values()]
        where = "past runs of this model at this precision"
    deltas = [float(r["measured"]) - float(r["predicted"]) for r in runs if "measured" in r and "predicted" in r]
    if not deltas:
        return 0.0, ""
    c = max(-CORRECTION_DOWN_GB, max(deltas))
    return c, f"{'+' if c >= 0 else ''}{c:.2f} GB learned from {len(deltas)} of {where}"


def margin(value) -> float:
    """The Preferences margin as GB (0-8), DEFAULT_MARGIN_GB when blank or unreadable."""
    try:
        v = float(str(value).strip())
    except (TypeError, ValueError):
        return DEFAULT_MARGIN_GB
    return min(8.0, max(0.0, v))


__all__ = ["DEFAULT_MARGIN_GB", "STATE_BYTES", "file_weights_gb", "run_extras_gb", "TRAIN_MEMORY_RECIPE", "adapter_gb", "correction", "extra_gb", "key",
           "linear_shapes", "margin", "record", "state_bytes"]
