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
               "lora_tsd": 4, "cmuon": 4, "adafactor": 1}
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


# What a step holds for a moment on top of that (bytes per trained weight) and over which weights at once: "group" for
# the optimizers that stack each shape group of same-sized weights and step it as one (PoLoRA and LoRA-TSD in fp32,
# Muon in bf16) - the largest group is the peak; "tensor" for the per-weight ones (fused AdamW, the 8-bit ones,
# Prodigy, Automagic, SOAP) - the largest single weight is.
STEP_BYTES = {"polora": (24, "group"), "lora_tsd": (16, "group"), "muon": (6, "group"), "cmuon": (6, "group"),
              "adamw": (8, "tensor"), "adamw8bit": (8, "tensor"), "pagedadamw8bit": (8, "tensor"),
              "ademamix8bit": (8, "tensor"), "pagedademamix8bit": (8, "tensor"), "lion8bit": (4, "tensor"),
              "prodigy": (12, "tensor"), "automagic3": (8, "tensor"), "automagic4": (8, "tensor"),
              "soap": (16, "tensor"), "adafactor": (8, "tensor")}
UNKNOWN_STEP_BYTES = (8, "tensor")
# An optimizer with a Compile option (PoLoRA): its compiled kernels keep two more fp32 copies of the largest group
# alive between calls, plus the compiler's own buffers on the card. Chris's 4070 Ti, Nov 2026: the same PoLoRA run
# took ~22 s a step uncompiled and ~66 s compiled at swap 17 (both spilled), so compile=True holds measurably more.
COMPILE_BYTES = {"polora": 8}
COMPILE_FIXED_GB = 0.25
EMA_BYTES = 4                                # the EMA's fp32 shadow, per trained weight
ADAPTER_FLOOR_GB = -0.25                     # a LoRA smaller than the figures' plans at most this much less
# The families' figures are peaks measured on big cards (a 5090 for Krea 2). On a card under 20 GB the allocator's
# reserve on top of the allocated peak and Windows' own share are a bigger part of the card: Chris's 4070 Ti (Krea 2,
# 0.25 MP, int8, Nov 2026) spilled at swap 17, planned ~0.45 GB under the figure's budget once the EMA, PoLoRA and the
# context LoRA were counted, and ran clean at swap 20. Applied only with the big-card table (not train_memory_small).
SMALL_CARD_GB = 20.0
SMALL_CARD_RESERVE_GB = 0.8


def optimizer_compiles(optimizer, optimizer_args="") -> bool:
    """Whether this optimizer runs compiled kernels: PoLoRA's Compile (on unless compile=False / 0 / off)."""
    if str(optimizer or "").strip().lower() not in COMPILE_BYTES:
        return False
    import re
    m = re.search(r"(?:^|\s)compile\s*=\s*(\S+)", str(optimizer_args or ""))
    return not (m and m.group(1).strip().lower() in ("false", "0", "off", "no"))


def step_gb(shapes, rank, optimizer) -> float:
    """GB one optimizer step holds for a moment beyond the resident state (STEP_BYTES)."""
    per, scope = STEP_BYTES.get(str(optimizer or "").strip().lower(), UNKNOWN_STEP_BYTES)
    return _peak_weights(shapes, rank, scope) * per / 1024 ** 3


def _peak_weights(shapes, rank, scope) -> int:
    r = max(1, int(rank))
    if not shapes:
        return 0
    if scope == "group":
        groups = {}
        for o, i in shapes:
            groups[(o, i)] = groups.get((o, i), 0) + r * (o + i)
        return max(groups.values())
    return max(r * max(o, i) for o, i in shapes)      # the larger of a pair's two factors


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


def extra_gb(desc, dit_path, network_type, rank, optimizer, *, optimizer_args="", ema=False, frozen_files=(),
             card_gb=None) -> tuple:
    """(GB, why) this run holds beyond what the family's figures were measured with (their recipe:
    TRAIN_MEMORY_RECIPE, rank 16 on AdamW8bit unless the family says otherwise), every term from this run's own
    choices:
      * the LoRA and its optimizer: weights + gradients + state against the recipe's - a proper difference, so a
        smaller LoRA or a lighter optimizer plans less (never below ADAPTER_FLOOR_GB) - plus the difference in what
        one optimizer step holds for a moment (STEP_BYTES), and compiled kernels' extra (COMPILE_BYTES);
      * the EMA's fp32 shadow; frozen LoRA files (a context LoRA, a training adapter) in bf16;
      * on a card under SMALL_CARD_GB planned from big-card figures, SMALL_CARD_RESERVE_GB.
    The adapter terms need a plain LoRA and a readable model file (LoKr's factors are far smaller: 0 for them)."""
    terms = []
    plain = str(network_type or "lora").lower() == "lora"
    shapes = linear_shapes(dit_path) if (dit_path and plain) else []
    opt = str(optimizer or "").strip().lower()
    if shapes:
        ref_rank, ref_opt = getattr(desc, "train_memory_recipe", None) or TRAIN_MEMORY_RECIPE
        d = (adapter_gb(shapes, rank, opt) + step_gb(shapes, rank, opt)
             - adapter_gb(shapes, ref_rank, ref_opt) - step_gb(shapes, ref_rank, ref_opt))
        terms.append((f"rank {int(rank)} {opt} vs the figures' rank {ref_rank} {ref_opt}", max(ADAPTER_FLOOR_GB, d)))
        if optimizer_compiles(opt, optimizer_args):
            c = _peak_weights(shapes, rank, "group") * COMPILE_BYTES[opt] / 1024 ** 3 + COMPILE_FIXED_GB
            terms.append((f"{opt} compile", c))
        if ema:
            terms.append(("EMA", sum(max(1, int(rank)) * (o + i) for o, i in shapes) * EMA_BYTES / 1024 ** 3))
    for f in frozen_files or ():
        if f:
            w = file_weights_gb(f)
            if w:
                terms.append((os.path.basename(str(f)), w))
    small = bool(getattr(desc, "train_memory_small", None)) and _uses_small_table(desc)
    if card_gb and 0 < float(card_gb) < SMALL_CARD_GB and not small:
        terms.append((f"{float(card_gb):.0f} GB card reserve", SMALL_CARD_RESERVE_GB))
    gb = sum(v for _n, v in terms)
    why = (" + ".join(f"{n} {v:+.2f}" for n, v in terms) + f" = {gb:+.2f} GB on the figures") if terms else ""
    return gb, why


def _uses_small_table(desc) -> bool:
    try:
        from fizgig.families import quant
        return quant.small_card_figures(desc)
    except Exception:
        return False


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


__all__ = ["DEFAULT_MARGIN_GB", "STATE_BYTES", "file_weights_gb", "optimizer_compiles", "step_gb", "TRAIN_MEMORY_RECIPE", "adapter_gb", "correction", "extra_gb", "key",
           "linear_shapes", "margin", "record", "state_bytes"]
