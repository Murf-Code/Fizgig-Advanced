"""VRAM planning that knows the LoRA's rank and the optimizer, and learns from the peaks past runs measured.

The Auto plan (quant.plan) reads each family's measured training peaks (description.train_memory), taken with a
modest LoRA (TRAIN_MEMORY_RECIPE: rank 16 on AdamW8bit unless a family says otherwise). A run past that recipe holds
more: every LoRA weight keeps its value and gradient, and the optimizer its own state per weight - AdamW two fp32
moments, Muon one, SOAP two moments plus two square matrices per side of every weight. adapter_gb() sizes that from
the model file's own Linear shapes (its header, nothing loaded) and extra_gb() adds what this run holds beyond the
recipe (never less - a smaller LoRA's saving is too small to plan swap on).

Past runs correct the figures: at each epoch end, and again after its previews, the trainer records what it
measured - training's own peak and the previews' peak - with the precision, swap, size, rank and optimizer it ran at
(record()). correction() turns those into an offset on the figures: for each recent run, what it measured less what
today's figures predict for that run's own precision, swap and size, so it carries over when the plan changes (other
swap, other rank) and stays right when the figures themselves are retuned. The largest such offset wins (the
cautious one), plus LEARN_HEADROOM_GB when it lowers the plan. Runs of the same model, precision, network and
optimizer count first, then the same model and precision, then the same model at another precision (half of a
lowering only). A run that renders previews only lowers its plan from runs whose previews were measured (the
figures are the preview's peak on most families). The small card's reserve and the safety margin (Preferences >
Training memory, default 1.5 GB) stay on top of all of it. Kept in user_data/vram_history.json.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

logger = logging.getLogger(__name__)

DEFAULT_MARGIN_GB = 1.5
TRAIN_MEMORY_RECIPE = (16, "adamw8bit")      # (rank, optimizer) the train_memory figures were measured with
CORRECTION_DOWN_GB = 6.0                     # a learned correction lowers the figures by at most this
LEARN_HEADROOM_GB = 0.3                      # a lowering stops this far above the highest measured peak
OTHER_PRECISION_SHARE = 0.5                  # runs at another precision lower the plan by half their offset
LEARN_VERSION = 2                            # history entries before this (no base / preview figures) are not used
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
            n += _numel(shape)
    return n * bytes_per / 1024 ** 3


# Frozen files (a context LoRA, a training adapter) are not merged into the base weights. FamilyLoRA.add_file
# keeps them as a separate forward path (W x + s * B(A(x))) in bf16, and LoRAFactor is skipped by block-swap
# offloaders (its class name does not end in "Linear"), so the whole adapter stays resident on the GPU. Only the
# low-rank intermediate of the busiest block is an extra at the run's resolution (frozen adapters do not keep
# intermediates for backward; with checkpointing only one block is live).
FROZEN_BYTES = 2
_DOWN_SUF = (".lora_down.weight", ".lora_A.weight", ".lora.down.weight")
_UP_SUF = (".lora_up.weight", ".lora_B.weight", ".lora.up.weight")
_OTHER_WT = (".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a", ".lokr_w2_b", ".lokr_w1", ".lokr_w2",
             ".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b")
_BLOCK_RE = re.compile(
    r"(?:^|[._])(?:double_blocks?|single_blocks?|transformer_blocks?|blocks?|layers?)[._](\d+)(?:[._]|$)",
    re.I,
)


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n if shape else 0


def _stem_kind(key):
    for suf in _DOWN_SUF:
        if key.endswith(suf):
            return key[:-len(suf)], "down"
    for suf in _UP_SUF:
        if key.endswith(suf):
            return key[:-len(suf)], "up"
    for suf in _OTHER_WT:
        if key.endswith(suf):
            return key[:-len(suf)], "other"
    return None, None


def inspect_frozen_lora(path) -> dict:
    """A frozen LoRA file's GPU footprint, read from its header only (nothing loaded).

    Fizgig loads frozen adapters in bf16 (families/lora.py LoRALinear.add), even when the file is fp32, and
    only the down/up (or A/B, LoKR, LoHa) weights stay on the card. Returns rank, layer count, weight GB,
    and whether the header could be read (otherwise a file-size fallback)."""
    name = os.path.basename(str(path or ""))
    out = {"path": path, "name": name, "rank": None, "layers": 0, "block_layers": 0,
           "weights_gb": 0.0, "ok": False, "fallback": False, "kind": "lora"}
    if not path:
        return out
    if not os.path.isfile(path):
        logger.warning("[vram] %s is missing; not counted in the plan", name or path)
        return out
    try:
        from fizgig.utils.st_meta import _header
        with open(path, "rb") as f:
            head = _header(f)[1]
    except Exception:
        gb = os.path.getsize(path) / 1024 ** 3
        logger.warning("[vram] couldn't read %s; using its file size (%.2f GB)", name, gb)
        out.update(weights_gb=gb, fallback=True)
        return out

    tensors = []
    for k, info in head.items():
        if k == "__metadata__" or not isinstance(info, dict):
            continue
        shape = info.get("shape")
        if shape:
            tensors.append((k, tuple(int(d) for d in shape)))

    elems, ranks, stems, other = 0, [], [], False
    counted = set()
    for k, shape in tensors:
        stem, kind = _stem_kind(k)
        if stem is None:
            continue
        elems += _numel(shape)
        counted.add(k)
        if kind == "down":
            ranks.append(int(shape[0]))          # (rank, in, [kH, kW])
            stems.append(stem)
        elif kind == "other":
            other = True
            stems.append(stem)

    if not counted:                              # unusual key names: skip alpha / bias (they don't stay on the GPU)
        twod = []
        for k, shape in tensors:
            kl = k.lower()
            if k.endswith(".alpha") or k.endswith(".diff_b") or kl.endswith(".bias"):
                continue
            elems += _numel(shape)
            if len(shape) >= 2:
                ranks.append(int(min(shape[:2])))
                twod.append(k)
        stems = twod[:len(twod) // 2] or twod    # each module is a down/up pair

    layers = len(set(stems))
    rank = None
    if ranks:
        lo, hi = min(ranks), max(ranks)
        rank = lo if lo == hi else (lo, hi)
    groups = {}
    for stem in set(stems):
        m = _BLOCK_RE.search(stem)
        if m:                                        # only numbered blocks; outside layers aren't all live at once
            groups[m.group(1)] = groups.get(m.group(1), 0) + 1
    out.update(ok=True, rank=rank, layers=layers,
               block_layers=max(groups.values()) if groups else (1 if layers else 0),
               weights_gb=elems * FROZEN_BYTES / 1024 ** 3,
               kind="lokr" if other and not ranks else "lora")
    return out


def frozen_act_gb(rank, block_layers, *, megapixels=0.25, pictures=1, bucket_step=16) -> float:
    """GB of the low-rank intermediate at the run's size, for the busiest block (the peak: frozen adapters do
    not keep intermediates for backward, and with checkpointing only one block is live)."""
    r = max(rank) if isinstance(rank, tuple) else int(rank or 0)
    if r <= 0 or int(block_layers or 0) <= 0:
        return 0.0
    step = max(1, int(bucket_step or 16))
    tokens = max(1, int(float(megapixels if megapixels is not None else 0.25) * 1e6 / (step * step)))
    return tokens * max(1, int(pictures or 1)) * r * FROZEN_BYTES * int(block_layers) / 1024 ** 3


def frozen_file_term(path, *, megapixels=None, pictures=1, bucket_step=16) -> tuple:
    """(plan-line label, GB) for one frozen LoRA, or ("", 0) when there is nothing to count."""
    info = inspect_frozen_lora(path)
    gb = info["weights_gb"]
    if info["ok"] and not info["fallback"]:
        gb += frozen_act_gb(info["rank"], info["block_layers"], megapixels=megapixels,
                            pictures=pictures, bucket_step=bucket_step)
    if gb <= 0:
        return "", 0.0
    name = info["name"] or os.path.basename(str(path))
    if info["fallback"]:
        return f"{name} (from file size)", gb
    bits = []
    if info["kind"] != "lora" and info["rank"] is None:
        bits.append("LoKR")
    elif isinstance(info["rank"], tuple):
        bits.append(f"rank {info['rank'][0]}-{info['rank'][1]}")
    elif info["rank"] is not None:
        bits.append(f"rank {info['rank']}")
    if info["layers"]:
        bits.append(f"{info['layers']} layers" if info["layers"] != 1 else "1 layer")
    return (f"{name} ({', '.join(bits)})" if bits else name), gb


def extra_gb(desc, dit_path, network_type, rank, optimizer, *, optimizer_args="", ema=False, frozen_files=(),
             card_gb=None, megapixels=None, pictures=1) -> tuple:
    """(GB, why) this run holds beyond what the family's figures were measured with (their recipe:
    TRAIN_MEMORY_RECIPE, rank 16 on AdamW8bit unless the family says otherwise), every term from this run's own
    choices:
      * the LoRA and its optimizer: weights + gradients + state against the recipe's - a proper difference, so a
        smaller LoRA or a lighter optimizer plans less (never below ADAPTER_FLOOR_GB) - plus the difference in what
        one optimizer step holds for a moment (STEP_BYTES), and compiled kernels' extra (COMPILE_BYTES);
      * the EMA's fp32 shadow; frozen LoRA files (a context LoRA, a training adapter) in bf16 at the file's real
        rank and layer count, plus a little for the low-rank intermediate at this run's size;
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
    step = getattr(desc, "bucket_step", None) or 16
    for f in frozen_files or ():
        if f:
            label, w = frozen_file_term(f, megapixels=megapixels, pictures=pictures, bucket_step=step)
            if w:
                terms.append((label, w))
    r = reserve_gb(desc, card_gb)
    if r:
        terms.append((f"{float(card_gb):.0f} GB card reserve", r))
    gb = sum(v for _n, v in terms)
    why = (" + ".join(f"{n} {v:+.2f}" for n, v in terms) + f" = {gb:+.2f} GB on the figures") if terms else ""
    return gb, why


def reserve_gb(desc, card_gb) -> float:
    """SMALL_CARD_RESERVE_GB on a card under SMALL_CARD_GB planned from the big-card table, else 0 - the part of
    extra_gb() a learned correction never takes away."""
    small = bool(getattr(desc, "train_memory_small", None)) and _uses_small_table(desc)
    if card_gb and 0 < float(card_gb) < SMALL_CARD_GB and not small:
        return SMALL_CARD_RESERVE_GB
    return 0.0


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
    """This run's measured peak against the plan's prediction (one entry per run, the latest kept). For learning,
    info carries what correction() recomputes the prediction from: "base" (quant.predicted_gb's figure), "extra" (this
    run's own terms past the figures, without the small card's reserve), "swap", "mp", "batch", and "preview" (the
    previews' peak, GB) / "previews" (whether the run renders them)."""
    path = path or _history_path()
    d = _read(path)
    runs = d.setdefault(k, {})
    runs[str(run)] = {"predicted": round(float(predicted_gb), 3), "measured": round(float(measured_gb), 3),
                      "t": int(time.time()), "v": LEARN_VERSION, **(info or {})}
    for old in sorted(runs, key=lambda r: runs[r].get("t", 0))[:-KEEP_RUNS]:
        runs.pop(old, None)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
        os.replace(path + ".tmp", path)
    except OSError:
        logger.warning("[vram] couldn't save this run's measured peak", exc_info=True)


def _offset(r, precision, desc, previews):
    """GB one history entry says the figures are off by for a run like it (measured - predicted), or None when it
    can't say. With desc the prediction is today's figures for that run's own precision, swap and size; with
    previews on, only an entry whose previews were measured, at the larger of the two peaks."""
    if "measured" not in r or int(r.get("v", 1) or 1) < LEARN_VERSION:
        return None
    peak = float(r["measured"])
    unseen = previews and not r.get("preview")   # a run without measured previews can only raise the plan
    if previews and not unseen:
        peak = max(peak, float(r["preview"]))
    pred = None
    if desc is not None and "extra" in r and "swap" in r:
        try:
            from fizgig.families import quant
            base = quant.predicted_gb(desc, precision, int(r.get("swap") or 0), float(r.get("mp") or 1.0),
                                      int(r.get("batch") or 1))
        except Exception:
            base = None
        if base is not None:
            pred = base + float(r["extra"])
    if pred is None and "base" in r and "extra" in r:
        pred = float(r["base"]) + float(r["extra"])
    if pred is None:
        return None
    o = peak - pred
    return None if unseen and o <= 0 else o


def correction(k, *, path=None, desc=None, previews=False) -> tuple:
    """(GB, why) to add to the figures for runs like `k` (key()): the largest recent offset (_offset) among runs like
    this one, else runs of the same model and precision, else of the same model at another precision (a lowering
    counted at OTHER_PRECISION_SHARE). A lowering keeps LEARN_HEADROOM_GB above the highest measured peak and goes at
    most CORRECTION_DOWN_GB. (0, "") with nothing to learn from. Upward offsets (a run held more) count in full."""
    d = _read(path)
    parts = k.split("|")
    fam, prec = parts[0], (parts[1] if len(parts) > 1 else "")
    tiers = (("past runs like this one", lambda kk: kk == k, 1.0),
             ("past runs of this model at this precision", lambda kk: kk.startswith(f"{fam}|{prec}|"), 1.0),
             ("past runs of this model at another precision",
              lambda kk: kk.startswith(f"{fam}|") and not kk.startswith(f"{fam}|{prec}|"), OTHER_PRECISION_SHARE))
    for where, match, share in tiers:
        deltas = []
        for kk, v in d.items():
            if not match(kk) or not isinstance(v, dict):
                continue
            p = kk.split("|")[1] if "|" in kk else prec
            for r in v.values():
                o = _offset(r, p, desc, previews) if isinstance(r, dict) else None
                if o is not None:
                    deltas.append(o)
        if not deltas:
            continue
        c = max(deltas)
        if c < 0:
            c = min(0.0, c + LEARN_HEADROOM_GB) * share
            c = max(-CORRECTION_DOWN_GB, c)
        n = len(deltas)
        return c, (f"{'+' if c >= 0 else ''}{c:.2f} GB learned from {n} of {where}"
                   + (" (their previews counted)" if previews else ""))
    return 0.0, ""


def margin(value) -> float:
    """The Preferences margin as GB (0-8), DEFAULT_MARGIN_GB when blank or unreadable."""
    try:
        v = float(str(value).strip())
    except (TypeError, ValueError):
        return DEFAULT_MARGIN_GB
    return min(8.0, max(0.0, v))


__all__ = ["DEFAULT_MARGIN_GB", "STATE_BYTES", "file_weights_gb", "frozen_file_term", "inspect_frozen_lora", "optimizer_compiles", "reserve_gb", "step_gb", "TRAIN_MEMORY_RECIPE", "adapter_gb", "correction", "extra_gb", "key",
           "linear_shapes", "margin", "record", "state_bytes"]
