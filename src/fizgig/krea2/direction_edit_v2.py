"""Krea 2 Text fusion direction edit v2: a gradient-free, reversible edit of the text-fusion stack.

Krea 2 doesn't refuse a prompt; a tag it suppresses simply doesn't show, as if it were never read. This looks for
where that happens on the text path alone (Qwen3-VL taps -> layerwise blocks -> projector -> refiner blocks ->
txtmlp) and edits only the few weights that write the direction. Modelled on experiment 8 of "Sparks of
Weight-Space Programming" (one MLP-written direction changed on the 64 output columns per layer that wrote it
most).

1. Measure: every base prompt alone and with each tag appended. At each stage (taps, lw_0, lw_1, rf_0, rf_1,
   txtmlp) the masked mean over valid tokens; the layerwise stacks and the taps are mixed with the projector's own
   weights.
2. Score: each tag's contribution relative to the base, c = v(b + t) - v(b), rel = |c| / |v(b)|, as a z-score
   against the control tags on the same prompt (a control against the other controls); suppression = -z at the
   last stage. The stage where a tag's z falls most is logged as where it collapses.
3. Direction: per edited block, d = mean(c, targets) - mean(c, controls) at that block's output, u = d / |d|;
   optional extra PCA directions (num_directions > 1). Reliability: split-half cosine (even vs odd prompts) and the
   per-prompt cosine.
4. Edit: dW[:, S] = +s U U^T W[:, S] on attn.wo / mlp.down, S = the top-C columns by |U^T W[:, j]| x the input
   unit's typical size.
5. Verify: score again with the edit, and with a random direction rescaled to the same size per Linear as a
   control, on captions held out of the fit.
6. Export: dW is linear in s and of rank <= k x rounds, so a kohya LoRA (alpha = rank) reproduces it exactly: LoRA

What v2 changes against v1 (direction_edit.py, left as it is so the two can be compared):
  * The sign. Strength > 0 now AMPLIFIES the target-vs-control direction on the chosen columns; v1's +1 removed
    it, and a test at LoRA -1 gave the better result. A negative strength removes it.
  * "Improves" is signed. The edit has to LOWER the targets' suppression, and by more than the random control
    does; v1 compared absolute changes, so an edit that made things worse could pass.
  * Held-out verification: verify_count extra captions never used for fitting score the edit (before, after,
    random) on captions held out of the fit.
  * Blocks are measured and edited in pipeline order, each on the path as already edited (v1 measured every block
    on the unedited model).
  * The random control is rescaled to the real edit's norm per Linear; control_drift measures how far the controls
    themselves moved.
  * rounds > 1: each round draws new captions and (auto) controls, re-measures on the edited path and adds a
    share (strength / rounds) of a fresh edit; the LoRA rank rises to num_directions x rounds so the sum is exact.
  * Memory: prompts are encoded in chunks into a store that spills to disk past ram_budget_gb, so a large
    caption_count no longer holds every prompt's hidden states in RAM.
  * Stages are settable: score_stage (suppression score, default txtmlp as in v1) and scan_stage (default rf_1).
    txtmlp starts with an RMSNorm, which erases how loud a token is, so a tag that is quiet but not
    different in direction can be invisible there.
  * scan: score every frequent tag in its own real captions (with and without it), pick the tags the text path
    hears least as targets and mid-range tags as controls, then edit. No hand-picked targets or controls.

No backward pass and no image transformer: Qwen3-VL is loaded once for the prompts and freed; only the text path
(the four text-fusion blocks + txtmlp) is read from the DiT file. Only the optional renders load the whole model.
"""
import copy
import functools
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import tempfile

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
STAGES = ("taps", "lw_0", "lw_1", "rf_0", "rf_1", "txtmlp")
BLOCK_IDS = ("txt_lw_0", "txt_lw_1", "txt_rf_0", "txt_rf_1")
BLOCK_STAGE = {"txt_lw_0": "lw_0", "txt_lw_1": "lw_1", "txt_rf_0": "rf_0", "txt_rf_1": "rf_1"}
BLOCK_PATH = {"txt_lw_0": "txtfusion.layerwise_blocks.0", "txt_lw_1": "txtfusion.layerwise_blocks.1",
              "txt_rf_0": "txtfusion.refiner_blocks.0", "txt_rf_1": "txtfusion.refiner_blocks.1"}
# the Linears that write into the text residual stream: a direction there is a direction in their output columns
EDITABLE_LINEARS = ("attn.wo", "mlp.down")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}

DEFAULTS = {
    "base_prompts": [],
    "target_tags": [],
    "control_tags": [],
    "auto_control_tags": None,          # None: pick only when no controls are listed; true: top up; false: never
    "caption_folder": "",
    "caption_records": {},
    "verification_records": {},
    "caption_count": 32,
    "caption_seed": 1234,
    "caption_max_tags": 40,
    "caption_min_tags": 3,
    "control_tags_per_target": 2,
    "control_min_captions": 2,
    "edit_blocks": ["txt_rf_0", "txt_rf_1"],
    "edit_linears": ["attn.wo", "mlp.down"],
    "columns": 64,
    "num_directions": 1,
    "strength": 1.0,
    "rank": 4,
    "compute_dtype": "fp32",
    "save_dtype": "fp32",
    "random_seed": 0,
    "render": False,
    "render_prompts": [],
    "render_count": 2,
    # v2
    "rounds": 1,                        # edit rounds, each with its own captions and (auto) controls
    "verify_count": 6,                  # captions held out of the fit to check the edit on (0: verify in-sample)
    "column_select": "activation",      # "activation": weight x typical input size; "weight": weights alone (v1)
    "activation_prompts": 8,            # prompts the input sizes are read from
    "ram_budget_gb": 6.0,               # encoded prompts beyond this go to a temp folder (0: keep all in RAM)
    "scan": False,                      # pick targets and controls by scanning the whole caption folder
    "scan_min_count": 8,
    "scan_per_tag": 6,
    "scan_max_tags": 500,
    "scan_targets": 8,
    "scan_auto_top_fraction": 1.0,  # automatic eligible frequency rank; calibration run overrides
    "scan_controls": 12,
    "scan_z": 1.0,                      # weak = below -scan_z standard deviations in BOTH halves of its samples
    "scan_stage": "rf_1",               # stage the scan reads (txtmlp starts with an RMSNorm, which erases magnitude)
    "score_stage": "txtmlp",            # stage suppression is scored at (v1: txtmlp)
}

# the built-in sample (Direction edit config empty, no Target tags, no Caption folder): hand-written bases and
# placeholder tags, to see the pipeline run end to end. Its tags are never mixed into caption mode.
SAMPLE_CONFIG = {
    "base_prompts": [
        "a woman standing in a city street, daytime, photo",
        "a man sitting on a park bench, autumn, photo",
        "a girl walking on a beach at sunset, photo",
        "an old man in a kitchen, warm light, photo",
        "a young woman on a train platform, overcast, photo",
        "a boy in a garden, spring, photo",
    ],
    "target_tags": ["holding an umbrella", "red scarf"],
    "control_tags": ["blue jacket", "smiling", "wearing glasses", "short hair"],
}

_OPTION_PREFIX = "direction_edit_"


# ---- config ---------------------------------------------------------------------------------------------------
def _tag_list(value):
    """A tag list from a JSON list or a comma-separated text: stripped, empties and repeats dropped, order kept."""
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    out = []
    for t in items:
        t = str(t).strip()
        if t and _norm(t) not in {_norm(o) for o in out}:
            out.append(t)
    return out


def _truthy(v):
    return str(v).strip().lower() in ("1", "true", "on", "yes")


def _block_id(name):
    n = str(name).strip().lower()
    n = n if n.startswith("txt_") else f"txt_{n}"
    if n not in BLOCK_IDS:
        raise ValueError(f"edit_blocks: {name!r} is not a text-fusion block (use {', '.join(BLOCK_IDS)}).")
    return n


def make_config(options=None, default_caption_folder=None, config_path=None):
    """The edit's config from the run's family options (direction_edit_config / _caption_folder / _target_tags /
    _control_tags / _caption_count / _strength / _render), a config JSON file and the defaults.

    Caption mode (real captions as the base prompts) turns on when a Caption folder is given, Target tags are
    typed, or the config file has no base_prompts; the folder is the Caption folder, else the config's, else
    `default_caption_folder` (the Start tab's training folder). With no config file and none of those, the built-in
    sample runs. Raises ValueError with a plain message for anything wrong."""
    opts = {str(k): v for k, v in (options or {}).items()}

    def opt(name):
        v = opts.get(_OPTION_PREFIX + name)
        return None if v is None or str(v).strip() == "" else str(v).strip()

    config_path = config_path or opt("config")
    ui_folder, ui_targets, ui_controls = opt("caption_folder"), opt("target_tags"), opt("control_tags")
    cfg = copy.deepcopy(DEFAULTS)
    from_file = {}
    if config_path:
        if not os.path.isfile(config_path):
            raise ValueError(f"the Direction edit config {config_path!r} doesn't exist.")
        try:
            with open(config_path, encoding="utf-8") as f:
                from_file = json.load(f)
        except (OSError, ValueError) as e:
            raise ValueError(f"the Direction edit config {config_path!r} isn't valid JSON ({e}).") from None
        if not isinstance(from_file, dict):
            raise ValueError("the Direction edit config must be a JSON object of settings.")
        unknown = sorted(set(from_file) - set(DEFAULTS) - {"comment", "_comment", "notes"})
        if unknown:
            raise ValueError(f"the Direction edit config has unknown settings: {', '.join(unknown)} "
                             f"(known: {', '.join(sorted(DEFAULTS))}).")
        cfg.update({k: v for k, v in from_file.items() if k in DEFAULTS})
    scan_on = _truthy(opt("scan")) if opt("scan") is not None else _truthy(cfg.get("scan"))
    caption_mode = bool(ui_folder or ui_targets or scan_on or (config_path and not from_file.get("base_prompts")))
    if not config_path and not caption_mode:
        cfg.update(copy.deepcopy(SAMPLE_CONFIG))
        cfg["config_source"] = "sample"
    else:
        cfg["config_source"] = config_path or "options"
    if ui_targets:
        cfg["target_tags"] = ui_targets
    if ui_controls:
        cfg["control_tags"] = ui_controls
    labels = {"caption_count": "Captions to sample", "strength": "Edit strength", "rounds": "Rounds",
              "verify_count": "Held-out captions"}
    for name, cast in (("caption_count", int), ("strength", float), ("rounds", int), ("verify_count", int)):
        v = opt(name)
        if v is not None:
            try:
                cfg[name] = cast(v)
            except ValueError:
                raise ValueError(f"{labels[name]} must be a number, not {v!r}.") from None
    if opt("scan") is not None:
        cfg["scan"] = _truthy(opt("scan"))
    if opt("render") is not None:
        cfg["render"] = _truthy(opt("render"))
    if opt("auto_controls") is not None:
        cfg["auto_control_tags"] = _truthy(opt("auto_controls"))
    cfg["caption_mode"] = caption_mode
    if caption_mode:
        cfg["base_prompts"] = []
        folder = ui_folder or str(cfg.get("caption_folder") or "").strip() or (default_caption_folder or "")
        if not folder:
            raise ValueError("caption mode needs a Caption folder (or a training folder on the Start tab).")
        cfg["caption_folder"] = folder
    return _normalise(cfg)


def _normalise(cfg):
    """Types, ranges and names checked; tag lists as lists. ValueError on anything out of range."""
    cfg["base_prompts"] = [str(p).strip() for p in (cfg.get("base_prompts") or []) if str(p).strip()]
    cfg["target_tags"] = _tag_list(cfg.get("target_tags"))
    cfg["control_tags"] = _tag_list(cfg.get("control_tags"))
    cfg["render_prompts"] = [str(p).strip() for p in (cfg.get("render_prompts") or []) if str(p).strip()]
    try:
        for k in ("caption_count", "caption_seed", "caption_max_tags", "caption_min_tags", "control_tags_per_target",
                  "control_min_captions", "columns", "num_directions", "rank", "random_seed", "render_count",
                  "rounds", "verify_count", "activation_prompts", "scan_min_count", "scan_per_tag", "scan_max_tags",
                  "scan_targets", "scan_controls"):
            cfg[k] = int(cfg[k])
        for k in ("strength", "ram_budget_gb", "scan_z", "scan_auto_top_fraction"):
            cfg[k] = float(cfg[k])
    except (TypeError, ValueError) as e:
        raise ValueError(f"a Direction edit setting isn't a number ({e}).") from None
    if not math.isfinite(cfg["strength"]):
        raise ValueError("Edit strength must be a finite number.")
    if not all(math.isfinite(cfg[k]) and cfg[k] >= 0 for k in ("ram_budget_gb", "scan_z")):
        raise ValueError("ram_budget_gb and scan_z must be finite numbers, 0 or more.")
    if not math.isfinite(cfg["scan_auto_top_fraction"]) or not 0 < cfg["scan_auto_top_fraction"] <= 1:
        raise ValueError("scan_auto_top_fraction must be greater than 0 and at most 1.")
    for k, least in (("caption_count", 2), ("caption_max_tags", 1), ("caption_min_tags", 1),
                     ("control_tags_per_target", 1), ("control_min_captions", 1), ("columns", 0),
                     ("num_directions", 1), ("rank", 1), ("rounds", 1), ("verify_count", 0),
                     ("activation_prompts", 1), ("scan_min_count", 2), ("scan_per_tag", 2), ("scan_max_tags", 1),
                     ("scan_targets", 1), ("scan_controls", 3)):
        if cfg[k] < least:
            raise ValueError(f"{k} must be at least {least}.")
    blocks = cfg.get("edit_blocks") or []
    blocks = blocks.split(",") if isinstance(blocks, str) else blocks
    cfg["edit_blocks"] = list(dict.fromkeys(_block_id(b) for b in blocks if str(b).strip()))
    if not cfg["edit_blocks"]:
        raise ValueError("edit_blocks is empty.")
    lins = cfg.get("edit_linears") or []
    lins = lins.split(",") if isinstance(lins, str) else lins
    cfg["edit_linears"] = list(dict.fromkeys(str(x).strip() for x in lins if str(x).strip()))
    bad = [x for x in cfg["edit_linears"] if x not in EDITABLE_LINEARS]
    if bad or not cfg["edit_linears"]:
        raise ValueError(f"edit_linears must be output Linears ({', '.join(EDITABLE_LINEARS)}); got "
                         f"{', '.join(bad) or 'none'}.")
    for k in ("compute_dtype", "save_dtype"):
        cfg[k] = str(cfg[k]).lower()
        if cfg[k] not in DTYPES:
            raise ValueError(f"{k} must be fp32 or bf16.")
    if cfg["auto_control_tags"] is not None and not isinstance(cfg["auto_control_tags"], bool):
        cfg["auto_control_tags"] = _truthy(cfg["auto_control_tags"])
    cfg["render"] = bool(cfg["render"]) if isinstance(cfg["render"], bool) else _truthy(cfg["render"])
    cfg["scan"] = bool(cfg["scan"]) if isinstance(cfg["scan"], bool) else _truthy(cfg["scan"])
    cfg["column_select"] = str(cfg["column_select"]).strip().lower()
    if cfg["column_select"] not in ("activation", "weight"):
        raise ValueError("column_select must be activation or weight.")
    for k in ("scan_stage", "score_stage"):
        cfg[k] = str(cfg[k]).strip().lower()
        if cfg[k] not in STAGES:
            raise ValueError(f"{k} must be one of {', '.join(STAGES)}.")
    return cfg


def _plan(cfg):
    """The measurement rounds: cfg["round_plan"] ({"base_prompts", "control_tags"} per round, set by apply_captions
    in caption mode), else the one round the config itself describes."""
    plan = cfg.get("round_plan")
    if plan:
        return plan
    return [{"base_prompts": cfg["base_prompts"], "control_tags": cfg["control_tags"]}]


def _needed(bases, targets, controls):
    """Each base alone and with each target and control appended: the prompts one scoring pass reads."""
    out = []
    for b in bases:
        out.append(b)
        out += [append_tag(b, t) for t in list(targets) + list(controls)]
    return list(dict.fromkeys(out))


def check_config(cfg):
    """What the measurement needs: at least 1 target, 3 controls and 2 base prompts (in every round), and no tag on
    both lists."""
    if len(cfg["target_tags"]) < 1:
        raise ValueError("list at least 1 target tag.")
    plan = _plan(cfg)
    if int(cfg.get("rounds", 1)) > 1 and len(plan) < 2:
        raise ValueError("rounds above 1 needs caption mode (a Caption folder): each round draws its own captions.")
    for i, rd in enumerate(plan):
        where = f" in round {i + 1}" if len(plan) > 1 else ""
        if len(rd["control_tags"]) < 3:
            raise ValueError(f"needs at least 3 control tags{where} (has {len(rd['control_tags'])}): list more, or "
                             "let them be picked from the captions.")
        if len(rd["base_prompts"]) < 2:
            raise ValueError(f"needs at least 2 base prompts{where} (has {len(rd['base_prompts'])}).")
    both = {_norm(t) for t in cfg["target_tags"]} & {_norm(c) for c in cfg["control_tags"]}
    if both:
        raise ValueError(f"a tag can't be both a target and a control: {', '.join(sorted(both))}.")
    return cfg


def dataset_main_folder(dataset_config):
    """The Start tab's training folder from the run's dataset TOML: the [[datasets]] block with no sub-folder prefix
    (item_prefix), skipping regularisation sets, edit pairs' control folders and detail-crop folders. None when
    there is no TOML or no such block."""
    if not dataset_config or not os.path.isfile(str(dataset_config)):
        return None
    try:
        import tomllib
        with open(dataset_config, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        try:
            import toml
            data = toml.load(dataset_config)
        except Exception:
            return None
    for ds in data.get("datasets") or []:
        d = str(ds.get("image_directory") or "").strip()
        if not d or ds.get("is_reg") or str(ds.get("item_prefix") or ""):
            continue
        if "fizgig_detail_crops" in d.replace("\\", "/").split("/"):
            continue
        return d
    return None


# ---- captions -------------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=1 << 16)
def _norm(text):
    return re.sub(r"\s+", " ", str(text).replace("_", " ")).strip().lower()


@functools.lru_cache(maxsize=1 << 14)
def _word_pattern(t):
    return re.compile(r"(?<!\w)" + re.escape(t) + r"(?!\w)")


def _holds(part, tag):
    """A caption part holds a tag when it equals it or contains it as whole words ("holding umbrella" holds
    "umbrella"), ignoring case and treating underscores as spaces."""
    p, t = _norm(part), _norm(tag)
    if not t:
        return False
    if p == t:
        return True
    # cheap substring test first: a big caption folder with many targets makes millions of these calls
    return t in p and _word_pattern(t).search(p) is not None


def split_tags(caption):
    """e621-style comma tags, stripped, empties dropped."""
    return [t.strip() for t in str(caption).replace("\n", ",").split(",") if t.strip()]


def strip_tags(tags, remove):
    """Every part that holds one of `remove` taken out."""
    return [t for t in tags if not any(_holds(t, r) for r in remove)]


def captions_holding(captions, tag):
    """How many captions (lists of tags) have a part holding `tag`."""
    return sum(1 for tags in captions if any(_holds(t, tag) for t in tags))


_MEDIA = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".avif", ".jxl", ".tif", ".tiff", ".mp4", ".webm",
          ".mov", ".mkv")


# .txt files that are never captions. Pictures are not looked at at all: only the .txt files are read.
_NOT_CAPTIONS = {"readme.txt", "notes.txt", "license.txt", "licence.txt", "requirements.txt"}


def read_captions(folder):
    """{file name: caption} for the folder's .txt captions (top level; with none there, its resized/ folder).
    Pictures are ignored whether or not they are there; a README or notes file is skipped by name."""
    def scan(d):
        if not os.path.isdir(d):
            return {}
        txt = [n for n in sorted(os.listdir(d))
               if n.lower().endswith(".txt") and n.lower() not in _NOT_CAPTIONS and os.path.isfile(os.path.join(d, n))]
        out = {}
        for n in txt:
            try:
                with open(os.path.join(d, n), encoding="utf-8", errors="replace") as f:
                    text = f.read().strip()
            except OSError:
                continue
            if text:
                out[n] = text
        return out
    caps = scan(folder)
    if not caps:
        caps = {f"resized/{k}": v for k, v in scan(os.path.join(folder, "resized")).items()}
    return caps





def _source_captions(cfg):
    """The actual dataset caption records, or a standalone caption folder for the CLI."""
    records = cfg.get("caption_records")
    if isinstance(records, dict) and records:
        return {str(k): str(v).strip() for k, v in records.items() if str(v or "").strip()}
    return read_captions(cfg.get("caption_folder") or "")


def captions_from_group(group):
    """Read caption1 from every non-regularisation item in Fizgig's assembled training group."""
    from safetensors import safe_open
    out = {}
    for ds_index, ds in enumerate(getattr(group, "datasets", []) or []):
        if getattr(ds, "is_reg", False):
            continue
        bm = getattr(ds, "batch_manager", None)
        for bucket in (getattr(bm, "buckets", {}) or {}).values():
            for it in bucket:
                cache = getattr(it, "text_encoder_output_cache_path", None)
                if not cache or not os.path.isfile(cache):
                    continue
                try:
                    with safe_open(cache, framework="pt", device="cpu") as f:
                        caption = str((f.metadata() or {}).get("caption1") or "").strip()
                except Exception:
                    continue
                if caption:
                    key = str(getattr(it, "item_key", "") or f"dataset-{ds_index}/{len(out)}")
                    if key in out and out[key] != caption:
                        key = f"dataset-{ds_index}/{key}"
                    out[key] = caption
    return out


def captions_from_held(held):
    """Read only the validation hold-out captions, which are never used for scan/fitting."""
    from safetensors import safe_open
    out = {}
    for key, (_reso, it) in (held or {}).items():
        cache = getattr(it, "text_encoder_output_cache_path", None)
        if not cache or not os.path.isfile(cache):
            continue
        try:
            with safe_open(cache, framework="pt", device="cpu") as f:
                caption = str((f.metadata() or {}).get("caption1") or "").strip()
        except Exception:
            continue
        if caption:
            out[str(key)] = caption
    return out

def pick_controls(captions, targets, listed, *, per_target=2, min_captions=2, least=3, exclude=()):
    """Auto controls from the captions' own tags: for target t in n_t captions, candidate c in n_c captions ranks by
    |ln(1 + n_c) - ln(1 + n_t)|, then the more common tag, then the name; the targets take turns. Anything
    overlapping a target or an already picked tag either way is left out, and so are tags in fewer than
    `min_captions` captions. The quota is per_target per target, at least `least` in total, listed controls
    included. Tags in `exclude` (an earlier round's picks) are left out. Returns [{"tag", "captions", "for"}]
    of the new picks."""
    want = max(int(least), int(per_target) * len(targets))
    counts, names = {}, {}
    for tags in captions:
        for t in {_norm(x): x for x in tags}.items():
            counts[t[0]] = counts.get(t[0], 0) + 1
            names.setdefault(t[0], t[1].strip())
    skip = {_norm(x) for x in exclude}
    cand = {k: (names[k], n) for k, n in counts.items()
            if k not in skip and n >= int(min_captions)
            and not any(_holds(names[k], t) or _holds(t, names[k]) for t in targets)}
    picked = [{"tag": c, "listed": True} for c in listed]
    new = []
    ref = {tgt: captions_holding(captions, tgt) for tgt in targets}
    while len(picked) < want:
        before = len(picked)
        for tgt in targets:
            if len(picked) >= want:
                break
            best = sorted((k for k in cand if not any(_holds(cand[k][0], p["tag"]) or _holds(p["tag"], cand[k][0])
                                                      for p in picked)),
                          key=lambda k: (abs(math.log1p(cand[k][1]) - math.log1p(ref[tgt])), -cand[k][1], k))
            if best:
                k = best[0]
                entry = {"tag": cand[k][0], "captions": cand[k][1], "for": tgt}
                picked.append(entry)
                new.append(entry)
        if len(picked) == before:
            break                       # nothing left to pick
    return new


def apply_captions(cfg):
    """Caption mode: the folder's captions become the base prompts. Every target and control tag is stripped from
    each caption first; captions with at least caption_min_tags tags left are eligible. Round 0 draws caption_count
    of them with caption_seed (exactly as v1 did); verify_count more, never used for fitting, are held out to check
    the edit on captions it was not fitted on; each further round (rounds > 1) draws its own captions from what is
    left, with its own auto controls (the earlier rounds' picks left out). Each caption keeps at most
    caption_max_tags tags. The source goes into cfg["caption_source"], the plan into cfg["round_plan"] and
    cfg["verify_prompts"]."""
    if not cfg.get("caption_mode"):
        return cfg
    folder = cfg.get("caption_folder") or "the training datasets"
    caps = _source_captions(cfg)
    if not caps:
        raise ValueError(f"no usable training captions were found in {folder!r}.")
    files = sorted(caps)
    tag_lists = [split_tags(caps[f]) for f in files]
    targets, listed = cfg["target_tags"], list(cfg["control_tags"])
    if not targets:
        raise ValueError("caption mode needs at least 1 target tag (Target tags).")
    auto = cfg.get("auto_control_tags")
    n_rounds = max(1, int(cfg.get("rounds", 1)))
    round_picks, used = [], []
    for _r in range(n_rounds):
        picks = []
        if auto is True or (auto is None and not listed):
            picks = pick_controls(tag_lists, targets, listed, per_target=cfg["control_tags_per_target"],
                                  min_captions=cfg["control_min_captions"], exclude=used)
        round_picks.append(picks)
        used += [p["tag"] for p in picks]
    round_controls = [listed + [p["tag"] for p in picks] for picks in round_picks]
    controls_all = list(dict.fromkeys(c for cs in round_controls for c in cs))
    cfg["control_tags"] = controls_all
    remove = targets + controls_all
    eligible = []
    for f, tags in zip(files, tag_lists):
        kept = strip_tags(tags, remove)
        if len(kept) >= cfg["caption_min_tags"]:
            eligible.append((f, kept[:cfg["caption_max_tags"]]))
    n = min(cfg["caption_count"], len(eligible))
    remaining = list(range(len(eligible)))
    fits = []
    for r in range(n_rounds):
        k = min(n, len(remaining))
        seed = cfg["caption_seed"] if r == 0 else cfg["caption_seed"] + 7919 * r
        chosen = sorted(random.Random(seed).sample(remaining, k)) if k else []
        fits.append(chosen)
        taken = set(chosen)
        remaining = [i for i in remaining if i not in taken]
    v_want = int(cfg.get("verify_count", 0))
    v_n = min(v_want, len(remaining))
    held = sorted(random.Random(cfg["caption_seed"] + 104729).sample(remaining, v_n)) if v_n else []
    plan = [{"base_prompts": [", ".join(eligible[i][1]) for i in fits[r]], "control_tags": round_controls[r],
             "sampled_files": [eligible[i][0] for i in fits[r]]} for r in range(n_rounds)]
    cfg["round_plan"] = plan
    cfg["base_prompts"] = plan[0]["base_prompts"]
    cfg["verify_prompts"] = [", ".join(eligible[i][1]) for i in held]
    cfg["caption_source"] = {
        "folder": folder,
        "captions_read": len(files),
        "eligible": len(eligible),
        "sampled_files": plan[0]["sampled_files"],
        "verify_files": [eligible[i][0] for i in held],
        "target_counts": {t: captions_holding(tag_lists, t) for t in targets},
        "listed_controls": listed,
        "auto_controls": round_picks[0],
        "auto_control_tags": auto,
    }
    if n_rounds > 1:
        cfg["caption_source"]["rounds"] = [{"sampled_files": plan[r]["sampled_files"], "controls": round_controls[r],
                                            "auto_controls": round_picks[r]} for r in range(n_rounds)]
    return cfg


def append_tag(base, tag):
    return f"{base}, {tag}" if base else tag


def prompt_list(cfg):
    """Every prompt the measurement encodes: for each round, its bases alone and with each target and that round's
    controls appended; and the held-out verification bases with every target and control."""
    out = []
    for rd in _plan(cfg):
        out += _needed(rd["base_prompts"], cfg["target_tags"], rd["control_tags"])
    out += _needed(cfg.get("verify_prompts") or [], cfg["target_tags"], cfg["control_tags"])
    return list(dict.fromkeys(out))


# ---- the text path --------------------------------------------------------------------------------------------
class FusionPath(nn.Module):
    """Krea 2's text path alone: txtfusion (2 layerwise blocks, the projector, 2 refiner blocks) and txtmlp, built
    as SingleStreamDiT builds them, so its state dict keys are the DiT's own."""

    def __init__(self, config=None):
        super().__init__()
        from fizgig.krea2.model import RMSNorm, TextFusionTransformer
        if config is None:
            from fizgig.krea2.utils import single_mmdit_large_wide as config
        self.config = config
        self.txtfusion = TextFusionTransformer(config.txtlayers, config.txtdim, config.txtheads, config.multiplier,
                                               config.bias, config.txtkvheads)
        self.txtmlp = nn.Sequential(RMSNorm(config.txtdim), nn.Linear(config.txtdim, config.features),
                                    nn.GELU(approximate="tanh"), nn.Linear(config.features, config.features))

    @torch.no_grad()
    def stages(self, hidden, mask=None):
        """{stage: vector} for one encoded prompt: hidden (seq, layers, dim), mask (seq,) of valid tokens. The
        masked mean over valid tokens at each stage; the taps and the layerwise outputs mixed with the projector's
        weights, as the projector mixes them."""
        from fizgig.krea2.attention import AttentionParams
        p = next(self.parameters())
        x = hidden if mask is None else hidden[mask.bool()]
        x = x.to(device=p.device, dtype=p.dtype)[None]                    # (1, n, L, D), every token valid
        b, n, nl, d = x.shape
        w = self.txtfusion.projector.weight[0].to(x.dtype)               # (L,)
        out = {"taps": torch.einsum("bnld,l->bnd", x, w).mean(1)[0]}
        nomask = AttentionParams.create_attention_params_from_mask("torch", False, 0, None)
        y = x.reshape(b * n, nl, d)
        for i, blk in enumerate(self.txtfusion.layerwise_blocks):
            y = blk(y.contiguous(), attn_params=nomask)
            out[f"lw_{i}"] = torch.einsum("bnld,l->bnd", y.reshape(b, n, nl, d), w).mean(1)[0]
        y = y.reshape(b, n, nl, d).transpose(2, 3)                       # (b, n, d, L)
        y = self.txtfusion.projector(y).squeeze(-1)
        params = AttentionParams.create_attention_params_from_mask("torch", False, 0,
                                                                   torch.ones(b, n, dtype=torch.bool, device=y.device))
        for i, blk in enumerate(self.txtfusion.refiner_blocks):
            y = blk(y, attn_params=params)
            out[f"rf_{i}"] = y.mean(1)[0]
        out["txtmlp"] = self.txtmlp(y).mean(1)[0]
        return {k: v.float().cpu() for k, v in out.items()}


def fusion_keys(config=None):
    with torch.device("meta"):
        return list(FusionPath(config).state_dict().keys())


def load_fusion_path(dit_path, config=None, device="cpu", dtype=torch.float32):
    """Only the text path's tensors (txtfusion.* and txtmlp.*) read from the DiT file - any of its layouts, with
    ComfyUI's prefixes and quantised Linears decoded - into a FusionPath held at `dtype`."""
    from fizgig.krea2.utils import ComfyDecodeReader
    keys = fusion_keys(config)
    with torch.device("meta"):
        path = FusionPath(config)
    sd = {}
    with ComfyDecodeReader(dit_path, keys, calc_device="cpu") as r:
        have = set(r.keys())
        missing = [k for k in keys if k not in have]
        if missing:
            raise ValueError(f"{os.path.basename(dit_path)} has no Krea 2 text path ({len(missing)} tensors missing, "
                             f"e.g. {missing[0]}).")
        for k in keys:
            sd[k] = r.get_tensor(k, device=torch.device("cpu"), dtype=torch.float32).to(dtype)
    path.load_state_dict(sd, assign=True)
    return path.to(device).eval().requires_grad_(False)


# ---- measure, score, direction --------------------------------------------------------------------------------
def measure(path, encoded, prompts):
    """{stage: {prompt: vector}} for every prompt."""
    out = {s: {} for s in STAGES}
    for p in prompts:
        e = encoded[p]
        v = path.stages(e["hidden_states"], e.get("attention_mask"))
        for s in STAGES:
            out[s][p] = v[s]
    return out


def contributions(vecs, cfg):
    """c[stage][tag] = (N, D) tag contributions per base and rel[stage][tag] = (N,) |c| / |v(b)|."""
    bases, tags = cfg["base_prompts"], cfg["target_tags"] + cfg["control_tags"]
    c, rel = {}, {}
    for s in STAGES:
        vb = torch.stack([vecs[s][b] for b in bases])
        c[s], rel[s] = {}, {}
        for t in tags:
            ct = torch.stack([vecs[s][append_tag(b, t)] for b in bases]) - vb
            c[s][t] = ct
            rel[s][t] = ct.norm(dim=1) / vb.norm(dim=1).clamp_min(1e-12)
    return c, rel


def score(rel, cfg):
    """{tag: {"role", "z": {stage: z}, "suppression", "collapse_stage", "collapse_drop"}}: per base, a tag's rel
    against the controls' (a control against the other controls), averaged over the bases."""
    targets, controls = cfg["target_tags"], cfg["control_tags"]
    last = cfg.get("score_stage") or STAGES[-1]
    out = {}
    for t in targets + controls:
        z = {}
        for s in STAGES:
            others = [rel[s][c] for c in controls if c != t]
            ref = torch.stack(others)                                   # (C, N)
            mu, sd = ref.mean(0), ref.std(0, unbiased=len(others) > 1).clamp_min(1e-8)
            z[s] = float(((rel[s][t] - mu) / sd).mean())
        drops = [(z[STAGES[i]] - z[STAGES[i - 1]], STAGES[i]) for i in range(1, len(STAGES))]
        drop, where = min(drops)
        out[t] = {"role": "target" if t in targets else "control", "z": z, "suppression": -z[last],
                  "collapse_stage": where, "collapse_drop": drop,
                  "rel_last": float(rel[last][t].mean())}
    return out


def _unit(v):
    n = v.norm()
    return v / n if float(n) > 0 else v


def _cos(a, b):
    return float(torch.dot(_unit(a), _unit(b)))


def directions(c, cfg, block):
    """U (D, k) for one edited block from the contributions at its output stage, with its reliability."""
    s = BLOCK_STAGE[block]
    targets, controls = cfg["target_tags"], cfg["control_tags"]
    ct = torch.stack([c[s][t] for t in targets])                       # (T, N, D)
    cc = torch.stack([c[s][x] for x in controls])                      # (C, N, D)
    d = ct.mean((0, 1)) - cc.mean((0, 1))
    if float(d.norm()) < 1e-12:
        raise ValueError(f"no direction at {block}: the targets and controls contribute the same there.")
    u = d / d.norm()
    n = ct.shape[1]
    even, odd = list(range(0, n, 2)), list(range(1, n, 2))
    d_even = ct[:, even].mean((0, 1)) - cc[:, even].mean((0, 1))
    d_odd = ct[:, odd].mean((0, 1)) - cc[:, odd].mean((0, 1))
    per_prompt = [_cos(ct[:, i].mean(0) - cc[:, i].mean(0), u) for i in range(n)]
    cols = [u]
    k = int(cfg["num_directions"])
    if k > 1:
        r = (ct - cc.mean(0, keepdim=True)).reshape(-1, ct.shape[-1])  # target contributions over the control mean
        r = r - (r @ u)[:, None] * u[None]
        if r.shape[0] > 0 and float(r.norm()) > 0:
            _u, _s, vh = torch.linalg.svd(r, full_matrices=False)
            cols += [vh[i] for i in range(min(k - 1, vh.shape[0])) if float(_s[i]) > 1e-8 * float(_s[0])]
    q, r = torch.linalg.qr(torch.stack(cols, 1))
    signs = torch.sign(torch.diagonal(r))
    q = q * torch.where(signs == 0, torch.ones_like(signs), signs)    # the first column is u itself, not -u
    info = {"stage": s, "norm": float(d.norm()), "directions": int(q.shape[1]),
            "split_half_cosine": _cos(d_even, d_odd) if odd else None,
            "per_prompt_cosine": per_prompt, "per_prompt_cosine_mean": sum(per_prompt) / len(per_prompt)}
    return q, info


# ---- edit and export ------------------------------------------------------------------------------------------
@torch.no_grad()
def edit_delta(weight, u, strength, columns, scale=None):
    """dW for an output Linear W (out x in) and orthonormal directions U (out x k): +s U U^T W on `columns` columns
    (0 or >= in: every column); zero elsewhere. A positive strength AMPLIFIES what those columns write along U (v1's
    +1 removed it; testing showed the reverse sign was the useful one); a negative one removes it. The columns are
    the top ones by |U^T W[:, j]|, times `scale[j]` (that input unit's typical size when the edited Linear runs)
    when given: a unit that rarely fires can't change the output however its weights point. Returns (delta, column
    indices)."""
    w = weight.float()
    u = u.float().to(w.device)
    proj = u.t() @ w                                               # (k, in)
    n_in = w.shape[1]
    if columns and 0 < int(columns) < n_in:
        rank = proj.norm(dim=0)
        if scale is not None:
            rank = rank * scale.float().to(w.device)
        idx = torch.topk(rank, int(columns)).indices.sort().values
    else:
        idx = torch.arange(n_in, device=w.device)
    delta = torch.zeros_like(w)
    if strength:
        delta[:, idx] = float(strength) * (u @ proj[:, idx])
    return delta, idx


@torch.no_grad()
def selection_energy(weight, u, idx, scale=None):
    """How much of the Linear's output along U the chosen columns carry: sum of |U^T W[:, j]|^2 over the chosen
    columns / over all (\"weights\"), and the same weighted by each input unit's typical size (\"activations\", when
    `scale` is given). Near 0 means the edit touches little of what the Linear writes along U."""
    w = weight.float()
    e = (u.float().to(w.device).t() @ w).pow(2).sum(0)
    out = {"projection_energy_removed": float(e[idx].sum() / e.sum().clamp_min(1e-30))}
    if scale is not None:
        ea = e * scale.float().to(w.device).pow(2)
        out["activation_energy_removed"] = float(ea[idx].sum() / ea.sum().clamp_min(1e-30))
    return out


@torch.no_grad()
def input_scales(path, encoded, prompts, names):
    """{module: (in,)} the RMS of each input unit of the named Linears over every token of `prompts`, measured by a
    forward hook while the text path runs them."""
    sums, counts, hooks = {}, {}, []

    def make_hook(name):
        def pre(_module, args):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            sums[name] = sums.get(name, 0) + (x * x).sum(0).cpu()
            counts[name] = counts.get(name, 0) + x.shape[0]
        return pre
    for n in names:
        hooks.append(path.get_submodule(n).register_forward_pre_hook(make_hook(n)))
    try:
        for p in prompts:
            e = encoded[p]
            path.stages(e["hidden_states"], e.get("attention_mask"))
    finally:
        for h in hooks:
            h.remove()
    return {n: (sums[n] / max(counts[n], 1)).sqrt() for n in names if n in sums}


@torch.no_grad()
def factorize(delta, rank):
    """delta (out x in) -> (up (out x rank), down (rank x in)) with up @ down = delta's best rank-`rank` part; exact
    when delta's rank <= rank. Unused rows / columns are zero."""
    u, s, vh = torch.linalg.svd(delta.float(), full_matrices=False)
    r = min(int(rank), s.shape[0])
    root = s[:r].sqrt()
    up = torch.zeros(delta.shape[0], int(rank))
    down = torch.zeros(int(rank), delta.shape[1])
    up[:, :r] = u[:, :r] * root
    down[:r] = root[:, None] * vh[:r]
    return up, down


def lora_stem(module):
    """Fizgig's kohya stem for a Krea 2 module (families/krea2.py LoRAFormat): lora_unet_<path, dots as _>."""
    return "lora_unet_" + module.replace(".", "_")


def build_edit(path, U, cfg, strength=None, scales=None, blocks=None, stats=None):
    """{module: (delta, column idx)} for every edited Linear of `blocks` (default: all edit_blocks), from `path`'s
    weights. `scales` ({module: per-input-unit size}) ranks columns by activation too when column_select is
    "activation". `stats`, a dict, receives each module's selection_energy."""
    s = cfg["strength"] if strength is None else strength
    use_act = scales is not None and cfg.get("column_select", "activation") == "activation"
    out = {}
    for block in (blocks or cfg["edit_blocks"]):
        for lin in cfg["edit_linears"]:
            name = f"{BLOCK_PATH[block]}.{lin}"
            w = path.get_submodule(name).weight
            sc = scales.get(name) if use_act else None
            out[name] = edit_delta(w, U[block], s, cfg["columns"], scale=sc)
            if stats is not None:
                stats[name] = selection_energy(w, U[block], out[name][1], sc)
    return out


@torch.no_grad()
def apply_edit(path, edit):
    """A copy of the path with every delta added (the original is left as it is)."""
    edited = copy.deepcopy(path)
    for name, (delta, _idx) in edit.items():
        m = edited.get_submodule(name)
        m.weight.add_(delta.to(m.weight.device, m.weight.dtype))
    return edited


def _activation_prompts(rc, cfg):
    """The prompts the activation scales are read from: a few bases and the same with the first targets appended."""
    n = max(1, int(cfg.get("activation_prompts", 8)))
    bases = rc["base_prompts"][:n]
    return list(dict.fromkeys(bases + [append_tag(b, t) for b in bases for t in cfg["target_tags"][:2]]))


@torch.no_grad()
def run_on_path(path, encoded, cfg):
    """Steps 1-5 on a loaded text path and the encoded prompts ({prompt: {"hidden_states", "attention_mask"}}).
    For each round, for each edited block in pipeline order: measure on the path AS EDITED SO FAR, score, direction,
    edit (so a later block's direction sees the earlier blocks' edits); the strength is split over the rounds and
    the deltas summed. Verify on the held-out captions when there are any (else on round 0's), against a random
    direction rescaled to the same size per Linear. Returns the result dict export() writes."""
    check_config(cfg)
    prompts = prompt_list(cfg)
    missing = [p for p in prompts if p not in encoded]
    if missing:
        raise ValueError(f"{len(missing)} prompts were not encoded (e.g. {missing[0]!r}).")
    plan = _plan(cfg)
    rounds = len(plan)
    per_round = float(cfg["strength"]) / rounds
    blocks = sorted(cfg["edit_blocks"], key=BLOCK_IDS.index)
    names_of = {b: [f"{BLOCK_PATH[b]}.{lin}" for lin in cfg["edit_linears"]] for b in blocks}
    use_act = cfg.get("column_select", "activation") == "activation"
    work = copy.deepcopy(path)
    total, stats, U_all, reliability = {}, {}, {}, {}
    before_fit = None
    for r, rd in enumerate(plan):
        rc = {**cfg, "base_prompts": rd["base_prompts"], "control_tags": rd["control_tags"]}
        rprompts = _needed(rc["base_prompts"], cfg["target_tags"], rc["control_tags"])
        for block in blocks:
            c, rel = contributions(measure(work, encoded, rprompts), rc)
            if before_fit is None:
                before_fit = score(rel, rc)
            key = block if rounds == 1 else f"{block}@r{r + 1}"
            U_all[key], reliability[key] = directions(c, rc, block)
            scales = input_scales(work, encoded, _activation_prompts(rc, cfg), names_of[block]) if use_act else None
            st = {}
            part = build_edit(work, {block: U_all[key]}, rc, strength=per_round, scales=scales, blocks=[block],
                              stats=st)
            for name, (delta, _idx) in part.items():
                m = work.get_submodule(name)
                m.weight.add_(delta.to(m.weight.device, m.weight.dtype))
                d = delta.detach().float().cpu()
                total[name] = d if name not in total else total[name] + d
                stats.setdefault(name, []).append({"round": r + 1, **st[name]})
    edit = {}
    for name, d in total.items():
        edit[name] = (d, (d.abs().sum(0) > 0).nonzero().squeeze(1))
    # verify: held-out captions when there are any, else round 0's own
    held = list(cfg.get("verify_prompts") or [])
    if held:
        vcfg = {**cfg, "base_prompts": held, "control_tags": cfg["control_tags"]}
    else:
        vcfg = {**cfg, "base_prompts": plan[0]["base_prompts"], "control_tags": plan[0]["control_tags"]}
    vprompts = _needed(vcfg["base_prompts"], cfg["target_tags"], vcfg["control_tags"])

    def scored(p):
        return score(contributions(measure(p, encoded, vprompts), vcfg)[1], vcfg)
    before = scored(path)
    after = scored(work)
    g = torch.Generator().manual_seed(int(cfg["random_seed"]))
    U_rand = {}
    for block in blocks:
        ref = next(U_all[k] for k in U_all if k == block or k.startswith(block + "@"))
        q, _ = torch.linalg.qr(torch.randn(ref.shape, generator=g))
        U_rand[block] = q
    scales0 = None
    if use_act:
        every = [n for b in blocks for n in names_of[b]]
        scales0 = input_scales(path, encoded, _activation_prompts(
            {"base_prompts": plan[0]["base_prompts"]}, cfg), every)
    rand_edit = build_edit(path, U_rand, cfg, strength=float(cfg["strength"]), scales=scales0)
    rand_norm = {}
    for name, (d, idx) in list(rand_edit.items()):
        rn, tn = float(d.norm()), float(edit[name][0].norm())
        if rn > 0 and tn > 0:                   # the control is as large as the real edit, Linear by Linear
            rand_edit[name] = (d * (tn / rn), idx)
        rand_norm[name] = float(rand_edit[name][0].norm())
    random_ctl = scored(apply_edit(path, rand_edit))
    tg, ctl = cfg["target_tags"], vcfg["control_tags"]

    def mean_sup(sc, tags):
        return sum(sc[t]["suppression"] for t in tags) / len(tags)

    def drift(sc):
        return sum(abs(sc[c]["suppression"] - before[c]["suppression"]) for c in ctl) / len(ctl)
    sb, sa, sr = mean_sup(before, tg), mean_sup(after, tg), mean_sup(random_ctl, tg)
    verify = {"held_out": bool(held), "verify_captions": len(vcfg["base_prompts"]),
              "target_suppression_before": sb, "target_suppression_after": sa, "target_suppression_random": sr,
              "improvement": sb - sa, "random_improvement": sb - sr,
              "change": sa - sb, "random_change": sr - sb,
              "control_drift": drift(after), "random_control_drift": drift(random_ctl)}
    # an edit helps when it LOWERS the targets' suppression and by more than a random one of the same size does
    verify["beats_random"] = bool(verify["improvement"] > 0 and verify["improvement"] > verify["random_improvement"])
    return {"scores": {"before": before, "after": after, "random": random_ctl}, "scores_fit_before": before_fit,
            "reliability": reliability, "directions": {k: U_all[k].tolist() for k in U_all}, "edit": edit,
            "edit_stats": stats, "random_delta_norm": rand_norm, "verify": verify, "prompts_encoded": len(prompts),
            "rounds": rounds}


def lora_state_dict(edit, rank, dtype=torch.float32):
    """The edit as kohya LoRA tensors (alpha = rank, so the scale is 1 and strength 1 is the edit)."""
    sd, err = {}, {}
    for name, (delta, _idx) in edit.items():
        delta = delta.detach().float().cpu()      # the edit can be measured on the GPU; the file is written from CPU
        up, down = factorize(delta, rank)
        up, down = up.cpu(), down.cpu()
        stem = lora_stem(name)
        sd[f"{stem}.lora_down.weight"] = down.to(dtype).contiguous()
        sd[f"{stem}.lora_up.weight"] = up.to(dtype).contiguous()
        sd[f"{stem}.alpha"] = torch.tensor(float(rank))
        dn = float(delta.norm())
        err[name] = float((up @ down - delta.float()).norm()) / dn if dn > 0 else 0.0
    return sd, err


def export(result, cfg, output_dir, output_name, *, checkpoint=None, extra=None):
    """Write only the calibration LoRA. Verification details remain in memory for the training decision."""
    from safetensors.torch import save_file
    os.makedirs(output_dir, exist_ok=True)
    lora_path = os.path.join(output_dir, f"{output_name}.safetensors")
    rank = max(int(cfg["rank"]), int(cfg["num_directions"]) * int(result.get("rounds", 1)))
    sd, _err = lora_state_dict(result["edit"], rank, DTYPES[cfg["save_dtype"]])
    meta = {
        "ss_network_module": "networks.lora",
        "ss_network_dim": str(rank),
        "ss_network_alpha": str(rank),
        "ss_base_model_version": "krea2",
        "modelspec.architecture": "Krea-2/lora",
    }
    save_file(sd, lora_path, metadata=meta)
    return {"lora": lora_path}


# ---- whole-dataset scan: find suppressed tags; manual targets take priority, auto fills the usual quota --------
def plan_scan(cfg):
    """The ablation prompts for a scan of the caption folder: for each frequent tag (in at least scan_min_count
    captions, the scan_max_tags most frequent), up to scan_per_tag real captions that contain it, each as it stands
    and with that tag taken out. Returns (prompts to encode, pairs [{"tag", "caption", "full", "ablated"}], meta)."""
    folder = cfg.get("caption_folder") or "the training datasets"
    caps = _source_captions(cfg)
    if not caps:
        raise ValueError(f"no usable training captions were found in {folder!r}.")
    files = sorted(caps)
    maxt, mint = int(cfg["caption_max_tags"]), int(cfg["caption_min_tags"])
    heads = [split_tags(caps[f]) for f in files]
    where, names = {}, {}
    for i, head in enumerate(heads):
        # Count every distinct tag in the complete caption, even those beyond the prompt context limit.
        for k in {_norm(t) for t in head if _norm(t)}:
            where.setdefault(k, []).append(i)
        for t in head:
            names.setdefault(_norm(t), t.strip())
    counts = {k: len(v) for k, v in where.items()}
    requested = {_norm(t) for t in cfg.get("target_tags") or []}
    frequent = sorted((k for k, n in counts.items() if n >= cfg["scan_min_count"]),
                      key=lambda k: (-counts[k], k))[:cfg["scan_max_tags"]]
    # Scan common tags automatically; explicit manual requests can enter the
    # scan below the frequency tier when there are at least four captions.
    # Keep at least eight automatic candidates for regression/control fitting.
    common_count = min(len(frequent), max(8, math.ceil(
        len(frequent) * float(cfg.get("scan_auto_top_fraction", 1.0)))))
    common = frequent[:common_count]
    tags = list(dict.fromkeys(common + [k for k in sorted(requested) if counts.get(k, 0) >= 4]))
    rng = random.Random(cfg["caption_seed"] + 31)
    pairs, prompts = [], []
    for k in tags:
        pool = [i for i in where[k] if sum(1 for t in heads[i] if _norm(t) != k) >= mint]
        if len(pool) < 2:
            continue
        for i in sorted(rng.sample(pool, min(int(cfg["scan_per_tag"]), len(pool)))):
            full_tags = heads[i]
            context = list(full_tags[:maxt])
            target = next((t for t in full_tags if _norm(t) == k), names[k])
            if not any(_norm(t) == k for t in context):
                context = context[:max(0, maxt - 1)] + [target]
            full = ", ".join(context)
            ablated = ", ".join(t for t in context if _norm(t) != k)
            pairs.append({"tag": k, "caption": i, "full": full, "ablated": ablated})
            prompts += [full, ablated]
    return list(dict.fromkeys(prompts)), pairs, {
        "names": names, "counts": counts, "files": files, "auto_keys": set(common),
        "auto_pool_count": len(common), "frequent_count": len(frequent)
    }


@torch.no_grad()
def scan_measure(path, encoded, pairs):
    """Per pair, the tag's contribution in its own caption at every stage: |v(full) - v(ablated)| / |v(ablated)|."""
    cache = {}

    def vec(p):
        if p not in cache:
            e = encoded[p]
            cache[p] = path.stages(e["hidden_states"], e.get("attention_mask"))
        return cache[p]
    rows = []
    for pr in pairs:
        f, a = vec(pr["full"]), vec(pr["ablated"])
        rows.append({s: float((f[s] - a[s]).norm() / a[s].norm().clamp_min(1e-12)) for s in STAGES})
    return rows


# General SFW control concepts in broad categories, sampled reproducibly and evenly.
SFW_CONTROL_TAGS = {
    "people": ("portrait", "smiling", "standing", "sitting", "looking at viewer", "profile", "group"),
    "clothing": ("shirt", "jacket", "dress", "boots", "gloves", "hat", "scarf"),
    "animals": ("dog", "cat", "bird", "horse", "rabbit", "fox", "deer"),
    "objects": ("book", "chair", "table", "lamp", "camera", "bottle", "bag"),
    "places": ("forest", "city", "beach", "park", "bedroom", "garden", "street"),
    "architecture": ("house", "building", "castle", "bridge", "window", "door", "tower"),
    "activities": ("walking", "running", "reading", "dancing", "cooking", "working", "playing"),
    "nature": ("flowers", "tree", "grass", "clouds", "sunset", "river", "mountains"),
    "lighting": ("sunlight", "backlighting", "soft lighting", "overcast", "warm light", "daylight"),
    "composition": ("close-up", "full body", "side view", "wide shot", "upper body", "from behind"),
    "materials": ("wood", "metal", "glass", "leather", "stone", "fabric"),
    "colours": ("blue", "red", "green", "black", "white", "yellow", "purple"),
}


def builtin_controls(targets, seed=1234, per_category=2):
    """A reproducible category-balanced SFW sample, excluding targets and aliases."""
    rng = random.Random(int(seed))
    chosen = []
    for _category, tags in SFW_CONTROL_TAGS.items():
        pool = [t for t in tags if not any(_holds(t, x) or _holds(x, t) for x in targets + chosen)]
        rng.shuffle(pool)
        chosen.extend(pool[:max(1, int(per_category))])
    return chosen


def scan_select(cfg, pairs, rows, meta, requested_targets=None):
    """Rank the scanned tags and pick targets and controls. A tag's score is its mean log contribution at scan_stage
    (rf_1 by default: txtmlp starts with an RMSNorm, which erases how loud a token is). A short or rare tag
    legitimately moves the vector less, so the score is regressed on log(characters) and log(caption count) and
    the residual, in standard deviations, is what is compared. A tag counts as weak only when BOTH halves of its
    samples (alternate captions) fall below -scan_z. Automatic-only selection picks the weakest confirmed tags
    (up to scan_targets). With manual requests, keep confirmed manual targets first and fill to the SAME automatic-only
    target count using the weakest remaining confirmed tags; do not add extra targets beyond that count.
    If more manual targets qualify than the automatic budget, preserve them rather than silently discard user input.
    Controls: tags near the middle (|z| <= 0.5) closest to the targets in length and frequency (scan_controls).
    Returns {"targets", "manual_targets", "automatic_targets", "auto_only_count", "controls", "rows"}."""
    stage = cfg.get("scan_stage") or "rf_1"
    by_tag = {}
    for pr, row in zip(pairs, rows):
        by_tag.setdefault(pr["tag"], []).append(math.log(max(row[stage], 1e-12)))
    tags = sorted(t for t, v in by_tag.items() if len(v) >= 2)
    if len(tags) < 8:
        requested = list(requested_targets or cfg.get("target_tags") or [])
        skipped = [{"tag": t, "reason": f"only {len(tags)} distinct tags could be measured; at least 8 are needed"}
                   for t in requested]
        return {"targets": [], "manual_targets": [], "automatic_targets": [],
                "auto_only_count": 0, "controls": [], "skipped_targets": skipped, "rows": [],
                "fit": {"reason": f"only {len(tags)} distinct tags could be measured"}}
    mean = lambda v: sum(v) / len(v)          # noqa: E731
    y = torch.tensor([mean(by_tag[t]) for t in tags], dtype=torch.float64)
    ya = torch.tensor([mean(by_tag[t][0::2]) for t in tags], dtype=torch.float64)
    yb = torch.tensor([mean(by_tag[t][1::2]) for t in tags], dtype=torch.float64)
    feat = torch.tensor([[math.log(len(meta["names"][t])), math.log(meta["counts"][t])] for t in tags],
                        dtype=torch.float64)
    X = torch.cat([torch.ones(len(tags), 1, dtype=torch.float64), feat], 1)
    beta = torch.linalg.lstsq(X, y[:, None]).solution
    fit = (X @ beta)[:, 0]
    resid = y - fit
    sd = float(resid.std()) or 1e-12
    z, za, zb = resid / sd, (ya - fit) / sd, (yb - fit) / sd
    zmin = float(cfg["scan_z"])
    confirmed = [bool(za[i] < -zmin and zb[i] < -zmin) for i in range(len(tags))]
    auto_keys = set(meta.get("auto_keys", tags))
    # Only the most common eligible tags can be automatic targets; among
    # confirmed weak concepts prefer prevalence, then stronger suppression.
    order = sorted((i for i in range(len(tags)) if tags[i] in auto_keys),
                   key=lambda i: (-meta["counts"][tags[i]], float(z[i]), tags[i]))
    skipped = []
    requested = list(requested_targets or [])
    auto_only = [i for i in order if confirmed[i]][:int(cfg["scan_targets"])]
    manual = []
    if requested:
        by_key = {tag: i for i, tag in enumerate(tags)}
        for raw in requested:
            i = by_key.get(_norm(raw))
            if i is None:
                skipped.append({"tag": raw, "reason": "not present often enough in the training captions to measure"})
            elif not confirmed[i]:
                skipped.append({"tag": raw, "reason": "no consistent measured suppression",
                                "z": float(z[i]), "z_half_a": float(za[i]), "z_half_b": float(zb[i])})
            elif i not in manual:
                manual.append(i)
    # The calibration is not made larger just because the user has entered targets.
    # E.g. auto-only finds six; two qualifying manual targets + four auto targets = six.
    # Manual targets outside the top automatic ranks still take priority.
    automatic = [i for i in auto_only if i not in manual][:max(0, len(auto_only) - len(manual))]
    tgt = manual + automatic
    target_names = [meta["names"][tags[i]] for i in tgt]
    ref = feat[tgt].mean(0) if tgt else feat.mean(0)
    near = sorted((i for i in range(len(tags)) if abs(float(z[i])) <= 0.5 and i not in tgt),
                  key=lambda i: float(((feat[i] - ref) ** 2).sum()))
    controls = []
    for i in near:
        nm = meta["names"][tags[i]]
        if any(_holds(nm, o) or _holds(o, nm) for o in target_names + controls):
            continue
        controls.append(nm)
        if len(controls) >= int(cfg["scan_controls"]):
            break
    if tgt and len(controls) < 3:
        raise ValueError(f"the scan found only {len(controls)} mid-range tags to use as controls (needs 3): "
                         "raise scan_max_tags or lower scan_min_count.")
    role = {tags[i] for i in tgt}
    control_keys = {_norm(c) for c in controls}
    out_rows = []
    for i, t in enumerate(tags):
        nm = meta["names"][t]
        out_rows.append({"tag": nm, "count": meta["counts"][t], "samples": len(by_tag[t]), "chars": len(nm),
                         "mean_log_rel": float(y[i]), "z": float(z[i]), "z_half_a": float(za[i]),
                         "z_half_b": float(zb[i]), "confirmed_weak": confirmed[i],
                         "role": "target" if t in role else ("control" if t in control_keys else "")})
    out_rows.sort(key=lambda r: r["z"])
    return {"targets": target_names, "manual_targets": [meta["names"][tags[i]] for i in manual],
            "automatic_targets": [meta["names"][tags[i]] for i in automatic],
            "auto_only_count": len(auto_only), "controls": controls, "skipped_targets": skipped, "rows": out_rows,
            "fit": {"coefficients": [float(b) for b in beta[:, 0]], "residual_sd": sd}}


def _run_scan(cfg, store, *, dit_path, te_path, device, encoder, path_config, output_dir, output_name):
    """Measure tags found in training captions and select suppressed targets and suitable controls."""
    if not dit_path or not os.path.isfile(dit_path):
        raise ValueError(f"the Krea 2 DiT {dit_path!r} doesn't exist.")
    prompts, pairs, meta = plan_scan(cfg)
    if not pairs:
        raise ValueError("the scan found no tag in enough captions: lower scan_min_count.")
    encode_prompts(prompts, te_path, device, encoder=encoder, store=store)
    path = load_fusion_path(dit_path, path_config, device=device, dtype=DTYPES[cfg["compute_dtype"]])
    try:
        rows = scan_measure(path, store, pairs)
    finally:
        del path
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selection = scan_select(cfg, pairs, rows, meta, requested_targets=cfg.get("target_tags") or [])
    logger.info("[calibration] target scan: eligible frequent tags %d of %d; automatic budget %d",
                meta["auto_pool_count"], meta["frequent_count"], cfg["scan_targets"])
    logger.info("[calibration] target scan: auto-only %d, combined %d (%d manual + %d automatic)",
                selection["auto_only_count"], len(selection["targets"]),
                len(selection["manual_targets"]), len(selection["automatic_targets"]))
    manual_keys = {_norm(name) for name in selection["manual_targets"]}
    for row in selection["rows"]:
        if row["role"] == "target":
            source = "manual" if _norm(row["tag"]) in manual_keys else "automatic"
            logger.info("[calibration] candidate target: %s (%s; suppression z=%.2f; %d captions)",
                        row["tag"], source, row["z"], row["count"])
    for excluded in selection["skipped_targets"]:
        logger.info("[calibration] manual target skipped: %s (%s)", excluded["tag"], excluded["reason"])
    store.clear()
    dataset_controls = selection["controls"]
    general_controls = builtin_controls(selection["targets"], seed=cfg["caption_seed"])
    all_controls = list(dict.fromkeys(dataset_controls + general_controls))
    cfg["target_tags"], cfg["control_tags"], cfg["auto_control_tags"] = selection["targets"], all_controls, False


# ---- the whole run --------------------------------------------------------------------------------------------
class EncodedStore:
    """{prompt: {"hidden_states", "attention_mask"}} for the encoded prompts, behaving like that dict. Each prompt's
    hidden states are ~60 KB per token (12 layers x 2560 x bf16), 7-9 MB per caption, so a few thousand prompts
    outgrow RAM: past `budget_bytes` (0 = no limit) a prompt is written to a temporary folder instead and read back
    when it is measured. close() deletes the folder."""

    def __init__(self, budget_bytes=0):
        self.budget = int(budget_bytes or 0)
        self._ram, self._disk, self._bytes, self._dir = {}, {}, 0, None

    def __setitem__(self, prompt, value):
        self.pop(prompt)
        h = value["hidden_states"]
        size = h.numel() * h.element_size()
        if self.budget and self._bytes + size > self.budget:
            if self._dir is None:
                self._dir = tempfile.mkdtemp(prefix="fizgig_direction_edit_")
            f = os.path.join(self._dir, f"{len(self._disk)}_{abs(hash(prompt)) % (1 << 31)}.pt")
            torch.save(h.detach().cpu().contiguous(), f)
            self._disk[prompt] = f
        else:
            self._ram[prompt] = {"hidden_states": h, "attention_mask": value.get("attention_mask")}
            self._bytes += size

    def pop(self, prompt):
        e = self._ram.pop(prompt, None)
        if e is not None:
            h = e["hidden_states"]
            self._bytes -= h.numel() * h.element_size()
        f = self._disk.pop(prompt, None)
        if f is not None and os.path.isfile(f):
            os.remove(f)

    def __getitem__(self, prompt):
        if prompt in self._ram:
            return self._ram[prompt]
        return {"hidden_states": torch.load(self._disk[prompt]), "attention_mask": None}

    def __contains__(self, prompt):
        return prompt in self._ram or prompt in self._disk

    def __len__(self):
        return len(self._ram) + len(self._disk)

    def __iter__(self):
        yield from list(self._ram)
        yield from list(self._disk)

    def keys(self):
        return list(iter(self))

    def items(self):
        for k in self:
            yield k, self[k]

    def values(self):
        for k in self:
            yield self[k]

    @property
    def spilled(self):
        return len(self._disk)

    def clear(self):
        self._ram, self._disk, self._bytes = {}, {}, 0
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None

    close = clear


def encode_prompts(prompts, te_path, device, driver=None, encoder=None, te=None, store=None, chunk=32):
    """An EncodedStore ({prompt: {"hidden_states" (valid tokens, layers, dim) bf16, "attention_mask"}}) with every
    prompt encoded: Qwen3-VL loaded once (or `te`, already loaded, left loaded), `chunk` prompts per call to the
    driver's encode_text (one prompt per forward inside), each result stored as it comes so RAM never holds the
    raw list, then freed. Prompts already in `store` are skipped. `encoder` (a callable list -> list of the same
    dicts) replaces the model in tests."""
    store = store if store is not None else EncodedStore()
    todo = [p for p in dict.fromkeys(prompts) if p not in store]
    if not todo:
        return store
    own_te = False
    if encoder is not None:
        encode = encoder
    else:
        from fizgig.krea2.driver import Krea2Driver
        driver = driver or Krea2Driver()
        if te is None:
            te = driver.load_text_encoder(te_path, device)
            own_te = True

        def encode(part):
            return driver.encode_text(te, part)
    try:
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            for p, e in zip(part, encode(part)):
                h, m = e["hidden_states"], e.get("attention_mask")
                if m is not None:
                    h = h[m.bool()]
                store[p] = {"hidden_states": h, "attention_mask": None}     # valid tokens only
    finally:
        if own_te:
            driver.unload_text_encoder(te)
    return store


def render_prompts(cfg):
    if cfg["render_prompts"]:
        return cfg["render_prompts"]
    return [append_tag(b, ", ".join(cfg["target_tags"])) for b in cfg["base_prompts"][:max(1, cfg["render_count"])]]


def render(dit_path, vae_path, conds, neg_cond, edit, output_dir, output_name, *, seed, width, height, device):
    """Fixed-seed pictures with the full model, without and with the edit (merged into the text path's weights).
    RAW sampling (28 steps, CFG 4.5); blocks swap on a card under 40 GB."""
    from fizgig.families.registry import get as get_family
    from fizgig.krea2.driver import Krea2Driver
    desc = get_family("krea2")
    driver = Krea2Driver()
    raw = desc.sampling[0]
    dev = torch.device(device)
    small = dev.type == "cuda" and torch.cuda.get_device_properties(dev).total_memory < 40 * 1024 ** 3
    dit = driver.load_dit(dit_path, "cpu" if small else dev)
    if small:
        driver.enable_block_swap(dit, driver.max_blocks_to_swap(dit), dev, supports_backward=False)
        driver.block_swap_mode(dit, inference=True)
    vae = driver.load_vae(vae_path, dev)
    width, height = int(width or 1024), int(height or 1024)
    files = []
    for phase in ("before", "after"):
        if phase == "after":
            with torch.no_grad():
                for name, (delta, _idx) in edit.items():
                    m = dit.get_submodule(name)
                    m.weight.add_(delta.to(m.weight.device, m.weight.dtype))
        for i, cond in enumerate(conds):
            lat = driver.generate(dit, cond, width, height, steps=raw.steps, seed=seed, cfg=raw.cfg,
                                  neg_cond=neg_cond)
            img = driver.decode(vae, lat, width, height)
            f = os.path.join(output_dir, f"{output_name}_render_{i}_{phase}.png")
            img.save(f)
            files.append(f)
    del dit, vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return files


def run_direction_edit(cfg, *, dit_path, te_path, vae_path=None, output_dir, output_name, render_seed=42,
                       render_width=None, render_height=None, speed_lora=None, speed_lora_strength=None,
                       device=None, encoder=None, path_config=None):
    """Run calibration and return its temporary LoRA plus the in-memory verification result."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    store = EncodedStore(int(float(cfg.get("ram_budget_gb", 6.0)) * (1 << 30)))
    try:
        if cfg.get("scan"):
            _run_scan(cfg, store, dit_path=dit_path, te_path=te_path, device=device, encoder=encoder,
                      path_config=path_config, output_dir=output_dir, output_name=output_name)
            if not cfg.get("target_tags"):
                return {"status": "skipped", "verify": None}
        apply_captions(cfg)
        held = cfg.get("verification_records") or {}
        if held:
            cfg["verify_prompts"] = list(dict.fromkeys(", ".join(split_tags(v))
                                                        for v in held.values() if str(v or "").strip()))
            if cfg.get("caption_source") is not None:
                cfg["caption_source"]["verify_source"] = "Fizgig image-validation hold-out"
                cfg["caption_source"]["verify_files"] = list(held)
                cfg["caption_source"]["verify_count"] = len(cfg["verify_prompts"])
        check_config(cfg)
        if not dit_path or not os.path.isfile(dit_path):
            raise ValueError(f"the Krea 2 DiT {dit_path!r} doesn't exist.")
        prompts = prompt_list(cfg)
        render_conds = None
        if cfg["render"]:
            from fizgig.families.description import GENERAL_NEGATIVE
            extra_prompts = render_prompts(cfg) + [GENERAL_NEGATIVE]
        if cfg["render"] and encoder is None:
            from fizgig.krea2.driver import Krea2Driver
            drv = Krea2Driver()
            te = drv.load_text_encoder(te_path, device)
            try:
                encode_prompts(prompts, te_path, device, driver=drv, te=te, store=store)
                render_conds = drv.encode_text(te, extra_prompts)
            finally:
                drv.unload_text_encoder(te)
        else:
            encode_prompts(prompts, te_path, device, encoder=encoder, store=store)
        path = load_fusion_path(dit_path, path_config, device=device, dtype=DTYPES[cfg["compute_dtype"]])
        result = run_on_path(path, store, cfg)
        del path
        files = export(result, cfg, output_dir, output_name)
        if cfg["render"]:
            try:
                if not vae_path:
                    raise ValueError("no VAE given")
                files["renders"] = render(dit_path, vae_path, render_conds[:-1], render_conds[-1], result["edit"],
                                          output_dir, output_name, seed=int(render_seed), width=render_width,
                                          height=render_height, device=device)
            except Exception:
                pass
        return {**files, "verify": result["verify"]}
    finally:
        store.close()


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Krea 2 text-fusion calibration (closed-form LoRA edit).")
    p.add_argument("--dit", required=True)
    p.add_argument("--text_encoder", required=True)
    p.add_argument("--vae", default=None, help="needed only with --render")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_name", required=True)
    p.add_argument("--config", default=None, help="a JSON file of settings")
    p.add_argument("--caption_folder", default=None)
    p.add_argument("--target_tags", default=None)
    p.add_argument("--control_tags", default=None)
    p.add_argument("--auto_controls", action="store_true", help="top listed controls up from the captions")
    p.add_argument("--caption_count", type=int, default=None)
    p.add_argument("--strength", type=float, default=None,
                   help="positive amplifies the target-vs-control direction (v2); negative removes it")
    p.add_argument("--rounds", type=int, default=None, help="edit rounds with fresh captions/controls each")
    p.add_argument("--verify_count", type=int, default=None, help="captions held out of the fit to verify on")
    p.add_argument("--column_select", default=None, choices=["activation", "weight"])
    p.add_argument("--ram_budget_gb", type=float, default=None)
    p.add_argument("--scan", action="store_true", help="pick targets and controls by scanning the caption folder")
    p.add_argument("--columns", type=int, default=None)
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--num_directions", type=int, default=None)
    p.add_argument("--edit_blocks", default=None, help="comma list, e.g. txt_rf_0,txt_rf_1")
    p.add_argument("--render", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    opts = {f"{_OPTION_PREFIX}{k}": v for k, v in (
        ("caption_folder", a.caption_folder), ("target_tags", a.target_tags), ("control_tags", a.control_tags),
        ("caption_count", a.caption_count), ("strength", a.strength), ("render", "1" if a.render else None),
        ("auto_controls", "1" if a.auto_controls else None), ("scan", "1" if a.scan else None)) if v is not None}
    cfg = make_config(opts, config_path=a.config)
    for k in ("columns", "rank", "num_directions", "edit_blocks", "rounds", "verify_count", "column_select",
              "ram_budget_gb"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    cfg = _normalise(cfg)
    out = run_direction_edit(cfg, dit_path=a.dit, te_path=a.text_encoder, vae_path=a.vae, output_dir=a.output_dir,
                             output_name=a.output_name, render_seed=a.seed, device=a.device)


if __name__ == "__main__":
    main()
