"""Four-arm cache isolation diagnostic; never modifies a source cache."""
import hashlib
import json
import os
import shutil
from pathlib import Path

CASES = {
    "a-existing": (False, False),
    "b-fresh-text": (False, True),
    "c-fresh-latents": (True, False),
    "d-fresh-both": (True, True),
}


# Explicit profiles, not hardware detection: the chosen recipe stays fixed across all arms.
GPU_PROFILES = {
    "3090": {"swap": 0, "compile": "outside"},
    "4070-ti": {"swap": 24, "compile": "Off"},
}


# Only cache-related fields may cross from the user's normal training form.
# Do not inherit optimizer, replay, LR, precision, swap, seed or sample settings.
DATASET_KEYS = (
    "DATASET_MEGAPIXELS", "ENABLE_BUCKET", "BUCKET_NO_UPSCALE",
    "DATASET_CAPTION_EXT", "TAG_SHUFFLE", "TAG_KEEP_FIRST", "TAG_DROPOUT",
    "TAG_VARIANTS", "TAG_VARIANT_CAP", "CAPTION_DROPOUT",
)


def settings(group, root, prompt, *, gpu_profile, masked=False, dataset_values=None):
    if gpu_profile not in GPU_PROFILES:
        raise ValueError("Choose the RTX 3090 or RTX 4070 Ti cache test.")
    profile = GPU_PROFILES[gpu_profile]
    swap = profile["swap"]
    base = {
        "FAMILY_KIND": "standard", "NETWORK_TYPE": "LoRA (standard)",
        "NETWORK_DIM": "8", "NETWORK_ALPHA": "8", "GRADIENT_ACCUMULATION": "1",
        "BATCH_SIZE": "1", "MAX_GRAD_NORM": "1.0", "MAX_TRAIN_EPOCHS": "1",
        "SAVE_EVERY_N_EPOCHS": "1", "SAVE_STATE": False, "SAVE_STATE_ON_TRAIN_END": False,
        "SEED": "42", "DATASET_MEGAPIXELS": "1.0", "ENABLE_BUCKET": True,
        "BUCKET_NO_UPSCALE": True, "DATASET_CAPTION_EXT": ".txt", "ENABLE_CACHE": True,
        "TAG_SHUFFLE": True, "TAG_KEEP_FIRST": "1", "TAG_VARIANTS": "4",
        "MASK_FLOOR": "0.1 (recommended)", "MASK_PRIOR": "Off", "FAMILY_EMA": "0.98 (recommended)",
        "VALIDATION_LOSS": False, "VALIDATION_EARLY_STOP": False,
        "SAMPLE_ENABLED": False, "SAMPLE_AT_FIRST": False, "TRAINING_MAPS": False,
        "KREA2_AUTO_RECAPTION": False, "KREA2_LOSS_WATCH": True,
        "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "0.01", "ADAPTIVE_LR_MAX": "0.04",
        "KREA2_PER_IMAGE_LR": True, "KREA2_TRAINING_METHOD": "Standard",
        "KREA2_SELECTIVE_BLOCK_CHECK": False, "INT8_ROTATED_BACKWARD": False,
        "RESUME_TRAINING": "", "CONTEXT_LORA_PATH": "",
    }
    base.update(MAX_TRAIN_EPOCHS="2", VALIDATION_LOSS=True, VALIDATION_HOLDOUT=True,
                VALIDATION_GROUPS="1", VALIDATION_IMAGES="4", VALIDATION_SEED="42",
                VALIDATION_TIMESTEPS="4", ADAPTIVE_LR=False, KREA2_PER_IMAGE_LR=False,
                KREA2_LOSS_WATCH=False, KREA2_DIRECTION_EDIT=False, SAMPLE_ENABLED=False,
                FAMILY_TURBO_ON=False, FAMILY_TURBO_STRENGTH="0",
                COMPILE_BLOCKS=profile["compile"], POLORA_COMPILE=True, MASKED_TRAINING=bool(masked),
                OPTIMIZER_TYPE="polora", FAMILY_PRECISION="INT8 (8-bit, fastest)",
                BLOCKS_SWAP=str(swap), LEARNING_RATE="0.02",
                OPTIMIZER_ARGS="magnitude_scale=3 curvature_beta=0.99 compile=True")
    if gpu_profile == "4070-ti" and dataset_values is not None:
        base.update({key: dataset_values[key] for key in DATASET_KEYS if key in dataset_values})
    return [dict(base, __cache_regression=case, __cache_prompt=prompt, __cache_gpu_profile=gpu_profile,
                 LORA_NAME=f"{group}-{case}", LORA_OUTPUT_DIR=os.path.join(root, group, case))
            for case in CASES]


def source_config(body):
    import toml
    cfg = toml.loads(body)
    sets = cfg.get("datasets") or []
    if len(sets) != 1 or not sets[0].get("image_directory"):
        raise ValueError("Cache regression requires one ordinary image dataset.")
    ds = sets[0]
    if ds.get("control_directory"):
        raise ValueError("Cache regression uses an unpaired dataset.")
    return cfg, Path(ds.get("cache_directory") or ds["image_directory"]).resolve()


def isolated_config(body, dest):
    import toml
    cfg, _ = source_config(body)
    cfg["datasets"][0]["cache_directory"] = str(Path(dest).resolve())
    return toml.dumps(cfg)


def configure_plan(desc, inputs, plan, values, launch):
    """Replace normal incremental stages with explicit isolated-axis stages."""
    case = values["__cache_regression"]
    if desc.key != "krea2" or case not in CASES:
        plan.problems.append("Invalid Krea 2 cache regression case.")
        return
    if plan.problems:
        return
    train = [s for s in plan.stages if s.name == "Training"]
    if len(train) != 1 or any(s.name not in ("Training", "Cache Preparation", "Text Encoder Caching") for s in plan.stages):
        plan.problems.append("Cache regression requires a single standard training stage.")
        return
    try:
        dest = Path(inputs["LORA_OUTPUT_DIR"]) / "isolated-cache"
        body = isolated_config(plan.dataset_toml, dest)
    except ValueError as exc:
        plan.problems.append(str(exc))
        return
    config = inputs["DATASET_CONFIG"]
    plan.dataset_toml = body
    plan.files = [(p, body if p == config else text) for p, text in plan.files]
    # Rebuild commands even if the source preflight omitted a stage.
    lat = launch.cache_command(desc, inputs, "latents")
    txt = launch.cache_command(desc, inputs, "text")
    for cmd in (lat, txt):
        cmd[:] = [v for v in cmd if v != "--skip_existing"]
    stages = []
    fresh_lat, fresh_text = CASES[case]
    if fresh_lat:
        stages.append(launch.Stage("Cache Preparation", lat))
    if fresh_text:
        stages.append(launch.Stage("Text Encoder Caching", txt))
    prompt = str(values.get("__cache_prompt") or "").strip()
    if not prompt or len(prompt) > 2000 or "\n" in prompt or "\r" in prompt:
        plan.problems.append("Provide one fixed sample prompt (up to 2000 characters).")
        return
    prompt_path = str(Path(inputs["LORA_OUTPUT_DIR"]) / "regression-prompt.txt")
    plan.files.append((prompt_path, prompt + "\n"))
    train[0].cmd.extend(["--family_option", "cache_regression=1",
                         "--family_option", f"cache_regression_dir={inputs['LORA_OUTPUT_DIR']}",
                         "--sample_prompts", prompt_path, "--sample_every_n_epochs", "1",
                         "--sample_at_first", "--sample_seed", "42", "--sample_width", "768",
                         "--sample_height", "768", "--sample_steps", "24", "--sample_cfg_scale", "3.5",
                         "--vae", lat[lat.index("--model") + 1],
                         "--text_encoder", txt[txt.index("--model") + 1]])
    plan.stages = stages + train
    plan.dirs.append(str(dest))
    plan.console.append(f"[cache-regression] {case}: isolated cache, fixed validation hold-out, two epochs.")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare_caches(source_body, configs, arch, held_paths=()):
    """Snapshot only the expected item files into A, then copy retained axes from A.
    No links, no writes to the source, no silently repairing a missing retained axis.
    """
    from fizgig.dataset.image_dataset import glob_images, BucketSelector, ImageDataset
    from fizgig.families.cache_preflight import _header
    from fizgig.dataset.caption_file import read_caption
    from fizgig.dataset.tag_variants import TagSettings, variant_captions, variants_path
    cfg, source = source_config(source_body)
    ds, g = cfg["datasets"][0], cfg["general"]
    ext = g.get("caption_extension", ".txt")
    paths = glob_images(ds["image_directory"], caption_extension=ext)
    held = {Path(p).resolve() for p in held_paths}
    if not held.issubset({Path(p).resolve() for p in paths}):
        raise ValueError("Hold-out contains pictures outside the diagnostic dataset.")
    if not paths:
        raise ValueError("No captioned images found.")
    stems = [Path(p).stem for p in paths]
    if len(set(stems)) != len(stems):
        raise ValueError("Duplicate image stems share cache names; use an unambiguous dataset.")
    selector = BucketSelector(tuple(g["resolution"]), g.get("enable_bucket", True),
                              g.get("bucket_no_upscale", True), 16)
    tags = TagSettings.from_params(shuffle=g.get("tag_shuffle", False),
        keep_first=g.get("keep_first_tags", 1), tag_dropout=g.get("tag_dropout", 0),
        caption_dropout=g.get("caption_dropout", 0), variants=g.get("caption_variants", 4),
        keep_words=ds.get("keep_words") or g.get("keep_words"), cap=g.get("caption_variant_cap", 4))
    expected = []
    for p in paths:
        stem = Path(p).stem
        w, h, _ = _header(p)
        is_held = Path(p).resolve() in held
        rel = Path("_fizgig_holdout") if is_held else Path()
        latent_name = f"{stem}_{w:04d}x{h:04d}_{arch}.safetensors"
        text_name = f"{stem}_{arch}_te.safetensors"
        # Use the original validation cache if present, otherwise an existing
        # ordinary cache; never invent missing training variants.
        src = source / rel
        lat = src / latent_name
        te = src / text_name
        if is_held and not (lat.is_file() and te.is_file()):
            lat, te = source / latent_name, source / text_name
        var = Path(variants_path(str(te)))
        for file in (lat, te):
            if not file.is_file():
                raise ValueError(f"Existing cache missing: {file}. Use the complete cache from the failing run with the same validation selection.")
        if ImageDataset.latent_cache_matches_reso(str(lat), selector.get_bucket_resolution((w,h)), arch) is not True:
            raise ValueError(f"Existing latent has the wrong bucket: {lat}. "
                             f"Expected {selector.get_bucket_resolution((w, h))} pixels at target {g['resolution']}. "
                             "For RTX 4070 Ti, set Target Megapixels and bucket options to those used to create "
                             "the existing cache. RTX 3090 uses the original fixed 1 MP recipe. "
                             "Do not regenerate the source cache before testing.")
        expected.extend([(lat, "latents", rel / lat.name), (te, "text", rel / te.name)])
        if is_held:
            continue
        caption = read_caption(str(Path(p).with_suffix(ext))).strip()
        if variant_captions(caption, tags) or tags.caption_dropout:
            if not var.is_file():
                raise ValueError(f"Existing caption variants missing: {var}. Do not rebuild the source before the test.")
            expected.append((var, "text", rel / var.name))
        elif var.is_file():
            expected.append((var, "text", rel / var.name))
    control = Path(configs[0]["LORA_OUTPUT_DIR"]) / "isolated-cache"
    control.mkdir(parents=True, exist_ok=False)
    records = []
    for file, axis, relative in expected:
        before = file.stat()
        dst = control / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, dst)
        after = file.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Source cache changed during snapshot: {file}")
        records.append({"file": relative.as_posix(), "source": str(file), "axis": axis, "sha256": digest(dst), "bytes": dst.stat().st_size})
    for values in configs[1:]:
        fresh_lat, fresh_text = CASES[values["__cache_regression"]]
        dest = Path(values["LORA_OUTPUT_DIR"]) / "isolated-cache"
        dest.mkdir(parents=True, exist_ok=False)
        for record in records:
            if (record["axis"] == "latents" and not fresh_lat) or (record["axis"] == "text" and not fresh_text):
                dst = dest / record["file"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(control / record["file"], dst)
    return {"source": str(source), "images": paths, "holdout": [str(p) for p in held_paths], "files": records}


def verify_start(plan, values):
    """Reject changed recipes, models/paths, or previously used arm caches."""
    case = values["__cache_regression"]
    out = Path(values["LORA_OUTPUT_DIR"])
    with open(out.parent / "diagnostic-settings.json", encoding="utf-8") as f:
        manifest = json.load(f)
    recipe = next(p for p in manifest["plans"] if p["case"] == case)
    actual = [{"name": s.name, "cmd": s.cmd} for s in plan.stages]
    if (plan.dataset_toml != recipe["dataset_toml"] or actual != recipe["stages"]
            or holdout_paths(plan) != manifest["snapshot"].get("holdout", [])):
        raise ValueError("Diagnostic plan changed after snapshot; start a new comparison.")
    if any(out.glob("*.safetensors")) or (out / "diagnostic.log").exists():
        raise ValueError("This cache regression arm has already run; create a new comparison.")
    fresh_lat, fresh_text = CASES[case]
    expected = {r["file"]: r for r in manifest["snapshot"]["files"]
                if (r["axis"] == "latents" and not fresh_lat) or (r["axis"] == "text" and not fresh_text)}
    cache = out / "isolated-cache"
    actual_files = {p.relative_to(cache).as_posix() for p in cache.rglob("*") if p.is_file()}
    if actual_files != set(expected):
        raise ValueError("Isolated cache contents changed before start; create a new comparison.")
    for name, record in expected.items():
        if digest(cache / name) != record["sha256"]:
            raise ValueError(f"Isolated cache bytes changed before start: {name}.")


def holdout_paths(plan):
    """Read this arm's planned hold-out list, not a previous run's disk state."""
    for path, body in plan.files:
        if str(path).endswith("_holdout.json"):
            return json.loads(body).get("paths") or []
    return []
