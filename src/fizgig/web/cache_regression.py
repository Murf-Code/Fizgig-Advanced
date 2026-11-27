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


def settings(group, root, prompt):
    from fizgig.web.polora_diagnostic import settings as old_settings
    base = old_settings(group, root)[0]
    base.pop("__polora_diagnostic")
    base.update(MAX_TRAIN_EPOCHS="2", VALIDATION_LOSS=True, VALIDATION_HOLDOUT=False,
                VALIDATION_GROUPS="1", VALIDATION_IMAGES="4", VALIDATION_SEED="42",
                VALIDATION_TIMESTEPS="4", ADAPTIVE_LR=False, KREA2_PER_IMAGE_LR=False,
                KREA2_LOSS_WATCH=False, KREA2_DIRECTION_EDIT=False, SAMPLE_ENABLED=False)
    return [dict(base, __cache_regression=case, __cache_prompt=prompt,
                 LORA_NAME=f"{group}-{case}", LORA_OUTPUT_DIR=os.path.join(root, group, case))
            for case in CASES]


def source_config(body):
    import toml
    cfg = toml.loads(body)
    sets = cfg.get("datasets") or []
    if len(sets) != 1 or not sets[0].get("image_directory"):
        raise ValueError("Cache regression requires one ordinary image dataset.")
    ds = sets[0]
    if ds.get("control_directory") or cfg.get("general", {}).get("holdout_file"):
        raise ValueError("Cache regression uses one non-held-out, unpaired dataset.")
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
    train[0].cmd.extend(["--family_option", "polora_diagnostic=1",
                         "--family_option", "cache_regression=1",
                         "--sample_prompts", prompt_path, "--sample_every_n_epochs", "1",
                         "--sample_at_first", "--sample_seed", "42", "--sample_width", "768",
                         "--sample_height", "768", "--sample_steps", "24", "--sample_cfg_scale", "3.5",
                         "--vae", lat[lat.index("--model") + 1],
                         "--text_encoder", txt[txt.index("--model") + 1]])
    plan.stages = stages + train
    plan.dirs.append(str(dest))
    plan.console.append(f"[cache-regression] {case}: isolated cache, fixed validation panel (not held out), two epochs.")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare_caches(source_body, configs, arch):
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
        lat = source / f"{stem}_{w:04d}x{h:04d}_{arch}.safetensors"
        te = source / f"{stem}_{arch}_te.safetensors"
        var = Path(variants_path(str(te)))
        for file in (lat, te):
            if not file.is_file():
                raise ValueError(f"Existing cache missing: {file}. Use the complete no-holdout cache from the failing run.")
        if ImageDataset.latent_cache_matches_reso(str(lat), selector.get_bucket_resolution((w,h)), arch) is not True:
            raise ValueError(f"Existing latent has the wrong bucket: {lat}. Match the failing run's 1 MP cache.")
        expected.extend([(lat, "latents"), (te, "text")])
        caption = read_caption(str(Path(p).with_suffix(ext))).strip()
        if variant_captions(caption, tags) or tags.caption_dropout:
            if not var.is_file():
                raise ValueError(f"Existing caption variants missing: {var}. Do not rebuild the source before the test.")
            expected.append((var, "text"))
        elif var.is_file():
            expected.append((var, "text"))
    control = Path(configs[0]["LORA_OUTPUT_DIR"]) / "isolated-cache"
    control.mkdir(parents=True, exist_ok=False)
    records = []
    for file, axis in expected:
        before = file.stat()
        dst = control / file.name
        shutil.copy2(file, dst)
        after = file.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Source cache changed during snapshot: {file}")
        records.append({"file": file.name, "axis": axis, "sha256": digest(dst), "bytes": dst.stat().st_size})
    for values in configs[1:]:
        fresh_lat, fresh_text = CASES[values["__cache_regression"]]
        dest = Path(values["LORA_OUTPUT_DIR"]) / "isolated-cache"
        dest.mkdir(parents=True, exist_ok=False)
        for record in records:
            if (record["axis"] == "latents" and not fresh_lat) or (record["axis"] == "text" and not fresh_text):
                shutil.copy2(control / record["file"], dest / record["file"])
    return {"source": str(source), "images": paths, "files": records}


def verify_start(plan, values):
    """Reject changed recipes, models/paths, or previously used arm caches."""
    case = values["__cache_regression"]
    out = Path(values["LORA_OUTPUT_DIR"])
    with open(out.parent / "diagnostic-settings.json", encoding="utf-8") as f:
        manifest = json.load(f)
    recipe = next(p for p in manifest["plans"] if p["case"] == case)
    actual = [{"name": s.name, "cmd": s.cmd} for s in plan.stages]
    if plan.dataset_toml != recipe["dataset_toml"] or actual != recipe["stages"]:
        raise ValueError("Diagnostic plan changed after snapshot; start a new comparison.")
    if any(out.glob("*.safetensors")) or (out / "diagnostic.log").exists():
        raise ValueError("This cache regression arm has already run; create a new comparison.")
    fresh_lat, fresh_text = CASES[case]
    expected = {r["file"]: r for r in manifest["snapshot"]["files"]
                if (r["axis"] == "latents" and not fresh_lat) or (r["axis"] == "text" and not fresh_text)}
    cache = out / "isolated-cache"
    actual_files = {p.name for p in cache.iterdir()}
    if actual_files != set(expected):
        raise ValueError("Isolated cache contents changed before start; create a new comparison.")
    for name, record in expected.items():
        if digest(cache / name) != record["sha256"]:
            raise ValueError(f"Isolated cache bytes changed before start: {name}.")
