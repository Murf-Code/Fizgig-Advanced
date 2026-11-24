"""Fixed Krea 2 A/B controls for the stream-sync test branch (real training, not a microbenchmark)."""
import os
from pathlib import Path

STYLE_PROMPT = "A small wooden boat on a quiet lake surrounded by autumn trees, distant mountains and soft morning light"
DEFAULT_PREVIEW_PROMPTS = "A character portrait, three-quarter view, expressive face\n" + STYLE_PROMPT

CASES = (
    ("replay", "Global replay", False),
    ("downhill", "Global replay", True),
    ("uphill", "Global replay", True),
    ("high_gradient", "Global replay", True),
    ("low_gradient", "Global replay", True),
    ("random", "Global replay", True),
    ("rotational", "Global replay", True),
    ("standard", "Standard", False),
)
CASES = CASES + tuple((case + "-lr450", method, selective)
                      for case, method, selective in CASES if selective)


def preview_prompts(folder, supplied="", trigger=""):
    """Resolve the same prompt set once, before queuing any of the fourteen cases."""
    supplied = str(supplied or "").strip()
    if supplied:
        return supplied
    from fizgig.dataset.caption_file import read_caption
    likeness = ""
    for image in sorted(Path(folder).iterdir(), key=lambda p: (p.name.casefold(), p.name)):
        if not image.is_file() or image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            continue
        try:
            caption = " ".join(read_caption(str(image.with_suffix(".txt"))).split())
        except OSError:
            continue
        if caption and not caption.startswith("#"):
            likeness = caption[:3000]
            break
    trigger = " ".join(str(trigger or "").strip(" ,").split())[:200]
    prefix = trigger + ", " if trigger else ""
    return (likeness or prefix + "A character portrait, three-quarter view, expressive face") + "\n" + prefix + STYLE_PROMPT


def settings(group, output_root, epochs=3, prompt=""):
    epochs = int(epochs)
    if not 1 <= epochs <= 30:
        raise ValueError("Replay comparison epochs must be between 1 and 30.")
    prompt = str(prompt or "").strip() or DEFAULT_PREVIEW_PROMPTS
    if len(prompt) > 4000:
        raise ValueError("Preview prompts must be at most 4000 characters.")
    common = {
        "FAMILY_KIND": "standard", "NETWORK_TYPE": "LoRA (standard)",
        "NETWORK_DIM": "32", "NETWORK_ALPHA": "32", "LEARNING_RATE": "0.00025",
        "OPTIMIZER_TYPE": "muon", "GRADIENT_ACCUMULATION": "1", "BATCH_SIZE": "1",
        "MAX_TRAIN_EPOCHS": str(epochs), "SAVE_EVERY_N_EPOCHS": "1",
        "SEED": "42", "FAMILY_PRECISION": "INT8 (8-bit, fastest)", "BLOCKS_SWAP": "22",
        "DATASET_MEGAPIXELS": "0.25", "COMPILE_BLOCKS": "Off",
        "VALIDATION_LOSS": True, "VALIDATION_HOLDOUT": True, "VALIDATION_IMAGES": "4",
        "VALIDATION_GROUPS": "1", "VALIDATION_TIMESTEPS": "4", "VALIDATION_SEED": "42",
        "VALIDATION_EARLY_STOP": False, "ADAPTIVE_LR": False, "KREA2_LOSS_WATCH": False,
        "KREA2_PER_IMAGE_LR": False, "KREA2_AUTO_RECAPTION": False,
        "SAMPLE_ENABLED": True, "SAMPLE_PROMPT": prompt, "SAMPLE_SEED": "1234",
        "SAMPLE_EVERY_N_EPOCHS": "1", "SAMPLE_AT_FIRST": True,
        "SAMPLE_WIDTH": "512", "SAMPLE_HEIGHT": "512", "SAMPLE_STEPS": "20",
    }
    return [{**common, "LEARNING_RATE": "0.00045" if case.endswith("-lr450") else "0.00025",
             "__replay_compare": case, "LORA_NAME": f"{group}-{case}",
             "LORA_OUTPUT_DIR": os.path.join(output_root, group, case),
             "KREA2_TRAINING_METHOD": method, "KREA2_SELECTIVE_BLOCK_CHECK": selective}
            for case, method, selective in CASES]


def adjust_plan(desc, values, plan):
    """Validate the fixed comparison; keep legacy queued matched controls runnable."""
    case = values.get("__replay_compare")
    if not case:
        return plan
    if desc.key != "krea2" or case not in ({c[0] for c in CASES} | {"standard-matched", "selective"}):
        raise ValueError("Invalid Krea 2 replay comparison case.")
    if plan.problems:
        return plan
    stages = [stage for stage in plan.stages if stage.name == "Training"]
    if len(stages) != 1:
        plan.problems.append("Replay comparison needs exactly one normal Training stage.")
        return plan
    from fizgig.krea2.selective_policy import STRATEGIES
    strategy = case.removesuffix("-lr450")
    if strategy in STRATEGIES:
        stages[0].cmd.extend(["--family_option", f"selective_strategy={strategy}",
                              "--family_option", "selective_budget=14"])
    if case == "standard-matched":
        cmd = stages[0].cmd
        if "--train_blocks" in cmd:
            plan.problems.append("Matched Standard already has a custom block selection.")
            return plan
        cmd.extend(["--train_blocks", ",".join(f"block_{i}" for i in range(desc.n_blocks))])
    if "--holdout_file" not in stages[0].cmd and "holdout_file =" not in plan.dataset_toml:
        plan.problems.append("Replay comparison needs a validation hold-out. Choose a larger image dataset.")
    if values.get("SAMPLE_ENABLED") and "--sample_prompts" not in stages[0].cmd:
        plan.problems.append("Replay comparison requires a usable sample prompt.")
    plan.console.append(f"[replay comparison] {case}: fixed seed, rank, optimizer, resolution and validation.")
    return plan
