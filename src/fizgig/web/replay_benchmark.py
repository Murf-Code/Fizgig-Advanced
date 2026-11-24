"""Fixed Krea 2 A/B controls for the stream-sync test branch (real training, not a microbenchmark)."""
import os

CASES = (
    ("replay", "Global replay", False),
    ("selective", "Global replay", True),
    ("standard", "Standard", False),
)


def settings(group, output_root, epochs=3, prompt=""):
    epochs = int(epochs)
    if not 1 <= epochs <= 30:
        raise ValueError("Replay comparison epochs must be between 1 and 30.")
    prompt = str(prompt or "").strip()
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
        "SAMPLE_ENABLED": bool(prompt), "SAMPLE_PROMPT": prompt, "SAMPLE_SEED": "1234",
        "SAMPLE_EVERY_N_EPOCHS": "1", "SAMPLE_AT_FIRST": True,
        "SAMPLE_WIDTH": "512", "SAMPLE_HEIGHT": "512", "SAMPLE_STEPS": "20",
    }
    return [{**common, "__replay_compare": case, "LORA_NAME": f"{group}-{case}",
             "LORA_OUTPUT_DIR": os.path.join(output_root, group, case),
             "KREA2_TRAINING_METHOD": method, "KREA2_SELECTIVE_BLOCK_CHECK": selective}
            for case, method, selective in CASES]


def adjust_plan(desc, values, plan):
    """Validate the fixed comparison; keep legacy queued matched controls runnable."""
    case = values.get("__replay_compare")
    if not case:
        return plan
    if desc.key != "krea2" or case not in ({c[0] for c in CASES} | {"standard-matched"}):
        raise ValueError("Invalid Krea 2 replay comparison case.")
    if plan.problems:
        return plan
    stages = [stage for stage in plan.stages if stage.name == "Training"]
    if len(stages) != 1:
        plan.problems.append("Replay comparison needs exactly one normal Training stage.")
        return plan
    if case == "standard-matched":
        cmd = stages[0].cmd
        if "--train_blocks" in cmd:
            plan.problems.append("Matched Standard already has a custom block selection.")
            return plan
        cmd.extend(["--train_blocks", ",".join(f"block_{i}" for i in range(desc.n_blocks))])
    if "--holdout_file" not in stages[0].cmd and "holdout_file =" not in plan.dataset_toml:
        plan.problems.append("Replay comparison needs a validation hold-out. Choose a larger image dataset.")
    plan.console.append(f"[replay comparison] {case}: fixed seed, rank, optimizer, resolution and validation.")
    return plan
