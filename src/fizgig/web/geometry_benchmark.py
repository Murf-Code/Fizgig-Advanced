"""Matrix geometry pilot: reuse fixed-run queue controls from test/global-replay-comparison.

The rates are starting points from existing Fizgig recipes, not calibrated optima.
This is a one-seed feasibility pilot, not a definitive optimizer ranking.
"""
import os

DEFAULT_PREVIEW_PROMPTS = "\n".join((
    "Ferwildir, anthropomorphic red fox, female, full body, wearing a green traveling cloak, standing on a stone bridge above a forest stream, expressive eyes, detailed fur",
    "Ferwildir, solo, anthro, male wolf, white fur, blue and purple fur accents, spiky blue hair, blue eyes, full body, running along a beach, looking over shoulder, joyful expression",
    "solo, anthropomorphic male wolf, white fur, blue and purple fur accents, spiky blue hair, blue eyes, sitting in a cafe, realistic watercolor illustration, soft washes and natural lighting",
    "A small wooden boat on a quiet lake surrounded by autumn trees, distant mountains and soft morning light",
))

EPOCHS = 5
CASES = (
    ("adamw8bit", "adamw8bit", "0.0002", "weight_decay=0"),
    ("batched-muon", "muon", "0.0002", "adjust_lr_fn=match_rms_adamw weight_decay=0 momentum=0.95 nesterov=True ns_steps=5"),
    ("polora-budgeted", "polora", "0.02", "magnitude_scale=3 compile=False chunk_memory_mib=128"),
    ("tsd-balanced", "lora_tsd", "0.005", "magnitude_scale=3 balance=True max_delta_norm=0.1"),
    ("tsd-unbalanced", "lora_tsd", "0.005", "magnitude_scale=3 balance=False max_delta_norm=0.1"),
)


def settings(group, output_root):
    common = {
        "FAMILY_KIND": "standard", "NETWORK_TYPE": "LoRA (standard)",
        "NETWORK_DIM": "32", "NETWORK_ALPHA": "32",
        "GRADIENT_ACCUMULATION": "1", "BATCH_SIZE": "1",
        "MAX_TRAIN_EPOCHS": str(EPOCHS), "SAVE_EVERY_N_EPOCHS": "1",
        "SEED": "42", "FAMILY_PRECISION": "INT8 (8-bit, fastest)", "BLOCKS_SWAP": "22",
        "DATASET_MEGAPIXELS": "0.25", "COMPILE_BLOCKS": "Off",
        "FAMILY_EMA": "Off", "LR_SCHEDULER": "constant", "LR_WARMUP_STEPS": "0",
        "MAX_GRAD_NORM": "1.0", "CONTEXT_LORA_PATH": "", "FAMILY_DOP": False,
        "KREA2_TRAINING_METHOD": "Standard", "KREA2_SELECTIVE_BLOCK_CHECK": False,
        "KREA2_DIRECTION_EDIT": False, "KREA2_WARMUP_LOOK": False,
        "VALIDATION_LOSS": True, "VALIDATION_HOLDOUT": True, "VALIDATION_IMAGES": "4",
        "VALIDATION_GROUPS": "1", "VALIDATION_TIMESTEPS": "4", "VALIDATION_SEED": "42",
        "VALIDATION_EARLY_STOP": False, "ADAPTIVE_LR": False, "KREA2_LOSS_WATCH": False,
        "KREA2_PER_IMAGE_LR": False, "KREA2_AUTO_RECAPTION": False,
        "SAMPLE_ENABLED": True, "SAMPLE_PROMPT": DEFAULT_PREVIEW_PROMPTS, "SAMPLE_SEED": "1234",
        "SAMPLE_EVERY_N_EPOCHS": "1", "SAMPLE_AT_FIRST": True,
        "SAMPLE_WIDTH": "512", "SAMPLE_HEIGHT": "512", "SAMPLE_STEPS": "20",
        "POLORA_COMPILE": False, "BEST_CHECKPOINT_MODE": "Off",
        "TAG_SHUFFLE": False, "TAG_DROPOUT": "0", "CAPTION_DROPOUT": "Off",
        "RESUME_TRAINING": "", "MASKED_TRAINING": False, "FAMILY_REG_ON": False,
    }
    return [{**common, "__geometry_compare": case, "OPTIMIZER_TYPE": opt,
             "OPTIMIZER_ARGS": args, "LEARNING_RATE": lr,
             "LORA_NAME": f"{group}-{case}", "LORA_OUTPUT_DIR": os.path.join(output_root, group, case)}
            for case, opt, lr, args in CASES]


def adjust_plan(desc, values, plan):
    case = values.get("__geometry_compare")
    if not case:
        return plan
    if desc.key != "krea2" or case not in {x[0] for x in CASES}:
        raise ValueError("Invalid Krea 2 matrix geometry comparison case.")
    if plan.problems:
        return plan
    stages = [stage for stage in plan.stages if stage.name == "Training"]
    if len(stages) != 1:
        plan.problems.append("Matrix geometry comparison needs exactly one normal Training stage.")
        return plan
    cmd = stages[0].cmd
    if "--holdout_file" not in cmd and "holdout_file =" not in plan.dataset_toml:
        plan.problems.append("Matrix geometry comparison requires a validation holdout.")
    if "--sample_prompts" not in cmd:
        plan.problems.append("Matrix geometry comparison requires usable sample prompts.")
    if "--train_blocks" in cmd:
        plan.problems.append("Matrix geometry comparison requires unrestricted eligible blocks.")
    plan.console.append(f"[matrix geometry pilot] {case}: fixed rank, data, seed, validation; provisional LR.")
    return plan
