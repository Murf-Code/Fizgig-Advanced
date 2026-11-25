"""Fixed Krea 2 A/B controls for the stream-sync test branch (real training, not a microbenchmark)."""
import os

DEFAULT_PREVIEW_PROMPTS = "\n".join((
    "Ferwildir, anthropomorphic red fox, female, full body, wearing a green traveling cloak, standing on a stone bridge above a forest stream, expressive eyes, detailed fur",
    "Ferwildir, solo, anthro, male wolf, white fur, blue and purple fur accents, spiky blue hair, blue eyes, full body, running along a beach, looking over shoulder, joyful expression",
    "solo, anthropomorphic male wolf, white fur, blue and purple fur accents, spiky blue hair, blue eyes, sitting in a cafe, realistic watercolor illustration, soft washes and natural lighting",
    "A small wooden boat on a quiet lake surrounded by autumn trees, distant mountains and soft morning light",
))
EPOCHS = 5
CASES = (
    ("replay", "Global replay", False),
    ("high_gradient-lr450", "Global replay", True),
    ("random-lr450", "Global replay", True),
    ("balanced_random-lr450", "Global replay", True),
    ("static_first_half-lr450", "Global replay", True),
    ("rotational-lr450", "Global replay", True),
    ("hybrid_coverage-lr450", "Global replay", True),
    ("hybrid_coverage-lr350", "Global replay", True),
    ("walking_coverage-lr450", "Global replay", True),
)

def case_strategy(case):
    return case.removesuffix("-lr450").removesuffix("-lr350")


def settings(group, output_root, epochs=EPOCHS, prompt=""):
    # Fixed server-side controls prevent accidental UI overrides.
    epochs = EPOCHS
    prompt = DEFAULT_PREVIEW_PROMPTS
    common = {
        "FAMILY_KIND": "standard", "NETWORK_TYPE": "LoRA (standard)",
        "NETWORK_DIM": "32", "NETWORK_ALPHA": "32", "LEARNING_RATE": "0.00025",
        "OPTIMIZER_TYPE": "muon", "GRADIENT_ACCUMULATION": "1", "BATCH_SIZE": "1",
        "MAX_TRAIN_EPOCHS": str(epochs), "SAVE_EVERY_N_EPOCHS": "1",
        "SEED": "43", "FAMILY_PRECISION": "INT8 (8-bit, fastest)", "BLOCKS_SWAP": "22",
        "DATASET_MEGAPIXELS": "0.25", "COMPILE_BLOCKS": "Off",
        "VALIDATION_LOSS": True, "VALIDATION_HOLDOUT": True, "VALIDATION_IMAGES": "4",
        "VALIDATION_GROUPS": "1", "VALIDATION_TIMESTEPS": "4", "VALIDATION_SEED": "42",
        "VALIDATION_EARLY_STOP": False, "ADAPTIVE_LR": False, "KREA2_LOSS_WATCH": False,
        "KREA2_PER_IMAGE_LR": False, "KREA2_AUTO_RECAPTION": False,
        "SAMPLE_ENABLED": True, "SAMPLE_PROMPT": prompt, "SAMPLE_SEED": "1234",
        "SAMPLE_EVERY_N_EPOCHS": "1", "SAMPLE_AT_FIRST": True,
        "SAMPLE_WIDTH": "512", "SAMPLE_HEIGHT": "512", "SAMPLE_STEPS": "20",
    }
    return [{**common, "LEARNING_RATE": ("0.00045" if case.endswith("-lr450") else
                                         "0.00035" if case.endswith("-lr350") else "0.00025"),
             "__replay_compare": case, "LORA_NAME": f"{group}-{case}",
             "LORA_OUTPUT_DIR": os.path.join(output_root, group, case),
             "KREA2_TRAINING_METHOD": method, "KREA2_SELECTIVE_BLOCK_CHECK": selective}
            for case, method, selective in CASES]


def adjust_plan(desc, values, plan):
    """Validate the fixed comparison; keep legacy queued matched controls runnable."""
    case = values.get("__replay_compare")
    if not case:
        return plan
    from fizgig.krea2.selective_policy import STRATEGIES
    legacy = {"standard-matched", "selective", "replay", "standard"} | set(STRATEGIES)
    if desc.key != "krea2" or case_strategy(case) not in legacy:
        raise ValueError("Invalid Krea 2 replay comparison case.")
    if plan.problems:
        return plan
    stages = [stage for stage in plan.stages if stage.name == "Training"]
    if len(stages) != 1:
        plan.problems.append("Replay comparison needs exactly one normal Training stage.")
        return plan
    strategy = case_strategy(case)
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
