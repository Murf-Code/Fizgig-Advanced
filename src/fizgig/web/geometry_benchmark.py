"""Matrix geometry pilot: reuse fixed-run queue controls from test/global-replay-comparison.

The rates are starting points from existing Fizgig recipes, not calibrated optima.
This is a one-seed feasibility pilot, not a definitive optimizer ranking.
"""
import os
from fizgig.web import memory_benchmark

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


FEASIBILITY_CASE = "polora-r64-feasibility"
FEASIBILITY_STEPS = 50


def feasibility_settings(group, output_root):
    v = next(v for v in settings(group, output_root)
             if v["__geometry_compare"] == "polora-budgeted")
    return [{**v, "__geometry_compare": FEASIBILITY_CASE,
             "LORA_NAME": f"{group}-{FEASIBILITY_CASE}",
             "LORA_OUTPUT_DIR": os.path.join(output_root, group, FEASIBILITY_CASE),
             "NETWORK_DIM": "64", "NETWORK_ALPHA": "64",
             "MAX_TRAIN_EPOCHS": "1", "SAMPLE_ENABLED": False,
             "SAVE_STATE": False, "SAVE_STATE_ON_TRAIN_END": False}]


# Group equal resolutions together so the second rank can reuse the latent cache.
# Repeat the successful r64 probe with the unchunked control last (OOM is useful data).
RESOLUTION_CASES = tuple(
    (f"polora-r{rank}-mp{mp.replace('.', 'p')}-fit", rank, mp, 128)
    for mp in ("0.25", "0.50", "0.75", "1.00") for rank in (16, 32)
) + (("polora-r64-mp0p25-chunked-fit", 64, "0.25", 128),
     ("polora-r64-mp0p25-unchunked-fit", 64, "0.25", 0))


# Isolate the state-view patch at identical chunk sizes. TSD uses the repaired
# implementation in this branch, not the unsafe historical version.
QUALITY_CASES = (
    ("quality-adamw8bit", "adamw8bit", "0.0002", "weight_decay=0"),
    ("quality-muon", "muon", "0.0002", "adjust_lr_fn=match_rms_adamw weight_decay=0 momentum=0.95 nesterov=True ns_steps=5"),
    ("quality-polora-reference", "polora", "0.02", "magnitude_scale=3 compile=False chunk_memory_mib=128 research_state_views=False"),
    ("quality-polora-views", "polora", "0.02", "magnitude_scale=3 compile=False chunk_memory_mib=128 research_state_views=True"),
    ("quality-tsd-balanced", "lora_tsd", "0.005", "magnitude_scale=3 balance=True max_delta_norm=0.1 ball_iters=3 log_every=1"),
    ("quality-tsd-unbalanced", "lora_tsd", "0.005", "magnitude_scale=3 balance=False max_delta_norm=0.1 ball_iters=3 log_every=1"),
)
TSD_FIT_CASES = tuple(("fit-" + case, opt, lr, args)
                      for case, opt, lr, args in QUALITY_CASES[-2:])


def is_feasibility(case):
    return (case == FEASIBILITY_CASE or case in {x[0] for x in RESOLUTION_CASES}
            or case in {x[0] for x in TSD_FIT_CASES} or case in memory_benchmark.CASE_MAP)


def optimizer_settings(group, output_root, short=False):
    common = settings(group, output_root)[0]
    cases = TSD_FIT_CASES if short else QUALITY_CASES
    return [{**common, "__geometry_compare": case, "OPTIMIZER_TYPE": opt,
             "OPTIMIZER_ARGS": args, "LEARNING_RATE": lr,
             "NETWORK_DIM": "16", "NETWORK_ALPHA": "16", "BLOCKS_SWAP": "18",
             "MAX_TRAIN_EPOCHS": "1" if short else "5",
             "SAMPLE_ENABLED": not short, "SAVE_STATE": False,
             "SAVE_STATE_ON_TRAIN_END": False,
             "LORA_NAME": f"{group}-{case}",
             "LORA_OUTPUT_DIR": os.path.join(output_root, group, case)}
            for case, opt, lr, args in cases]


# Round 10: matched unbalanced arms avoid the additional gauge changes of scalar balancing.
_R10_TSD = "magnitude_scale=3 balance=False max_delta_norm=0.1 ball_iters=1 log_every=1"
R10_CASES = (
    ("r10-muon-a", *QUALITY_CASES[1][1:]),
    ("r10-muon-a-repeat", *QUALITY_CASES[1][1:]),
    ("r10-tsd-fixed", "lora_tsd", "0.005", _R10_TSD),
    ("r10-tsd-reduced", "lora_tsd", "0.005", _R10_TSD + " tangent_lr_scale=0.1"),
    ("r10-tsd-decay", "lora_tsd", "0.005", _R10_TSD + " tangent_lr_scale=0.3 tangent_decay=rsqrt tangent_horizon=10"),
    ("r10-tsd-snr", "lora_tsd", "0.005", _R10_TSD + " tangent_lr_scale=2 budget=snr"),
    ("r10-tsd-snr-svd", "lora_tsd", "0.005", _R10_TSD + " tangent_lr_scale=2 budget=snr polar=svd"),
)


def round10_settings(group, output_root):
    common = optimizer_settings(group, output_root)[0]
    return [{**common, "__geometry_compare": case, "OPTIMIZER_TYPE": opt,
             "OPTIMIZER_ARGS": args, "LEARNING_RATE": lr,
             "LORA_NAME": f"{group}-{case}",
             "LORA_OUTPUT_DIR": os.path.join(output_root, group, case)}
            for case, opt, lr, args in R10_CASES]


# Matched real-training quality arms, followed by deliberate memory-pressure arms.
MUON_ME_CASES = (
    ("muon-me-reference", *QUALITY_CASES[1][1:], 22, 5),
    ("muon-me-reference-repeat", *QUALITY_CASES[1][1:], 22, 5),
    ("muon-me-shape-local", "muon_me", QUALITY_CASES[1][2], QUALITY_CASES[1][3] + " chunk_memory_mib=0", 22, 5),
    ("muon-me-chunk128", "muon_me", QUALITY_CASES[1][2], QUALITY_CASES[1][3] + " chunk_memory_mib=128", 22, 5),
    ("muon-me-swap16-chunk128", "muon_me", QUALITY_CASES[1][2], QUALITY_CASES[1][3] + " chunk_memory_mib=128", 16, 1),
    ("muon-me-swap16-reference", *QUALITY_CASES[1][1:], 16, 1),
)


def muon_me_settings(group, output_root):
    common = optimizer_settings(group, output_root)[0]
    return [{**common, "__geometry_compare": case, "OPTIMIZER_TYPE": opt,
             "OPTIMIZER_ARGS": args, "LEARNING_RATE": lr,
             "NETWORK_DIM": "64", "NETWORK_ALPHA": "64", "BLOCKS_SWAP": str(swap),
             "MAX_TRAIN_EPOCHS": str(epochs),
             "LORA_NAME": f"{group}-{case}",
             "LORA_OUTPUT_DIR": os.path.join(output_root, group, case)}
            for case, opt, lr, args, swap, epochs in MUON_ME_CASES]


def memory_settings(group, output_root, paired=False, async_test=False, fixed_test=False):
    common = next(v for v in muon_me_settings(group, output_root) if v['__geometry_compare'] == 'muon-me-chunk128')
    return memory_benchmark.settings(group, output_root, common, paired=paired, async_test=async_test, fixed_test=fixed_test)


def resolution_settings(group, output_root):
    v, = feasibility_settings(group, output_root)
    return [{**v, "__geometry_compare": case,
             "NETWORK_DIM": str(rank), "NETWORK_ALPHA": str(rank),
             "DATASET_MEGAPIXELS": mp,
             "OPTIMIZER_ARGS": f"magnitude_scale=3 compile=False chunk_memory_mib={budget}",
             "LORA_NAME": f"{group}-{case}",
             "LORA_OUTPUT_DIR": os.path.join(output_root, group, case)}
            for case, rank, mp, budget in RESOLUTION_CASES]


def adjust_plan(desc, values, plan):
    case = values.get("__geometry_compare")
    if not case:
        return plan
    if desc.key != "krea2" or (case not in {x[0] for x in CASES + QUALITY_CASES + R10_CASES + MUON_ME_CASES} and not is_feasibility(case)):
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
    if case in memory_benchmark.CASE_MAP:
        memory_benchmark.apply_flags(cmd, case)
    elif is_feasibility(case):
        cmd.extend(["--family_option", f"geometry_feasibility_steps={FEASIBILITY_STEPS}"])
    elif "--sample_prompts" not in cmd:
        plan.problems.append("Matrix geometry comparison requires usable sample prompts.")
    if case in {x[0] for x in QUALITY_CASES + TSD_FIT_CASES + R10_CASES + MUON_ME_CASES}:
        cmd.extend(["--family_option", "geometry_optimizer_timing=1"])
    if "--train_blocks" in cmd:
        plan.problems.append("Matrix geometry comparison requires unrestricted eligible blocks.")
    plan.console.append(f"[matrix geometry pilot] {case}: fixed rank, data, seed, validation; provisional LR.")
    return plan
