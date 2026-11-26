"""Fixed Krea 2 A/B controls for the stream-sync test branch (real training, not a microbenchmark)."""
import os

DEFAULT_PREVIEW_PROMPTS = "\n".join((
    "Ferwildir, anthropomorphic red fox, female, full body, wearing a green traveling cloak, standing on a stone bridge above a forest stream, expressive eyes, detailed fur",
    "Ferwildir, solo, anthro, male wolf, white fur, blue and purple fur accents, spiky blue hair, blue eyes, full body, running along a beach, looking over shoulder, joyful expression",
    "solo, anthropomorphic male wolf, white fur, blue and purple fur accents, spiky blue hair, blue eyes, sitting in a cafe, realistic watercolor illustration, soft washes and natural lighting",
    "A small wooden boat on a quiet lake surrounded by autumn trees, distant mountains and soft morning light",
))
EPOCHS = 5
RESEARCH_MODES = {
    "repro-normal-a-lr450": "normal",
    "repro-normal-b-lr450": "normal",
    "repro-strict-a-lr450": "strict",
    "repro-strict-b-lr450": "strict",
    "repro-strict-telemetry-lr450": "telemetry",
}
CAUSAL_ROUTES = {
    "causal-unrestricted-lr450": None,
    "causal-gradient-lr450": None,
    "causal-random-lr450": "random",
    "causal-exact-lr450": "exact",
    "causal-block26-lr450": "block26",
    "causal-block13-lr450": "block13",
    "causal-decay10-lr450": None,
    "causal-decay30-lr450": None,
}
CAUSAL_DECAYS = {case: (30.0 if case == "causal-decay30-lr450" else
                       10.0 if case == "causal-decay10-lr450" else 0.01)
                 for case in CAUSAL_ROUTES}
UNRESTRICTED_CASES = {"causal-unrestricted-lr450", "causal-decay10-lr450", "causal-decay30-lr450"}
CALIBRATION_CASES = {
    "calibration-unrestricted-lr200": ("0.00020", "replay"),
    "calibration-unrestricted-lr250": ("0.00025", "replay"),
    "calibration-unrestricted-lr264": ("0.000264", "replay"),
    "calibration-unrestricted-lr300": ("0.00030", "replay"),
    "calibration-unrestricted-lr350": ("0.00035", "replay"),
    "calibration-unrestricted-lr450": ("0.00045", "replay"),
    "calibration-random-lr450": ("0.00045", "random"),
    "calibration-gradient-lr450": ("0.00045", "adaptive_gradient"),
}
CASES = tuple((case, "Global replay", strategy != "replay")
              for case, (_, strategy) in CALIBRATION_CASES.items())

MOVEMENT_PILOT_CASE = "movement-sensitivity-lr450"
MOVEMENT_ISOLATION_CASE = "movement-isolation-lr450"
MOVEMENT_DUAL_CASE = "movement-dual-lr450"
MOVEMENT_MULTI_CASES = {"movement-multi-seed42-lr450": "42", "movement-multi-seed43-lr450": "43"}
HISTORY_CASES = {"history-seed42-lr450": "42", "history-seed43-lr450": "43"}
PREDICTOR_CASES = {"predictors-seed42-lr450": "42", "predictors-seed43-lr450": "43"}
ALLOCATION_CASES = {f"allocation-{mode}-lr450": mode for mode in ("full", "rotate", "uniform", "compensated")}

def case_strategy(case):
    if case in (MOVEMENT_PILOT_CASE, MOVEMENT_ISOLATION_CASE, MOVEMENT_DUAL_CASE) or case in MOVEMENT_MULTI_CASES or case in HISTORY_CASES or case in PREDICTOR_CASES or case in ALLOCATION_CASES:
        return "replay"
    if case in CALIBRATION_CASES:
        return CALIBRATION_CASES[case][1]
    if case in UNRESTRICTED_CASES:
        return "replay"
    if case in RESEARCH_MODES or case in CAUSAL_ROUTES:
        return "adaptive_gradient"
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
    return [{**common, "LEARNING_RATE": CALIBRATION_CASES[case][0],
             "__replay_compare": case, "LORA_NAME": f"{group}-{case}",
             "LORA_OUTPUT_DIR": os.path.join(output_root, group, case),
             "KREA2_TRAINING_METHOD": method, "KREA2_SELECTIVE_BLOCK_CHECK": selective}
            for case, method, selective in CASES]


def pilot_settings(group, output_root):
    """One fresh, strict unrestricted parent; keep legacy comparisons callable."""
    v = next(x for x in settings(group, output_root)
             if x["__replay_compare"] == "calibration-unrestricted-lr450")
    return [{**v, "__replay_compare": MOVEMENT_PILOT_CASE,
             "LORA_NAME": f"{group}-{MOVEMENT_PILOT_CASE}",
             "LORA_OUTPUT_DIR": os.path.join(output_root, group, MOVEMENT_PILOT_CASE),
             "MAX_TRAIN_EPOCHS": "1", "SAVE_EVERY_N_EPOCHS": "1",
             "SAVE_STATE": False, "SAVE_STATE_ON_TRAIN_END": False,
             "SAMPLE_ENABLED": False, "SAMPLE_AT_FIRST": False}]

def isolation_settings(group, output_root):
    v = pilot_settings(group, output_root)[0]
    case = MOVEMENT_ISOLATION_CASE
    return [{**v, "__replay_compare": case, "LORA_NAME": f"{group}-{case}",
             "LORA_OUTPUT_DIR": os.path.join(output_root, group, case)}]

def dual_settings(group, output_root):
    v = pilot_settings(group, output_root)[0]
    case = MOVEMENT_DUAL_CASE
    return [{**v, "__replay_compare": case, "LORA_NAME": f"{group}-{case}",
             "LORA_OUTPUT_DIR": os.path.join(output_root, group, case)}]

def multi_settings(group, output_root):
    v = dual_settings(group, output_root)[0]
    return [{**v, "__replay_compare": case, "SEED": seed,
             "LORA_NAME": f"{group}-{case}",
             "LORA_OUTPUT_DIR": os.path.join(output_root, group, case)}
            for case, seed in MOVEMENT_MULTI_CASES.items()]


def history_settings(group, output_root):
    v = pilot_settings(group, output_root)[0]
    return [{**v, "__replay_compare": case, "SEED": seed,
             "LORA_NAME": f"{group}-{case}",
             "LORA_OUTPUT_DIR": os.path.join(output_root, group, case)}
            for case, seed in HISTORY_CASES.items()]


def predictor_settings(group, output_root):
    v = pilot_settings(group, output_root)[0]
    return [{**v, "__replay_compare": case, "SEED": seed,
             "LORA_NAME": f"{group}-{case}",
             "LORA_OUTPUT_DIR": os.path.join(output_root, group, case)}
            for case, seed in PREDICTOR_CASES.items()]


def allocation_settings(group, output_root):
    v = pilot_settings(group, output_root)[0]
    return [{**v, "__replay_compare": case,
             "LORA_NAME": f"{group}-{case}",
             "LORA_OUTPUT_DIR": os.path.join(output_root, group, case)}
            for case in ALLOCATION_CASES]


def adjust_plan(desc, values, plan):
    """Validate the fixed comparison; keep legacy queued matched controls runnable."""
    case = values.get("__replay_compare")
    if not case:
        return plan
    from fizgig.krea2.selective_policy import STRATEGIES, ADAPTIVE_STRATEGIES
    legacy = {"standard-matched", "selective", "replay", "standard"} | set(STRATEGIES + ADAPTIVE_STRATEGIES)
    if desc.key != "krea2" or case_strategy(case) not in legacy:
        raise ValueError("Invalid Krea 2 replay comparison case.")
    if plan.problems:
        return plan
    stages = [stage for stage in plan.stages if stage.name == "Training"]
    if len(stages) != 1:
        plan.problems.append("Replay comparison needs exactly one normal Training stage.")
        return plan
    strategy = case_strategy(case)
    if strategy in STRATEGIES + ADAPTIVE_STRATEGIES:
        stages[0].cmd.extend(["--family_option", f"selective_strategy={strategy}"])
        if strategy in STRATEGIES:
            stages[0].cmd.extend(["--family_option", "selective_budget=14"])
    if case in (MOVEMENT_PILOT_CASE, MOVEMENT_ISOLATION_CASE, MOVEMENT_DUAL_CASE) or case in MOVEMENT_MULTI_CASES or case in HISTORY_CASES or case in PREDICTOR_CASES:
        if (values.get("LEARNING_RATE") != "0.00045" or values.get("MAX_TRAIN_EPOCHS") != "1"
                or values.get("KREA2_SELECTIVE_BLOCK_CHECK") or values.get("SAMPLE_ENABLED")):
            raise ValueError("Movement sensitivity settings differ from the fixed recipe.")
        stages[0].cmd.extend(["--family_option", "replay_research=calibration",
                              "--family_option", "replay_calibration_lr=0.00045",
                              "--family_option", "movement_pilot_step=8"])
        if case in PREDICTOR_CASES:
            if values.get("SEED") != PREDICTOR_CASES[case] or values.get("VALIDATION_SEED") != "42":
                raise ValueError("Predictor verification requires fixed training and validation seeds.")
            stages[0].cmd.extend(["--family_option", "movement_predictors=1"])
        elif case in HISTORY_CASES:
            if values.get("SEED") != HISTORY_CASES[case] or values.get("VALIDATION_SEED") != "42":
                raise ValueError("Historical reversion requires fixed training and validation seeds.")
            stages[0].cmd.extend(["--family_option", "movement_history=1"])
        elif case in MOVEMENT_MULTI_CASES:
            if values.get("SEED") != MOVEMENT_MULTI_CASES[case] or values.get("VALIDATION_SEED") != "42":
                raise ValueError("Multi-checkpoint training and validation seeds differ from fixed recipe.")
            stages[0].cmd.extend(["--family_option", "movement_dual=1",
                                  "--family_option", "movement_dual_multi=1"])
        elif case == MOVEMENT_ISOLATION_CASE:
            stages[0].cmd.extend(["--family_option", "movement_isolation=1"])
        elif case == MOVEMENT_DUAL_CASE:
            stages[0].cmd.extend(["--family_option", "movement_dual=1"])
    if case in ALLOCATION_CASES:
        if (values.get("LEARNING_RATE") != "0.00045" or values.get("MAX_TRAIN_EPOCHS") != "1"
                or values.get("KREA2_SELECTIVE_BLOCK_CHECK") or values.get("SAMPLE_ENABLED")
                or values.get("SEED") != "42" or values.get("VALIDATION_SEED") != "42"):
            raise ValueError("Allocation experiment differs from the fixed recipe.")
        stages[0].cmd.extend(["--family_option", "replay_research=calibration",
                              "--family_option", "replay_calibration_lr=0.00045",
                              "--family_option", f"movement_allocation={ALLOCATION_CASES[case]}"])
    if case in RESEARCH_MODES:
        stages[0].cmd.extend(["--family_option", f"replay_research={RESEARCH_MODES[case]}"])
    if case in CALIBRATION_CASES:
        expected_lr, expected_strategy = CALIBRATION_CASES[case]
        if (values.get("LEARNING_RATE") != expected_lr
                or values.get("KREA2_SELECTIVE_BLOCK_CHECK") != (expected_strategy != "replay")):
            raise ValueError("Experiment 5C queue settings differ from the fixed recipe.")
        stages[0].cmd.extend(["--family_option", "replay_research=calibration",
                              "--family_option", f"replay_calibration_lr={expected_lr}"])
    if case in CAUSAL_ROUTES:
        stages[0].cmd.extend(["--family_option", "replay_research=causal",
                              "--family_option", f"replay_causal_decay={CAUSAL_DECAYS[case]}"])
        route = CAUSAL_ROUTES[case]
        if route:
            stages[0].cmd.extend(["--family_option", f"replay_causal_route={route}"])
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
