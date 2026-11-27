"""One-epoch compatibility tests; intentionally separate from production presets."""
import json
import math
import os
from pathlib import Path

CASES = ("adamw", "adamw8bit", "pagedadamw8bit")
EXPECTED_UPDATES = 13
REPORT = "selective-optimizer-report.json"
AUDIT = "selective-optimizer-audit.json"

COMMON = {
    "FAMILY_KIND": "standard", "NETWORK_TYPE": "LoRA (standard)",
    "NETWORK_DIM": "32", "NETWORK_ALPHA": "32", "LEARNING_RATE": "0.00025",
    "OPTIMIZER_ARGS": "weight_decay=0.01", "GRADIENT_ACCUMULATION": "1",
    "BATCH_SIZE": "1", "MAX_TRAIN_EPOCHS": "1", "SAVE_EVERY_N_EPOCHS": "1",
    "SEED": "42", "FAMILY_PRECISION": "INT8 (8-bit, fastest)", "BLOCKS_SWAP": "22",
    "DATASET_MEGAPIXELS": "0.25", "COMPILE_BLOCKS": "Off",
    "LR_SCHEDULER": "constant", "LR_WARMUP_STEPS": "0",
    "VALIDATION_LOSS": True, "VALIDATION_HOLDOUT": True, "VALIDATION_IMAGES": "4",
    "VALIDATION_GROUPS": "1", "VALIDATION_TIMESTEPS": "4", "VALIDATION_SEED": "42",
    "VALIDATION_EARLY_STOP": False, "ADAPTIVE_LR": False,
    "KREA2_LOSS_WATCH": False, "KREA2_PER_IMAGE_LR": False,
    "KREA2_AUTO_RECAPTION": False, "KREA2_WARMUP_LOOK": False,
    "KREA2_DIRECTION_EDIT": False, "KREA2_TRAINING_METHOD": "Global replay",
    "KREA2_SELECTIVE_BLOCK_CHECK": True, "KREA2_SELECTIVE_COMPENSATION": True,
    "FAMILY_EMA": "0.98 (recommended)", "SAMPLE_ENABLED": False,
    "SAMPLE_AT_FIRST": False, "SAVE_STATE": False, "SAVE_STATE_ON_TRAIN_END": False,
    "TRAIN_SUBFOLDERS": False, "FOLDER_BALANCE": False, "CONTEXT_LORA_PATH": "",
    "MASKED_TRAINING": False, "FAMILY_DOP": False, "TRAINING_MAPS": False,
}


def settings(group_dir):
    group = Path(group_dir)
    return [{**COMMON, "OPTIMIZER_TYPE": case, "__selective_optimizer_smoke": case,
             "LORA_NAME": f"{group.name}-{case}", "LORA_OUTPUT_DIR": str(group / case)}
            for case in CASES]


def attach(plan, values, family, resume=""):
    case = values.get("__selective_optimizer_smoke")
    if not case:
        return plan
    if family != "krea2" or resume or values.get("RESUME_TRAINING") or case not in CASES:
        raise ValueError("Selective optimizer test requires a fresh Krea 2 run")
    if values.get("OPTIMIZER_TYPE") != case or any(values.get(k) != v for k,v in COMMON.items()):
        raise ValueError("Selective optimizer test settings changed; requeue the fixed recipe")
    if Path(values["LORA_OUTPUT_DIR"]).name != case:
        raise ValueError("Selective optimizer test output folder differs from its case")
    if not plan.problems:
        train = [s for s in plan.stages if s.name == "Training"]
        if len(train) != 1:
            raise ValueError("Selective optimizer test needs exactly one Training stage")
        train[0].cmd.extend(["--family_option", "selective_optimizer_smoke=" + case])
    return plan


def _write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def initialize(group_dir, dataset, values):
    group = Path(group_dir)
    group.mkdir(parents=True, exist_ok=False)
    _write(group / "selective-optimizer-manifest.json",
           {"dataset": dataset, "expected_updates": EXPECTED_UPDATES, "settings": values})
    return refresh(group)


def assess(case, rows, validation, state_dtypes):
    if len(rows) != EXPECTED_UPDATES or [r["update"] for r in rows] != list(range(1,14)):
        return "incomplete"
    for r in rows:
        count = len(r["active_blocks"])
        if (not 1 <= count <= 28 or len(set(r["active_blocks"])) != count
                or any(b not in range(28) for b in r["active_blocks"])
                or r["multiplier"] != min(1.5, math.sqrt(28/count))
                or not r["groups_restored"]):
            return "failed_audit"
        if r["update"] <= 8 and count != 28:
            return "failed_audit"
    if case != "adamw" and not state_dtypes.get("torch.uint8", 0):
        return "8bit_state_not_exercised"
    if not any(r["multiplier"] > 1. for r in rows):
        return "compensation_not_exercised"
    if not validation or not math.isfinite(validation[-1]["loss"]) or validation[-1]["step"] != 13:
        return "validation_missing"
    return "passed"


def finish(output_dir, case, implementation, rows, validation, state_dtypes, global_step):
    out = Path(output_dir)
    if out.name != case or case not in CASES:
        raise ValueError("Unexpected Selective optimizer smoke output folder")
    audit = {"case": case, "implementation": implementation, "updates": global_step,
             "compensated_updates": sum(r["multiplier"] > 1. for r in rows),
             "max_multiplier": max((r["multiplier"] for r in rows), default=1.),
             "state1_dtypes": state_dtypes, "steps": rows, "validation": validation,
             "status": assess(case, rows, validation, state_dtypes)}
    if global_step != EXPECTED_UPDATES:
        audit["status"] = "incomplete"
    _write(out / AUDIT, audit)
    refresh(out.parent)
    return audit


def refresh(group):
    group = Path(group)
    manifest = json.loads((group / "selective-optimizer-manifest.json").read_text(encoding="utf-8"))
    runs = []
    for case in CASES:
        p = group / case / AUDIT
        runs.append(json.loads(p.read_text(encoding="utf-8")) if p.is_file()
                    else {"case": case, "status": "not_completed"})
    statuses = [r["status"] for r in runs]
    report = {**manifest, "runs": runs,
              "status": ("passed" if all(s == "passed" for s in statuses)
                         else "incomplete" if "not_completed" in statuses
                         else "review_required")}
    _write(group / REPORT, report)
    return report
