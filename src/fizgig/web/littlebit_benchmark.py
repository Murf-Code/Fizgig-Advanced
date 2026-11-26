"""Fixed Krea2 frozen-base comparison; in-tree independent codec."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tomllib
import zipfile

CASES = {"stock": ("int8", 0), "q6": ("q6", 128), "q4": ("q4", 128),
         "littlebit-r512": ("littlebit", 512), "littlebit2-r512": ("littlebit2", 512),
         "littlebit2-r1024": ("littlebit2", 1024)}
EPOCHS = 5
SWAPS = 22
COMMON = {
    "FAMILY_KIND": "standard", "NETWORK_TYPE": "LoRA (standard)",
    "KREA2_TRAINING_METHOD": "Standard", "KREA2_SELECTIVE_BLOCK_CHECK": False,
    "NETWORK_DIM": "32", "NETWORK_ALPHA": "32", "LEARNING_RATE": "0.00025",
    "OPTIMIZER_TYPE": "muon", "GRADIENT_ACCUMULATION": "1", "BATCH_SIZE": "1",
    "MAX_TRAIN_EPOCHS": str(EPOCHS), "SAVE_EVERY_N_EPOCHS": "5",
    "SAVE_STATE": False, "SAVE_STATE_ON_TRAIN_END": False, "SEED": "42",
    "FAMILY_PRECISION": "INT8 (8-bit, fastest)", "BLOCKS_SWAP": str(SWAPS),
    "DATASET_MEGAPIXELS": "0.25", "COMPILE_BLOCKS": "Off", "FAMILY_EMA": "Off",
    "VALIDATION_LOSS": True, "VALIDATION_HOLDOUT": True, "VALIDATION_IMAGES": "4",
    "VALIDATION_GROUPS": "1", "VALIDATION_TIMESTEPS": "4", "VALIDATION_SEED": "42",
    "VALIDATION_EARLY_STOP": False, "ADAPTIVE_LR": False,
    "KREA2_LOSS_WATCH": False, "KREA2_PER_IMAGE_LR": False,
    "KREA2_AUTO_RECAPTION": False, "KREA2_WARMUP_LOOK": False, "SAMPLE_ENABLED": False,
}


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def settings(root):
    root = Path(root).resolve()
    return [{**COMMON, "__littlebit_case": case,
             "LORA_NAME": root.name + "-" + case, "LORA_OUTPUT_DIR": str(root / case)}
            for case in CASES]


def arg(cmd, flag):
    if cmd.count(flag) != 1:
        raise ValueError("Missing or repeated trainer option: " + flag)
    return cmd[cmd.index(flag) + 1]


def adjust_plan(desc, values, plan, repo, resume=False):
    case = values.get("__littlebit_case")
    if not case or plan.problems:
        return plan
    try:
        if resume or desc.key != "krea2" or case not in CASES:
            raise ValueError("LittleBit comparisons must be fresh Krea2 runs; resume is not supported.")
        if any(values.get(k) != v for k,v in COMMON.items()):
            raise ValueError("LittleBit comparison settings are fixed. Remove and queue again.")
        stages = [s for s in plan.stages if s.name == "Training"]
        if len(stages) != 1:
            raise ValueError("Expected one native Training stage.")
        stage = stages[0]
        cmd = stage.cmd
        expected = {"--family":"krea2", "--precision":"int8", "--blocks_to_swap":str(SWAPS),
                    "--network_dim":"32", "--network_alpha":"32", "--seed":"42",
                    "--max_train_epochs":str(EPOCHS), "--optimizer_type":"muon"}
        for flag, val in expected.items():
            if arg(cmd, flag) != val:
                raise ValueError(f"Unexpected resolved trainer setting: {flag}")
        if any(x in cmd for x in ("--finetune","--resume","--adaptive_lr","--training_adapter",
                                  "--context_lora_path","--sample_prompts","--save_state",
                                  "--save_state_on_train_end","--int8_rotated_backward")):
            raise ValueError("Conflicting trainer option in LittleBit comparison.")
        if not tomllib.loads(plan.dataset_toml).get("general",{}).get("holdout_file"):
            raise ValueError("Comparison needs an actual validation holdout. Use a larger dataset.")
        # Freeze allocator recovery and classic block swap identically in every arm.
        cmd[:] = [x for x in cmd if x not in ("--send_only_force","--send_only_swap","--no-send_only_swap",
                        "--auto_reclaim_vram","--no-auto_reclaim_vram","--background_loading","--no-background_loading")]
        cmd += ["--no-auto_reclaim_vram", "--no-send_only_swap", "--no-background_loading"]
        if "--ema_decay" in cmd:
            cmd[cmd.index("--ema_decay")+1] = "0"
        else:
            cmd += ["--ema_decay","0"]
        raw = Path(arg(cmd,"--dit")).resolve()
        if not raw.is_file() or raw.suffix.lower() != ".safetensors":
            raise ValueError("Select the original Krea2 Raw BF16 safetensors in Preferences.")
        out = Path(arg(cmd,"--output_dir")).resolve()
        root = out.parent
        name = arg(cmd,"--output_name")
        if out.name != case or name != root.name + "-" + case:
            raise ValueError("Invalid comparison output identity.")
        archive = out / "compact.safetensors"
        prepare = Path(repo) / "scripts/research/krea2_littlebit/prepare_case.py"
        prep_cmd = [cmd[0],str(prepare),"--raw",str(raw),
                    "--case",case,"--out",str(out),"--dataset",arg(cmd,"--dataset_config")]
        from fizgig.families.launch import Stage
        plan.stages.insert(0, Stage("LittleBit projection screen",prep_cmd))
        if case != "stock":
            stock_name = root.name + "-stock"
            stock = root / "stock"
            stage.cmd = [cmd[0],str(prepare.with_name("native_runner.py")),"train","--repo",str(repo),"--archive",str(archive),"--",*cmd[2:]]
            plan.stages.append(Stage("Stock-base transfer validation", [
                cmd[0],str(prepare.with_name("native_runner.py")),"evaluate","--repo",str(repo),"--dit",str(raw),
                "--snapshot",str(stock/f".fizgig_val_{stock_name}.pt"),
                "--stock-lora",str(stock/f"{stock_name}.safetensors"),
                "--compressed-lora",str(out/f"{name}.safetensors"),
                "--out",str(out/"transfer.json"),"--swaps",str(SWAPS)]))
        plan.console.append(f"[LittleBit test] {case}: fixed 5 epochs; synthetic projection gate; no previews or resume.")
    except (ValueError, OSError, TypeError) as e:
        plan.problems.append(str(e))
    return plan


def atomic_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")
    os.replace(tmp,path)


def create(root, folder, configs):
    root = Path(root)
    root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/"littlebit-report.json",{"dataset":folder,"scope":"research feasibility",
        "cases":{v["__littlebit_case"]:{"status":"queued","settings":v} for v in configs}})


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def update(values, status, log=None):
    case = (values or {}).get("__littlebit_case")
    if case not in CASES:
        return
    out = Path(values["LORA_OUTPUT_DIR"])
    root = out.parent
    report = root/"littlebit-report.json"
    data = read_json(report)
    if not data or case not in data["cases"]:
        return
    entry = data["cases"][case]
    preparation = read_json(out/"prepare.json")
    if preparation is not None:
        entry["preparation"] = preparation
        if preparation.get("status") == "rejected":
            status = "screen_rejected"
        elif preparation.get("status") == "blocked":
            status = "blocked"
    entry["status"] = status
    entry["transfer"] = read_json(out/"transfer.json")
    for p in out.glob(".perf_*.json"):
        entry["performance"] = read_json(p)
    if log and Path(log).is_file() and out.is_dir():
        shutil.copyfile(log,out/"run.log")
    atomic_json(report,data)
    lines = []
    for label,row in data["cases"].items():
        lines.append(f"=== {label}: {row['status']} ===\n")
        if row.get("transfer"):
            lines.append(json.dumps(row["transfer"],indent=2)+"\n")
        p = root/label/"run.log"
        if p.is_file():
            lines.append(p.read_text(encoding="utf-8",errors="replace")+"\n")
    (root/"littlebit-comparison.log").write_text("\n".join(lines),encoding="utf-8")
    terminal = {"finished","failed","screen_rejected","blocked","stopped"}
    if all(r["status"] in terminal for r in data["cases"].values()):
        with zipfile.ZipFile(root/"littlebit-results.zip","w",zipfile.ZIP_DEFLATED) as z:
            for p in (report,root/"littlebit-comparison.log"):
                z.write(p,p.name)
            for label in CASES:
                for p in (root/label).glob("*"):
                    if p.is_file() and p.suffix in (".json",".log"):
                        z.write(p,str(p.relative_to(root)))
