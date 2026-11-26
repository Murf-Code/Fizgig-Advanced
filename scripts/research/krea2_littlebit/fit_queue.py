"""Fresh-process GPU residency sweep for a single fixed compressed LoRA case."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import tomllib

SWAP_STEPS = (0, 4, 8, 12, 16, 20, 22, 24, 26)
OOM_EXIT = 86


def arg(cmd, flag):
    if cmd.count(flag) != 1 or cmd.index(flag) + 1 == len(cmd):
        raise ValueError(f"Expected one {flag}")
    return cmd[cmd.index(flag) + 1]


def set_arg(cmd, flag, value):
    arg(cmd, flag)
    cmd[cmd.index(flag) + 1] = str(value)


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def run_sweep(command, out, launch=subprocess.run):
    """Train from scratch at each tested swap count; retry only explicit CUDA OOM.

    command invokes the native worker and ends in -- followed by trainer flags.
    Failed attempts are isolated; only completed output files are published.
    """
    cut = command.index("--")
    head, original = command[:cut], command[cut+1:]
    out = Path(out).resolve()
    if Path(arg(original, "--output_dir")).resolve() != out:
        raise ValueError("Output directory mismatch")
    if any(x in original for x in ("--resume", "--finetune", "--stage_dataset_config")):
        raise ValueError("Only fresh fixed LoRA comparisons are supported")
    config = Path(arg(original, "--dataset_config"))
    config_text = config.read_text(encoding="utf-8")
    hold = Path(tomllib.loads(config_text)["general"]["holdout_file"])
    hold_bytes = hold.read_bytes()
    if not json.loads(hold_bytes).get("paths"):
        raise ValueError("Missing validation holdout")
    name = arg(original, "--output_name")
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Invalid output name")
    out.mkdir(parents=True, exist_ok=True)
    work = out / "fit_attempts"
    work.mkdir(exist_ok=False)
    report = out / "fit-report.json"
    info = {"schedule": list(SWAP_STEPS), "attempts": [], "selected_swaps": None,
            "status": "running"}
    atomic_json(report, info)
    try:
        for swaps in SWAP_STEPS:
            folder = work / f"swap-{swaps:02d}"
            folder.mkdir()
            attempt_hold = folder / "holdout.json"
            attempt_hold.write_bytes(hold_bytes)
            # The native trainer returns holdout images on exit. Keep its list
            # isolated per attempt, while retaining original dataset/cache paths.
            line = "holdout_file = " + json.dumps(attempt_hold.as_posix(), ensure_ascii=False)
            text, n = re.subn(r"(?m)^holdout_file\s*=\s*[^\n]*$", lambda _: line, config_text)
            if n != 1:
                raise ValueError("Expected one holdout_file in generated dataset TOML")
            attempt_config = folder / "dataset.toml"
            attempt_config.write_text(text, encoding="utf-8")
            args = list(original)
            set_arg(args, "--blocks_to_swap", swaps)
            set_arg(args, "--output_dir", folder)
            set_arg(args, "--dataset_config", attempt_config)
            metrics = folder / "attempt.json"
            row = {"swaps": swaps, "status": "running", "directory": str(folder)}
            info["attempts"].append(row)
            atomic_json(report, info)
            print(f"[LittleBit fit] Trying {swaps} swapped blocks, fresh seed and optimizer", flush=True)
            started = time.monotonic()
            proc = launch([*head, "--attempt-result", str(metrics), "--", *args])
            row.update(exit_code=proc.returncode, seconds=time.monotonic()-started)
            worker = json.loads(metrics.read_text(encoding="utf-8")) if metrics.is_file() else {}
            row["worker"] = worker
            if proc.returncode == OOM_EXIT and worker.get("status") == "cuda_oom":
                row["status"] = "cuda_oom"
                atomic_json(report, info)
                print(f"[LittleBit fit] CUDA OOM at {swaps} swaps; trying next count", flush=True)
                continue
            if proc.returncode != 0 or worker.get("status") != "finished":
                row["status"] = "failed"
                info["status"] = "failed"
                atomic_json(report, info)
                return proc.returncode or 1
            final = folder / (name + ".safetensors")
            snapshot = folder / (".fizgig_val_" + name + ".pt")
            if not final.is_file() or not snapshot.is_file():
                row["status"] = "incomplete"
                raise RuntimeError("Worker finished without final LoRA and validation snapshot")
            # Keep failed attempts private. Publish successful training artifacts
            # (not the attempt's private configuration, status or holdout list).
            for src in folder.iterdir():
                if src.is_file() and src.name not in ("attempt.json", "dataset.toml", "holdout.json"):
                    dest = out / src.name
                    if dest.exists():
                        raise FileExistsError(dest)
                    shutil.copy2(src, dest)
            row["status"] = "finished"
            info.update(status="finished", selected_swaps=swaps)
            atomic_json(report, info)
            print(f"[LittleBit fit] Completed training at {swaps} swapped blocks", flush=True)
            return 0
        info["status"] = "all_cuda_oom"
        atomic_json(report, info)
        return OOM_EXIT
    except BaseException:
        info["status"] = "failed"
        atomic_json(report, info)
        raise
    finally:
        # The original generated holdout is not handed to the worker. Return
        # these pictures once the whole sweep ends, as the ordinary trainer does.
        from fizgig.training import holdout
        holdout.give_back(str(hold))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    argv = sys.argv[1:]
    cut = argv.index("--")
    a = p.parse_args(argv[:cut])
    args = argv[cut+1:]
    sys.path.insert(0, str(a.repo.resolve() / "src"))
    runner = Path(__file__).with_name("native_runner.py")
    command = [sys.executable, str(runner), "train", "--repo", str(a.repo),
               "--archive", str(a.archive), "--", *args]
    sys.exit(run_sweep(command, arg(args, "--output_dir")))


if __name__ == "__main__":
    main()
