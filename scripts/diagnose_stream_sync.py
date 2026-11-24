"""Standalone GPU correctness diagnostic. Does not modify training preferences or caches.

Run inside Fizgig's venv. Default: reduced-width 28-block Krea 2, actual INT8
kernels, checkpointed forward/backward, Muon, 22 swapped blocks. Optional
--model and --cache use the real checkpoint and fixed cached training examples.
No VAE/text encoder or full training run is needed. Never disables block swap
for the full checkpoint. CPU-only mode is for harness smoke tests, not GPU proof.
"""
import argparse
from dataclasses import asdict
import gc
import json
import math
import os
import platform
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
import contextlib
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from fizgig.families import quant
from fizgig.families.lora import FamilyLoRA
from fizgig.families.krea2 import KREA2
from fizgig.krea2.model import SingleMMDiTConfig, SingleStreamDiT
from fizgig.krea2.offloading import LoRAStreamOffloader, BlockSwapConfig
from fizgig.krea2.stream_calibration import recommend_stream_strategy
from fizgig.training.optimizers import create_optimizer


def inputs(args, config):
    """Freeze the entire input sequence before any of the runs."""
    gen = torch.Generator(device="cpu").manual_seed(args.seed + 100)
    pairs = []
    if args.cache:
        from safetensors.torch import load_file
        from fizgig.dataset import compact_text
        folder = Path(args.cache)
        arch = KREA2.arch_id
        for path in sorted(folder.glob(f"*_{arch}.safetensors")):
            stem = path.name.rsplit("_", 2)[0]
            te = folder / f"{stem}_{arch}_te.safetensors"
            if te.is_file():
                pairs.append((path, te))
        if not pairs:
            raise ValueError("No matching Krea 2 latent/text cache pairs in " + str(folder))
    # Repeat the same caption/latent shapes after warmup, while drawing
    # independent fixed noise per step. Keep a separate validation example
    # whenever the cache contains enough pictures.
    train_count = min(len(pairs), max(1, args.warmup or args.steps))
    if len(pairs) > 1:
        train_count = min(train_count, len(pairs) - 1)
    out = []
    for i in range(args.steps + 1):
        if pairs:
            index = (train_count % len(pairs)) if i == args.steps else (i % train_count)
            path, te = pairs[index]
            data = load_file(str(path))
            keys = [k for k in data if k.startswith("latent_") and data[k].ndim == 3]
            if len(keys) != 1:
                raise ValueError(f"Expected one still latent in {path}, found {keys}")
            latent = data[keys[0]].unsqueeze(0)
            text = compact_text.load(str(te))
            cond = {k: text["cond__" + k] for k in ("hidden_states", "attention_mask")}
            name = path.name
        else:
            latent = torch.randn(1, config.channels, args.latent_size, args.latent_size, generator=gen)
            cond = {"hidden_states": torch.randn(1, 24, config.txtlayers, config.txtdim, generator=gen),
                    "attention_mask": torch.ones(1, 24, dtype=torch.bool)}
            name = f"synthetic-{i}"
        noise = torch.randn(latent.shape, generator=gen)
        timestep = torch.tensor([0.25 + 0.5 * (i % 5) / 4], dtype=torch.float32)
        out.append((name, latent, noise, timestep, cond))
    return out


class UnsafePinning(RuntimeError):
    """Skip a high-pinning case, rather than exhaust system RAM."""


def configure_swap(driver, model, count, device, backward=True, send_only=False, *, case):
    import psutil
    pinned = case["pinned"]
    ring = min(case["ring"], 28 - count)
    if pinned and device.type == "cuda":
        # Classic's rotating pin buffers eventually cover the offloaded blocks.
        # Direct streaming pins every master. Allow room for one additional
        # block and keep substantial unpinned system RAM for the OS.
        block_bytes = sum(m.weight.numel() * m.weight.element_size()
                          for m in model.blocks[0].modules()
                          if isinstance(m, torch.nn.Linear))
        estimate = block_bytes * (min(28, count + ring) if send_only else count + 1)
        vm = psutil.virtual_memory()
        reserve = max(6 * 2**30, int(vm.total * 0.15))
        print(f"  pinned allocation estimate={estimate/2**30:.2f} GiB; available RAM={vm.available/2**30:.2f} GiB; reserve={reserve/2**30:.2f} GiB", flush=True)
        if vm.available < estimate + reserve:
            raise UnsafePinning("Insufficient available system RAM for conservative pinning budget")
    model.enable_block_swap(count, BlockSwapConfig(torch.device(device), supports_backward=backward,
                                                   h2d_only=send_only, use_pinned_memory=pinned,
                                                   ring_size=ring, match_classic_vram=True))
    model.move_to_device_except_swap_blocks(torch.device(device))
    model.switch_block_swap_for_training()


def build(args, device, case):
    mode = case["mode"]
    driver = KREA2.load_driver()
    # Override the driver's allocation configuration only inside this worker;
    # keep the production quantized loading path (including short-RAM loading).
    driver.enable_block_swap = lambda model, count, dev, supports_backward=True, send_only=False: configure_swap(
        driver, model, count, dev, supports_backward, send_only, case=case)
    os.environ["FIZGIG_EXPERIMENTAL_STREAM_RING"] = str(case["ring"])
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.model:
        model, swapped = quant.load_base(driver, args.model, device, args.precision,
                                         args.swap, True, send_only=mode.startswith("fast"))
        if swapped != args.swap:
            raise RuntimeError(f"Requested {args.swap} swapped blocks, got {swapped}")
    else:
        config = SingleMMDiTConfig(features=args.width, tdim=32, txtdim=64, heads=args.width // 64,
                                  kvheads=max(1, args.width // 128), multiplier=4, layers=28,
                                  patch=2, channels=16, txtlayers=2, txtheads=4, txtkvheads=4)
        model = SingleStreamDiT(config, attn_mode="torch")
        model.to(dtype=torch.bfloat16)
        model.requires_grad_(False)
        quant.quantize(model, driver, args.precision, device, store_device="cpu")
        if mode == "resident":
            model.to(device)
        else:
            driver.enable_block_swap(model, args.swap, device, True, send_only=mode.startswith("fast"))
    if mode.startswith("fast"):
        if not isinstance(model.offloader, LoRAStreamOffloader) or model.offloader.B != case["ring"]:
            raise RuntimeError("Requested send-only ring was not actually allocated")
        if model.offloader.num_blocks - model.offloader.S + model.offloader.B != 28 - args.swap:
            raise RuntimeError("Ring does not preserve the classic block-weight capacity")
    elif mode != "resident" and isinstance(model.offloader, LoRAStreamOffloader):
        raise RuntimeError("Classic mode unexpectedly uses send-only")
    if mode != "resident" and model.offloader.use_pinned_memory != case["pinned"]:
        raise UnsafePinning("Requested pinning strategy was not applied; host safety fallback selected a different mode")
    if mode.startswith("fast") and device.type == "cuda":
        host_plan = model.offloader.host_memory_plan
        if host_plan is None:
            raise RuntimeError("Streaming host allocation policy was not reported")
        if not case["pinned"] and host_plan.staging_slots != case["ring"]:
            raise UnsafePinning(
                f"Requested {case['ring']} staging slots, allocated {host_plan.staging_slots}; "
                "reduced host pool would confound ring benchmark")
    model.enable_gradient_checkpointing()
    model.train()
    # Reset AFTER all model-loading/quantization/offloader allocations. FamilyLoRA
    # constructs its factors on CPU before moving them onto the GPU.
    torch.manual_seed(args.seed + 1)
    net = FamilyLoRA(model, driver, device)
    net.add_trainable(args.rank, args.rank)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not params:
        raise RuntimeError("No trainable adapters")
    opt, label = create_optimizer(args.optimizer, list(params.values()), args.lr)
    print(f"  optimizer={label}; adapters={len(params)}; trainable={sum(p.numel() for p in params.values()):,}",
          flush=True)
    if mode == "fast_sync" and device.type == "cuda":
        off = model.offloader
        original_load = off._load
        original_wait = off.wait_for_block

        def sync_load(*a, **kw):
            # Prevent reuse until all prior compute is complete, then finish
            # the worker and DMA before any subsequent computation can begin.
            torch.cuda.synchronize(device)
            result = original_load(*a, **kw)
            off.copier.sync()
            torch.cuda.synchronize(device)
            return result

        def sync_wait(*a, **kw):
            result = original_wait(*a, **kw)
            off.copier.sync()
            torch.cuda.synchronize(device)
            return result

        off._load, off.wait_for_block = sync_load, sync_wait
    return driver, model, params, opt


def compare(expected, actual):
    """Report exactness and global/worst-tensor relative L2, never hide NaNs.

    Relative L2 uses a small absolute floor for near-zero tensors; no pass
    threshold is assumed to be appropriate for every BF16 kernel/optimizer.
    """
    if expected.keys() != actual.keys():
        raise RuntimeError("Adapter tensor names differ")
    squared_error = squared_reference = 0.0
    exact = 0
    worst = []
    finite = True
    max_abs = 0.0
    for name, a in expected.items():
        b = actual[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError(f"Tensor shape/dtype mismatch: {name}")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            finite = False
            continue
        exact += int(torch.equal(a, b))
        af, bf = a.float(), b.float()
        delta = af - bf
        error = float(torch.linalg.vector_norm(delta))
        base = float(torch.linalg.vector_norm(af))
        squared_error += error * error
        squared_reference += base * base
        abs_err = float(delta.abs().max())
        max_abs = max(max_abs, abs_err)
        worst.append((error / max(base, 1e-8), abs_err, name))
    worst.sort(reverse=True)
    return dict(finite=finite, tensors=len(expected), exact=exact,
                relative_l2=(squared_error ** 0.5 / max(squared_reference ** 0.5, 1e-8)),
                max_abs=max_abs,
                worst=[dict(name=n, relative_l2=r, max_abs=a) for r, a, n in worst[:5]])


def snapshot(params):
    return {n: p.detach().cpu().clone() for n, p in params.items()}


def forward(driver, model, batch, device):
    _, latent, noise, t, cond = batch
    return driver.loss_at(model, latent.to(device), noise, t, cond)


def execute(args, device, case, batches, reference_dir, record):
    mode, label = case["mode"], case["label"]
    print(f"\n=== {label} ({mode}, pinned={case['pinned']}, ring={case['ring']}, attention={case['attention']}) ===", flush=True)
    driver, model, params, opt = build(args, device, case)
    off = getattr(model, "offloader", None) if mode != "resident" else None
    waits = {"calls": 0, "host_seconds": 0.0}
    if off is not None:
        original_wait = off.wait_for_block

        def timed_wait(*a, **kw):
            t = time.perf_counter()
            try:
                return original_wait(*a, **kw)
            finally:
                waits["calls"] += 1
                waits["host_seconds"] += time.perf_counter() - t

        off.wait_for_block = timed_wait
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    rows = []
    before = snapshot(params)
    init_path = reference_dir / "initial.pt"
    if record:
        torch.save(before, init_path)
    else:
        init = compare(torch.load(init_path, weights_only=True), before)
        if not init["finite"] or init["exact"] != init["tensors"]:
            raise RuntimeError(f"Initial LoRA tensors do not match reference: {init}")
    for i in range(args.steps):
        opt.zero_grad(set_to_none=True)
        waits.update(calls=0, host_seconds=0.0)
        transfer_start = dict(off.transfers) if off is not None else {}
        start = time.perf_counter()
        loss = forward(driver, model, batches[i], device)
        loss.backward()
        # No diagnostic synchronization inside normal forward or backward.
        # Capture gradients after optimizer.step (Muon/AdamW leave .grad intact).
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        step_wait = dict(waits)
        step_transfers = {k: v - transfer_start[k] for k, v in off.transfers.items()} if off is not None else {}
        if any(p.grad is None for p in params.values()):
            missing = [n for n, p in params.items() if p.grad is None]
            raise RuntimeError(f"Missing adapter gradients: {missing[:5]}")
        grads = {n: p.grad.detach().cpu().clone() for n, p in params.items()}
        after = snapshot(params)
        updates = {n: after[n] - before[n] for n in after}
        before = after
        # Fixed validation input exercises the forward-only -> training transition
        # identically in all modes, including the resident reference.
        if mode != "resident":
            driver.block_swap_mode(model, inference=True)
        model.eval()
        with torch.no_grad():
            val = float(forward(driver, model, batches[-1], device))
        model.train()
        if mode != "resident":
            driver.block_swap_mode(model, inference=False)
        row = dict(step=i + 1, image=batches[i][0], loss=float(loss.detach()), validation=val,
                   seconds=elapsed, host_wait_ms=step_wait["host_seconds"] * 1000,
                   wait_calls=step_wait["calls"], transfers=step_transfers,
                   gradient_norm=sum(float(torch.linalg.vector_norm(g.float())) ** 2
                                                       for g in grads.values()) ** 0.5)
        path = reference_dir / f"step-{i}.pt"
        if record:
            torch.save({"grads": grads, "updates": updates, "loss": row["loss"], "validation": val}, path)
            # Compare reference against itself to also detect nonfinite tensors.
            row["gradients"] = compare(grads, grads)
            row["updates"] = compare(updates, updates)
        else:
            expected = torch.load(path, weights_only=True)
            row["loss_abs_error"] = abs(row["loss"] - expected["loss"])
            row["validation_abs_error"] = abs(val - expected["validation"])
            row["gradients"] = compare(expected["grads"], grads)
            row["updates"] = compare(expected["updates"], updates)
            del expected
        print(f"step {i+1}: loss={row['loss']:.8f} val={val:.8f} gn={row['gradient_norm']:.6g} "
              f"gradient_rel_L2={row['gradients']['relative_l2']:.6g} "
              f"update_rel_L2={row['updates']['relative_l2']:.6g} "
              f"exact gradients={row['gradients']['exact']}/{len(grads)} "
              f"seconds={elapsed:.3f} host_wait_ms={row['host_wait_ms']:.1f} "
              f"H2D={step_transfers.get('h2d_bytes', 0)/2**30:.2f} GiB "
              f"D2H={step_transfers.get('d2h_bytes', 0)/2**30:.2f} GiB", flush=True)
        rows.append(row)
        del grads, updates, after, loss
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    reserved = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None
    pinned_bytes = pinned_storage(model, off) if off is not None else 0
    host_plan = getattr(off, "host_memory_plan", None)
    shared = windows_gpu_memory() if device.type == "cuda" else {}
    if off is not None:
        off.remove_hooks()
        del off.wait_for_block
        if mode == "fast_sync" and device.type == "cuda":
            del off._load
    measured = rows[args.warmup:] or rows
    import statistics
    summary = {"median_step_seconds": statistics.median(r["seconds"] for r in measured),
               "mean_step_seconds": statistics.mean(r["seconds"] for r in measured),
               "mean_host_wait_ms": statistics.mean(r["host_wait_ms"] for r in measured),
               "mean_h2d_gib": statistics.mean(r["transfers"].get("h2d_bytes", 0) / 2**30 for r in measured),
               "mean_d2h_gib": statistics.mean(r["transfers"].get("d2h_bytes", 0) / 2**30 for r in measured)}
    print(f"  measured mean={summary['mean_step_seconds']:.3f}s, median={summary['median_step_seconds']:.3f}s, "
          f"pinned storage={pinned_bytes/2**30:.3f} GiB, allocated peak={peak} GiB", flush=True)
    return {**case, "steps": rows, "summary": summary, "peak_allocated_gib": peak,
            "peak_reserved_gib": reserved, "pinned_storage_gib": pinned_bytes/2**30,
            "host_memory_plan": asdict(host_plan) if host_plan is not None else None,
            "windows_process_gpu_memory_end": shared}


def pinned_storage(model, off):
    """Unique live pinned storage, not WDDM's shared-memory counter."""
    seen = {}
    def add(x):
        if isinstance(x, torch.Tensor) and x.device.type == "cpu" and x.is_pinned():
            storage = x.untyped_storage()
            seen[storage.data_ptr()] = storage.nbytes()
        elif isinstance(x, dict):
            for y in x.values():
                add(y)
        elif isinstance(x, (list, tuple)):
            for y in x:
                add(y)
    for p in model.parameters():
        add(p)
    for name in ("cpu_flat", "pinned_buffer", "staging_buffer_a", "staging_buffer_b"):
        add(getattr(off, name, None))
    add(getattr(getattr(off, "copier", None), "_staging", None))
    return sum(seen.values())


def windows_gpu_memory():
    """Point sample of WDDM process counters, if available. Never infer shared from pinned."""
    if os.name != "nt":
        return {}
    try:
        cmd = ("$p=" + str(os.getpid()) + "; "
               "$x=Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory "
               "| Where-Object {$_.Name -like ('pid_'+$p+'_*')}; "
               "if($x){@{dedicated=($x|Measure-Object DedicatedUsage -Sum).Sum;"
               "shared=($x|Measure-Object SharedUsage -Sum).Sum}|ConvertTo-Json -Compress}")
        out = subprocess.check_output(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", cmd],
                                      timeout=8, stderr=subprocess.DEVNULL, creationflags=0x08000000)
        data = json.loads(out)
        return {k+"_gib": float(v)/2**30 for k,v in data.items() if v is not None}
    except Exception:
        return {}


class MemorySampler:
    """Sample system RAM / process RSS / NVML device memory, independently of timing."""
    def __init__(self, device):
        self.device = device
        self.stop = threading.Event()
        self.data = {}

    def sample(self):
        import psutil
        proc = psutil.Process()
        vm = psutil.virtual_memory()
        nv, handle = None, None
        if self.device.type == "cuda":
            from fizgig.training.perf_watch import _gpu_handle
            nv, handle = _gpu_handle(torch.cuda.current_device())
        def peak(k, v):
            self.data[k] = max(v, self.data.get(k, 0))
        while not self.stop.is_set():
            vm = psutil.virtual_memory()
            peak("system_used_peak_gib", vm.used/2**30)
            self.data["system_available_min_gib"] = min(vm.available/2**30,
                                                        self.data.get("system_available_min_gib", float("inf")))
            peak("process_rss_peak_gib", proc.memory_info().rss/2**30)
            if handle is not None:
                try:
                    peak("nvml_device_used_peak_gib", nv.nvmlDeviceGetMemoryInfo(handle).used/2**30)
                except Exception:
                    pass
            self.stop.wait(0.25)

    def __enter__(self):
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


def cases(args):
    def c(label, mode, pinned=False, ring=2, attention="auto", reference="auto", record=False):
        return dict(label=label, mode=mode, pinned=pinned, ring=ring, attention=attention,
                    reference=reference, record=record)
    if args.suite == "calibration":
        # Bracket the candidate measurements with classic repeats and reverse
        # the second pass to reduce consistent warm-device/order bias.
        strategies = [(False, ring) for ring in (2, 3, 4) if ring <= 28 - args.swap]
        if not args.cpu:
            strategies += [(True, ring) for ring in (2, 3, 4) if ring <= 28 - args.swap]
        def candidate(pinned, ring, repeat):
            label = f"CAL_{'direct' if pinned else 'staged'}_ring{ring}_{repeat}"
            return c(label, "fast", pinned=pinned, ring=ring)
        return ([c("A_classic_pageable", "classic", record=True)]
                + [candidate(p, r, 1) for p, r in strategies]
                + [candidate(p, r, 2) for p, r in reversed(strategies)]
                + [c("D_classic_repeat", "classic")])
    result = [c("A_classic_pageable", "classic", record=True), c("D_classic_repeat", "classic"),
              c("B_fast_staged_ring2", "fast"), c("C_fast_staged_repeat", "fast"),
              c("E_fast_serialized", "fast_sync")]
    if args.suite == "expanded":
        if not args.cpu:
            result += [c("G_classic_pinned", "classic", True), c("I_fast_fully_pinned", "fast", True)]
        result += [c("J_fast_staged_ring3", "fast", ring=3), c("K_fast_staged_ring4", "fast", ring=4)]
        result += [c("L_classic_math", "classic", attention="math", reference="math", record=True),
                   c("M_classic_math_repeat", "classic", attention="math", reference="math"),
                   c("N_fast_math", "fast", attention="math", reference="math"),
                   c("O_fast_math_repeat", "fast", attention="math", reference="math")]
    if not args.model:
        result.append(c("F_no_swap", "resident"))
    return result


def select_cases(available, requested):
    """Keep the classic reference and its repeat for every selected group."""
    if not requested:
        return available
    known = {c["label"]: c for c in available}
    unknown = set(requested) - known.keys()
    if unknown:
        raise ValueError("Unavailable diagnostic case(s): " + ", ".join(sorted(unknown)))
    groups = {known[label]["reference"] for label in requested}
    selected = set(requested)
    # A targeted calibration must still include the paired independent repeat.
    calibration_signatures = {(known[label]["pinned"], known[label]["ring"])
                              for label in requested if label.startswith("CAL_")}
    selected.update(c["label"] for c in available if c["label"].startswith("CAL_")
                    and (c["pinned"], c["ring"]) in calibration_signatures)
    for case in available:
        if case["reference"] in groups and case["mode"] == "classic" and (
                case["record"] or "classic_repeat" in case["label"] or "classic_math_repeat" in case["label"]):
            selected.add(case["label"])
    return [c for c in available if c["label"] in selected]


def evaluate_exact_parity(report, planned):
    """Fail closed on missing steps/cases, nonfinite results or any numerical drift.

    A failed gate is not automatically evidence of an offloader race: nondeterministic
    attention can also fail the classic repeat.
    """
    issues = []
    expected_steps = report["configuration"]["steps"]
    by_label = {}
    for run in report["runs"]:
        label = run["label"]
        if label in by_label:
            issues.append(f"{label}: duplicate result")
        by_label[label] = run
    for case in planned:
        label = case["label"]
        run = by_label.get(label)
        if run is None or run.get("status") != "completed":
            issues.append(f"{label}: missing, skipped or failed")
            continue
        rows = run.get("steps", [])
        if [row.get("step") for row in rows] != list(range(1, expected_steps + 1)):
            issues.append(f"{label}: incomplete or out-of-order step sequence")
        for row in rows:
            prefix = f"{label} step {row.get('step')}"
            for key in ("loss", "validation"):
                if not math.isfinite(row.get(key, float("nan"))):
                    issues.append(f"{prefix}: nonfinite or missing {key}")
                if not case["record"] and row.get(key + "_abs_error") != 0:
                    issues.append(f"{prefix}: {key} does not exactly match reference")
            for key in ("gradients", "updates"):
                metric = row.get(key, {})
                count = metric.get("tensors", 0)
                if (metric.get("finite") is not True or count <= 0
                        or metric.get("exact") != count or metric.get("relative_l2") != 0
                        or metric.get("max_abs") != 0):
                    issues.append(f"{prefix}: {key} not finite and bit-exact across all tensors")
    if report["errors"]:
        issues.append("One or more diagnostic workers failed")
    if not planned:
        issues.append("No diagnostic cases planned")
    return {"passed": not issues, "mode": "exact",
            "scope": "cpu-smoke-only" if report["configuration"]["cpu"] else "cuda",
            "required_cases": [case["label"] for case in planned], "issues": issues}


def worker(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    args = argparse.Namespace(**config["args"])
    case = config["case"]
    device = torch.device("cpu" if args.cpu else "cuda")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ["MUSUBI_TUNER_OFFLOADER_DEBUG"] = "0"
    # Force math SDPA through both forward and checkpoint recomputation; the
    # internal Fizgig attention context must not override this outer context.
    os.environ["FIZGIG_SDPA_BACKEND"] = "default" if case["attention"] == "math" else "auto"
    from torch.nn.attention import SDPBackend, sdpa_kernel
    ctx = sdpa_kernel(SDPBackend.MATH) if case["attention"] == "math" else contextlib.nullcontext()
    if case["attention"] == "math":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    result = {}
    try:
        batches = torch.load(config["batches"], weights_only=True)
        with MemorySampler(device) as mem, ctx:
            result = execute(args, device, case, batches, Path(config["reference_dir"]), case["record"])
        result["memory_samples"] = mem.data
        result["status"] = "completed"
    except UnsafePinning as e:
        result = {**case, "status": "skipped", "reason": str(e)}
        print("SKIPPED: " + str(e), flush=True)
    except Exception as e:
        result = {**case, "status": "failed", "error": str(e), "traceback": traceback.format_exc()}
        traceback.print_exc()
    Path(config["result"]).write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    return int(result["status"] == "failed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", help="Optional real Krea 2 safetensors checkpoint")
    p.add_argument("--cache", help="Optional real Krea 2 cache directory (requires --model)")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--suite", choices=("basic", "expanded", "calibration"), default="basic")
    p.add_argument("--case", action="append", help="Run only this case (repeatable); classic reference and repeat are included automatically")
    p.add_argument("--require-parity", action="store_true",
                   help="Exit nonzero unless every selected case completes with exact loss, validation, gradient and update parity")
    p.add_argument("--swap", type=int, default=22)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--precision", choices=("int8", "bf16"), default="int8")
    p.add_argument("--optimizer", choices=("muon", "adamw"), default="muon")
    p.add_argument("--lr", type=float, default=0.00025)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--latent-size", type=int, default=32)
    p.add_argument("--report", default="stream-sync-report.json")
    p.add_argument("--scratch", help="Directory for temporary CPU reference tensors")
    p.add_argument("--cpu", action="store_true", help="Harness smoke test only; not asynchronous GPU validation")
    args = p.parse_args()
    if args.steps < 1 or not 1 <= args.swap <= 26 or args.rank < 1 or args.width < 128 or args.width % 128:
        p.error("Require positive steps/rank, swap 1..26 and width a multiple of 128")
    if args.warmup < 0 or args.warmup >= args.steps:
        args.warmup = min(max(0, args.warmup), args.steps - 1)
    try:
        planned = select_cases(cases(args), args.case)
    except ValueError as exc:
        p.error(str(exc))
    if any(c["mode"].startswith("fast") and c["ring"] > 28 - args.swap for c in planned):
        p.error("Selected rings must fit the classic block-weight VRAM budget")
    if args.suite == "calibration" and not args.cpu and (args.warmup < 1 or args.steps - args.warmup < 3):
        p.error("Calibration needs at least one warmup and three measured steps; use --steps 4 or more")
    if args.latent_size < 2 or args.latent_size % 2:
        p.error("latent-size must be positive and divisible by 2")
    if args.cache and not args.model:
        p.error("--cache requires --model (matching actual conditioning dimensions)")
    if args.cpu and args.model:
        p.error("CPU smoke test only supports the reduced synthetic model")
    if not args.cpu and not torch.cuda.is_available():
        p.error("CUDA is not available in this Python environment")
    device = torch.device("cpu" if args.cpu else "cuda")
    # Override experimental settings only inside this diagnostic process.
    os.environ["FIZGIG_EXPERIMENTAL_STREAM_RING"] = "2"
    os.environ["MUSUBI_TUNER_OFFLOADER_DEBUG"] = "0"
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.model:
        from fizgig.krea2.utils import single_mmdit_large_wide as config
    else:
        config = SingleMMDiTConfig(features=args.width, tdim=32, txtdim=64, heads=args.width // 64,
                                  kvheads=max(1, args.width // 128), multiplier=4, layers=28,
                                  patch=2, channels=16, txtlayers=2, txtheads=4, txtkvheads=4)
    batches = inputs(args, config)
    report = {"configuration": vars(args), "torch": torch.__version__, "cuda": torch.version.cuda,
              "device": torch.cuda.get_device_name(device) if not args.cpu else "CPU smoke only",
              "batches": [b[0] for b in batches], "runs": [], "errors": []}
    import psutil
    vm = psutil.virtual_memory()
    hardware = {"os": platform.system(), "os_release": platform.release(),
                "ram_total_gib": vm.total / 2**30, "cpu_logical": psutil.cpu_count(),
                "gpu": report["device"], "torch": torch.__version__, "cuda": torch.version.cuda,
                "hip": torch.version.hip}
    if not args.cpu:
        props = torch.cuda.get_device_properties(device)
        hardware.update(vram_total_gib=props.total_memory / 2**30,
                        compute_capability=[props.major, props.minor])
        try:
            hardware["gpu_driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, timeout=5, stderr=subprocess.DEVNULL).strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            hardware["gpu_driver"] = None
    report["hardware"] = hardware
    print(json.dumps({k: v for k, v in report.items() if k not in ("runs", "errors")}, indent=2), flush=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report["skipped"] = []
    report["planned_cases"] = planned
    try:
        report["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        report["git_commit"] = None
    report["measurement_notes"] = [
        "Host wait time is wall time inside wait_for_block, not actual GPU-side stall time.",
        "WDDM shared/dedicated counters are optional end-of-case samples, not peaks.",
        "Peak allocated/reserved are PyTorch CUDA allocator metrics; NVML is device-wide.",
        "Math attention cases use their own reference and strict deterministic algorithms.",
        "Host safety fallback cases are skipped instead of timing a different strategy under the requested label.",
        "Each case runs in its own process; reference tensors and cached batches are shared on disk."]
    with tempfile.TemporaryDirectory(prefix="fizgig-stream-sync-", dir=args.scratch) as tmp:
        tmp = Path(tmp)
        if args.model:
            # Two reference groups at rank 32: FP32 initial parameters,
            # gradients and update tensors, plus serialization headroom.
            estimate = int(117_305_760 * (args.rank / 32) * 4 * (1 + 2 * args.steps)
                           * (len({c["reference"] for c in planned}) * 1.15)) + 2**30
            free = shutil.disk_usage(tmp).free
            if free < estimate:
                msg = f"Diagnostic reference storage needs about {estimate/2**30:.1f} GiB free; {free/2**30:.1f} GiB available."
                report["errors"].append({"label": "preflight", "error": msg})
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(msg, flush=True)
                return 1
        batch_path = tmp / "batches.pt"
        torch.save(batches, batch_path)
        ready = set()
        for case in planned:
            group = case["reference"]
            if not case["record"] and group not in ready:
                report["skipped"].append({**case, "reason": "Reference case unavailable"})
                continue
            refdir = tmp / group
            refdir.mkdir(exist_ok=True)
            result_path = tmp / (case["label"] + ".json")
            config_path = tmp / "worker.json"
            config_path.write_text(json.dumps(dict(args=vars(args), case=case, batches=str(batch_path),
                                                  reference_dir=str(refdir), result=str(result_path))), encoding="utf-8")
            env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")
            proc = subprocess.run([sys.executable, "-u", __file__, "--worker", str(config_path)], env=env)
            result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {
                **case, "status": "failed", "error": f"Worker exited {proc.returncode} without a report"}
            if result["status"] == "completed" and proc.returncode == 0:
                report["runs"].append(result)
                if case["record"]:
                    ready.add(group)
            elif result["status"] == "skipped":
                report["skipped"].append(result)
            else:
                report["errors"].append(result)
            report_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    report["exact_parity"] = evaluate_exact_parity(report, planned)
    if args.suite == "calibration":
        report["calibration"] = recommend_stream_strategy(report, evaluate_exact_parity)
        recommendation = report["calibration"]
        print(f"Calibration: {recommendation['status']} - {recommendation['reason']}", flush=True)
        if recommendation["selected"]:
            print(f"Recommended for this workload only: {recommendation['selected']}", flush=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    gate = report["exact_parity"]
    print(f"Exact parity ({gate['scope']}): {'PASS' if gate['passed'] else 'NOT PASSED'}", flush=True)
    for issue in gate["issues"]:
        print("  " + issue, flush=True)
    print("\n=== Summary (excluding warmup) ===", flush=True)
    for run in report["runs"]:
        print(f"{run['label']}: {run['summary']['mean_step_seconds']:.3f}s/step "
              f"pinned={run['pinned_storage_gib']:.3f} GiB "
              f"peak allocated={run['peak_allocated_gib']} GiB", flush=True)
    for item in report["skipped"] + report["errors"]:
        print(f"{item['label']}: {item.get('status', 'skipped')} - {item.get('reason', item.get('error'))}", flush=True)
    print(f"\nReport saved: {report_path.resolve()}", flush=True)
    print("Compare both classic repeats, fast repeats and serialized fast. "
          "Exact matches are strongest; compare numerical drift to the classic repeat, not just training loss.",
          flush=True)
    return int((args.require_parity and not gate["passed"]) or bool(report["errors"]) or any(
        not math.isfinite(row["loss"]) or not math.isfinite(row["validation"])
        or any(not row[key]["finite"] for key in ("gradients", "updates"))
        for run in report["runs"] for row in run["steps"]))


if __name__ == "__main__":
    raise SystemExit(worker(sys.argv[2]) if len(sys.argv) == 3 and sys.argv[1] == "--worker" else main())
