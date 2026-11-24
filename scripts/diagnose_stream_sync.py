"""Standalone GPU correctness diagnostic. Does not modify training preferences or caches.

Run inside Fizgig's venv. Default: reduced-width 28-block Krea 2, actual INT8
kernels, checkpointed forward/backward, Muon, 22 swapped blocks. Optional
--model and --cache use the real checkpoint and fixed cached training examples.
No VAE/text encoder or full training run is needed. Never disables block swap
for the full checkpoint. CPU-only mode is for harness smoke tests, not GPU proof.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from fizgig.families import quant
from fizgig.families.lora import FamilyLoRA
from fizgig.krea2.driver import Krea2Driver
from fizgig.krea2.model import SingleMMDiTConfig, SingleStreamDiT
from fizgig.krea2.offloading import LoRAStreamOffloader
from fizgig.training.optimizers import create_optimizer


def inputs(args, config):
    """Freeze the entire input sequence before any of the runs."""
    gen = torch.Generator(device="cpu").manual_seed(args.seed + 100)
    pairs = []
    if args.cache:
        from safetensors.torch import load_file
        from fizgig.dataset import compact_text
        folder = Path(args.cache)
        for path in sorted(folder.glob("*_krea2.safetensors")):
            stem = path.name.rsplit("_", 2)[0]
            te = folder / (stem + "_krea2_te.safetensors")
            if te.is_file():
                pairs.append((path, te))
        if not pairs:
            raise ValueError("No matching Krea 2 latent/text cache pairs in " + str(folder))
    out = []
    for i in range(args.steps + 1):
        if pairs:
            path, te = pairs[i % len(pairs)]
            data = load_file(str(path))
            keys = [k for k in data if k.startswith("latent_") and data[k].ndim == 3]
            if len(keys) != 1:
                raise ValueError(f"Expected one still latent in {path}, found {keys}")
            latent = data[keys[0]].unsqueeze(0)
            text = compact_text.load(str(te))
            cond = {k: text[k] for k in ("hidden_states", "attention_mask")}
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


def build(args, device, mode):
    driver = Krea2Driver()
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
        if not isinstance(model.offloader, LoRAStreamOffloader) or model.offloader.B != 2:
            raise RuntimeError("Fast mode must actually use the two-slot send-only offloader")
    elif mode != "resident" and isinstance(model.offloader, LoRAStreamOffloader):
        raise RuntimeError("Classic mode unexpectedly uses send-only")
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


def execute(args, device, mode, label, batches, reference_dir, record):
    print(f"\n=== {label} ({mode}) ===", flush=True)
    driver, model, params, opt = build(args, device, mode)
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
        start = time.perf_counter()
        loss = forward(driver, model, batches[i], device)
        loss.backward()
        # No diagnostic synchronization inside normal forward or backward.
        # Capture gradients after optimizer.step (Muon/AdamW leave .grad intact).
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
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
                   seconds=elapsed, gradient_norm=sum(float(torch.linalg.vector_norm(g.float())) ** 2
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
              f"seconds={elapsed:.3f}", flush=True)
        rows.append(row)
        del grads, updates, after, loss
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    if mode != "resident":
        model.offloader.remove_hooks()
        # Drop diagnostic closures that reference the offloader itself.
        if mode == "fast_sync" and device.type == "cuda":
            del model.offloader._load, model.offloader.wait_for_block
    return {"mode": mode, "label": label, "steps": rows, "peak_allocated_gib": peak}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", help="Optional real Krea 2 safetensors checkpoint")
    p.add_argument("--cache", help="Optional real Krea 2 cache directory (requires --model)")
    p.add_argument("--steps", type=int, default=3)
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
    print(json.dumps({k: v for k, v in report.items() if k not in ("runs", "errors")}, indent=2), flush=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    modes = [("classic", "A_classic_reference"), ("classic", "D_classic_repeat"),
             ("fast", "B_fast"), ("fast", "C_fast_repeat"), ("fast_sync", "E_fast_serialized")]
    if not args.model:
        modes.append(("resident", "F_no_swap"))
    with tempfile.TemporaryDirectory(prefix="fizgig-stream-sync-", dir=args.scratch) as tmp:
        for idx, (mode, label) in enumerate(modes):
            try:
                run = execute(args, device, mode, label, batches, Path(tmp), idx == 0)
                report["runs"].append(run)
            except Exception as e:
                report["errors"].append({"label": label, "error": str(e), "traceback": traceback.format_exc()})
                traceback.print_exc()
                # Without the reference, subsequent comparisons are meaningless.
                if idx == 0:
                    break
            finally:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                report_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print(f"\nReport saved: {report_path.resolve()}", flush=True)
    print("Compare both classic repeats, fast repeats and serialized fast. "
          "Exact matches are strongest; compare numerical drift to the classic repeat, not just training loss.",
          flush=True)
    return int(bool(report["errors"]) or any(not row[key]["finite"]
               for run in report["runs"] for row in run["steps"] for key in ("gradients", "updates")))


if __name__ == "__main__":
    raise SystemExit(main())
