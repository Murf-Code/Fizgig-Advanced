"""Opt-in Experiment 5A diagnostics; never used by ordinary training."""
import contextlib
import hashlib
import json
import os
import random

import numpy as np
import torch


def config(options):
    mode = (options or {}).get("replay_research")
    if mode is not None and mode not in ("normal", "strict", "telemetry", "causal"):
        raise ValueError("Unknown replay research mode")
    return mode


@contextlib.contextmanager
def execution(mode):
    if mode is None or mode == "normal":
        yield
        return
    old = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    det, bench = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    env = {k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "FIZGIG_SDPA_BACKEND")}
    try:
        if torch.cuda.is_initialized():
            raise RuntimeError("Strict replay research must start before CUDA initialization")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["FIZGIG_SDPA_BACKEND"] = "default"
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        torch.use_deterministic_algorithms(old, warn_only=warn)
        torch.backends.cudnn.deterministic = det
        torch.backends.cudnn.benchmark = bench
        for key, value in env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))


def record(logger, **row):
    logger.info("[replay-research] %s", json.dumps(row, sort_keys=True, allow_nan=False))


def environment(logger, mode):
    record(logger, kind="environment", mode=mode, torch=str(torch.__version__),
           cuda=torch.version.cuda, deterministic=torch.are_deterministic_algorithms_enabled(),
           cudnn_deterministic=torch.backends.cudnn.deterministic,
           cudnn_benchmark=torch.backends.cudnn.benchmark,
           workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
           sdpa_policy=os.environ.get("FIZGIG_SDPA_BACKEND", "auto"),
           flash_enabled=torch.backends.cuda.flash_sdp_enabled(),
           efficient_enabled=torch.backends.cuda.mem_efficient_sdp_enabled(),
           math_enabled=torch.backends.cuda.math_sdp_enabled(),
           cudnn_enabled=torch.backends.cuda.cudnn_sdp_enabled() if hasattr(torch.backends.cuda, "cudnn_sdp_enabled") else None,
           python_hash_seed=os.environ.get("PYTHONHASHSEED"))


def factors(net):
    from fizgig.families.lora import TRAINABLE
    for name, wrapped in sorted(net.wrapped.items()):
        if TRAINABLE not in wrapped.adapters:
            continue
        a, b = wrapped.adapters[TRAINABLE]
        block = int(name.split(".")[1])
        yield name + ".A", block, "A", a.weight
        yield name + ".B", block, "B", b.weight


def fingerprint(net):
    h = hashlib.sha256()
    for name, _, _, p in factors(net):
        cpu = p.detach().cpu().contiguous()
        h.update(json.dumps([name, str(cpu.dtype), list(cpu.shape)]).encode())
        h.update(memoryview(cpu.view(torch.uint8).numpy()).cast("B"))
    return h.hexdigest()


def active_mask(net):
    return sorted({block for _, block, _, p in factors(net) if p.grad is not None})


def probe_before(net):
    # CPU snapshots, not dense weights or extra long-lived GPU tensors.
    out = {}
    for name, block, factor, p in factors(net):
        w = p.detach().float().cpu().clone()
        g = p.grad
        grad_sq = float(g.detach().float().cpu().square().sum(dtype=torch.float64)) if g is not None else None
        out[name] = (w, grad_sq)
    return out


def probe_after(net, before):
    rows = {}
    for name, block, factor, p in factors(net):
        w, g2 = before[name]
        r = rows.setdefault((block, factor), [0, 0.0, 0, 0.0, 0.0])
        r[0] += w.numel()
        r[1] += float(w.square().sum(dtype=torch.float64))
        if g2 is not None:
            r[2] += w.numel()
            r[3] += g2
        r[4] += float((p.detach().float().cpu() - w).square().sum(dtype=torch.float64))
    return [{"block": b, "factor": f, "parameters": n,
             "parameter_rms": (w2/n)**0.5, "gradient_rms": (g2/ng)**0.5 if ng else None,
             "update_rms": (d2/n)**0.5} for (b, f), (n, w2, ng, g2, d2) in sorted(rows.items())]
