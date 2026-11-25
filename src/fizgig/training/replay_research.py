"""Opt-in replay research diagnostics; never used by ordinary training."""
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


def probe_before(net, optimizer=None):
    # CPU snapshots only. Read the OLD momentum without creating optimizer state.
    # Actual group LR here includes any temporary per-step LR multiplier.
    rates = ({id(p): float(g["lr"]) * float(g["weight_decay"])
              for g in optimizer.param_groups for p in g["params"]}
             if optimizer is not None else {})
    out = {}
    for name, block, factor, p in factors(net):
        w = p.detach().float().cpu().clone()
        g = p.grad
        gc = g.detach().float().cpu() if g is not None else None
        grad_sq = float(gc.square().sum(dtype=torch.float64)) if gc is not None else None
        detail = None
        if optimizer is not None:
            if id(p) not in rates:
                raise RuntimeError("Research adapter factor missing from optimizer groups")
            decay = rates[id(p)] if gc is not None else 0.0
            buf = optimizer.state.get(p, {}).get("momentum_buffer") if gc is not None else None
            agreement = None
            if buf is not None:
                mc = buf.detach().float().cpu()
                m2 = float(mc.square().sum(dtype=torch.float64))
                dot = float((gc * mc).sum(dtype=torch.float64))
                if grad_sq > 0 and m2 > 0:
                    agreement = (dot, grad_sq, m2)
            detail = (decay, agreement)
        out[name] = (w, grad_sq, detail)
    return out


def probe_after(net, before):
    rows = {}
    extras = {}
    for name, block, factor, p in factors(net):
        w, g2, detail = before[name]
        key = (block, factor)
        r = rows.setdefault(key, [0, 0.0, 0, 0.0, 0.0])
        r[0] += w.numel()
        r[1] += float(w.square().sum(dtype=torch.float64))
        if g2 is not None:
            r[2] += w.numel()
            r[3] += g2
        after = p.detach().float().cpu()
        delta = after - w
        r[4] += float(delta.square().sum(dtype=torch.float64))
        if detail is not None:
            decay, agreement = detail
            e = extras.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0, 0, 0])
            # Algebraic reconstruction, not a new optimizer operation.
            # Includes small floating-point rounding residuals.
            decay_delta = -decay * w
            muon_delta = delta - decay_delta
            e[0] += float(decay_delta.square().sum(dtype=torch.float64))
            e[1] += float(muon_delta.square().sum(dtype=torch.float64))
            e[6] += int(g2 is not None)
            if agreement is not None:
                dot, gg, mm = agreement
                e[2] += dot
                e[3] += gg
                e[4] += mm
                e[5] += 1
    result = []
    for (b, f), (n, w2, ng, g2, d2) in sorted(rows.items()):
        row = {"block": b, "factor": f, "parameters": n,
               "parameter_rms": (w2/n)**0.5, "gradient_rms": (g2/ng)**0.5 if ng else None,
               "update_rms": (d2/n)**0.5}
        if (b, f) in extras:
            wd2, mu2, dot, gg, mm, count, active = extras[(b, f)]
            row.update(decay_update_rms=(wd2/n)**0.5,
                       muon_update_rms_reconstructed=(mu2/n)**0.5,
                       gradient_momentum_cosine=(max(-1.0, min(1.0, dot / (gg * mm)**0.5))
                                                 if count and gg > 0 and mm > 0 else None),
                       momentum_agreement_matrices=count, gradient_bearing_matrices=active)
        result.append(row)
    return result
