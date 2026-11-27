"""Research-only AdamW effective movement comparison."""
import hashlib
import json
import math
import statistics
from pathlib import Path

from fizgig.web.selective_optimizer_smoke import COMMON, EXPECTED_UPDATES, _write

START = {
    "adamw": .00025, "adamw8bit": .00025, "pagedadamw8bit": .00025,
    "muon": .00045, "ademamix8bit": .00025, "pagedademamix8bit": .00025,
    "lion8bit": .000025, "soap-512": .00025, "soap-1024": .00025,
    "soap-compact32": .00025, "soap-compact64": .00025, "soap-compact128": .00025,
    "soap-chunked": .00025, "soap-offloaded": .00025, "soap": .00025,
    "polora": .02, "lora_tsd": .005,
    "prodigy": 1., "automagic3": .00025, "automagic4": .00025,
}
START["adamw-repeat"] = START["adamw"]
PILOT = ("adamw", "adamw-repeat", "soap-512", "soap-compact32")
SOAP_CAPS = {"soap-512": 512, "soap-1024": 1024}


def optimizer_name(case):
    return "soap" if case.startswith("soap-") else "adamw" if case == "adamw-repeat" else case


SOAP_ARGS = {
    **{case: f"max_precond_dim={cap}" for case, cap in SOAP_CAPS.items()},
    "soap-chunked": "memory_mode=chunked chunk_rows=128",
    "soap-offloaded": "memory_mode=offloaded",
    **{f"soap-compact{k}": f"memory_mode=compact compact_rank={k}" for k in (32, 64, 128)},
}


SPECIAL = {"prodigy", "automagic3", "automagic4", "polora", "lora_tsd"}
MANIFEST = "optimizer-match-manifest.json"
REPORT = "optimizer-match-report.json"
AUDIT = "optimizer-match-audit.json"


def settings(group, previous=None, cases=PILOT):
    group = Path(group)
    rates = dict(START)
    if previous:
        for r in previous["runs"]:
            name = r["case"]
            if name in rates and name not in SPECIAL and name not in ("adamw", "adamw-repeat"):
                proposed = r.get("suggested_lr")
                if proposed is not None and math.isfinite(proposed) and proposed > 0:
                    rates[name] = proposed
    return [{**COMMON, "LEARNING_RATE": str(lr), "OPTIMIZER_TYPE": optimizer_name(name),
             "OPTIMIZER_ARGS": (f"weight_decay=0.01 {SOAP_ARGS[name]}" if name in SOAP_ARGS
                                else "" if name in SPECIAL else "weight_decay=0.01"),
             "KREA2_SELECTIVE_BLOCK_CHECK": False, "KREA2_SELECTIVE_COMPENSATION": False,
             "__optimizer_match": name, "LORA_NAME": f"{group.name}-{name}",
             "LORA_OUTPUT_DIR": str(group / name)} for name in cases for lr in (rates[name],)]


def attach(plan, values, family, resume=""):
    name = values["__optimizer_match"]
    expected = next((r for r in settings(Path(values["LORA_OUTPUT_DIR"]).parent)
                     if r["__optimizer_match"] == name), None)
    if not expected or family != "krea2" or resume or values.get("RESUME_TRAINING"):
        raise ValueError("Optimizer match needs a fresh Krea 2 run")
    lr = float(values["LEARNING_RATE"])
    if not math.isfinite(lr) or lr <= 0 or (name in ("adamw", "adamw-repeat") and lr != START[name]):
        raise ValueError("Invalid calibration rate")
    if any(values.get(k) != v for k, v in expected.items() if k != "LEARNING_RATE"):
        raise ValueError("Optimizer match recipe changed")
    if not plan.problems:
        stages = [s for s in plan.stages if s.name == "Training"]
        if len(stages) != 1:
            raise ValueError("Optimizer match needs one Training stage")
        stages[0].cmd.extend(["--family_option", "optimizer_match=" + name])
    return plan


def initialize(group, dataset, values):
    group = Path(group)
    group.mkdir(parents=True, exist_ok=False)
    _write(group / MANIFEST, {"dataset": str(dataset), "settings": values,
                             "expected_updates": EXPECTED_UPDATES})
    return refresh(group)


def require_recipe(checks):
    failures = [name for name, ok in checks.items() if not ok]
    if failures:
        raise RuntimeError("Optimizer match requires the fixed fresh one-epoch recipe; "
                           "failed checks: " + ", ".join(failures))


def require_holdout(holdout_list, val_held):
    """The web launcher uses TOML holdout_file, not the retired CLI flag."""
    paths = (holdout_list or {}).get("paths") or []
    if len(paths) != 2 or len(set(paths)) != 2 or len(val_held) != 2:
        raise RuntimeError("Optimizer research requires two distinct actual held-out "
                           f"images with validation caches; paths={len(paths)}, "
                           f"cached_items={len(val_held)}")


def input_hash(*items):
    """Hash actual tensor bytes, dtype, shape and structured input identity."""
    import torch
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            y = x.detach().cpu().contiguous()
            h.update(repr((str(y.dtype), tuple(y.shape))).encode())
            h.update(y.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x):
                visit(k)
                visit(x[k])
        elif isinstance(x, (tuple, list)):
            h.update(str(len(x)).encode())
            for v in x: visit(v)
        else:
            h.update(repr(x).encode())
        h.update(b"\\0")
    for item in items: visit(item)
    return h.hexdigest()


def optimizer_manifest(optimizer, case):
    import torch
    from fizgig.training.soap import SOAP
    from fizgig.training.soap_memory import CompactSOAP
    expected = {"adamw": torch.optim.AdamW, "adamw-repeat": torch.optim.AdamW,
                "soap-512": SOAP, "soap-compact32": CompactSOAP}[case]
    if type(optimizer) is not expected:
        raise RuntimeError(f"Unexpected optimizer for {case}: {type(optimizer)}")
    groups, persistent = [], 0
    for group in optimizer.param_groups:
        shapes = [list(p.shape) for p in group["params"]]
        cfg = {k: v for k,v in group.items() if k != "params"
               and isinstance(v, (str, int, float, bool, tuple, list, type(None)))}
        if case == "soap-512" and group["max_precond_dim"] != 512:
            raise RuntimeError("SOAP cap mismatch")
        for p in group["params"]:
            persistent += 8*p.numel()
            if case == "soap-512":
                persistent += sum(8*d*d for d in p.shape if d <= 512)
            elif case == "soap-compact32":
                if optimizer.compact_rank != 32:
                    raise RuntimeError("Compact SOAP rank mismatch")
                persistent += sum(8*d*min(32,d)+4*min(32,d) for d in p.shape)
        groups.append({"config": cfg, "shapes": shapes})
    return {"class": type(optimizer).__name__, "groups": groups,
            "estimated_persistent_bytes": persistent,
            "compact_rank": getattr(optimizer, "compact_rank", None)}


def checkpoint(output, case, **fields):
    p = Path(output) / AUDIT
    old = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"case": case}
    _write(p, {**old, **fields})


def snapshot(net):
    from fizgig.families.lora import TRAINABLE
    import torch
    result = {}
    for name, wrapped in sorted(net.wrapped.items()):
        if TRAINABLE not in wrapped.adapters:
            continue
        a, b = wrapped.adapters[TRAINABLE]
        result[name] = (a.weight.detach().to(device="cpu", dtype=torch.float32, copy=True),
                        b.weight.detach().to(device="cpu", dtype=torch.float32, copy=True),
                        float(wrapped.scales[TRAINABLE]))
    if not result:
        raise RuntimeError("No LoRA factors for optimizer movement measurement")
    return result


def fingerprint(before):
    h = hashlib.sha256()
    for name, (a, b, scale) in before.items():
        h.update(repr((name, tuple(a.shape), tuple(b.shape), scale)).encode())
        h.update(a.contiguous().numpy().tobytes())
        h.update(b.contiguous().numpy().tobytes())
    return h.hexdigest()


def delta_sq(a0, b0, a1, b1, scale=1.):
    """Frobenius norm in compact factor space, including the cross term."""
    import torch
    # Stable difference factorization: B1(A1-A0)+(B1-B0)A0.
    left = torch.cat((b1.double(), b1.double()-b0.double()), dim=1)
    right = torch.cat((a1.double()-a0.double(), a0.double()), dim=0)
    terms = (left.T @ left) * (right @ right.T)
    sq = float(terms.sum()) * scale**2
    if not math.isfinite(sq):
        raise RuntimeError("Nonfinite effective LoRA movement")
    if sq <= 1e-12 * float(terms.abs().sum()) * scale**2:
        _, rl = torch.linalg.qr(left, mode="reduced")
        _, rr = torch.linalg.qr(right.T, mode="reduced")
        sq = float((rl @ rr.T).square().sum()) * scale**2
    return sq


def movement(before, net):
    from fizgig.families.lora import TRAINABLE
    import torch
    blocks = {}
    for name, (a0, b0, scale) in before.items():
        a, b = net.wrapped[name].adapters[TRAINABLE]
        a1 = a.weight.detach().to(device="cpu", dtype=torch.float32)
        b1 = b.weight.detach().to(device="cpu", dtype=torch.float32)
        sq = delta_sq(a0, b0, a1, b1, scale)
        block = name.split(".")[1]
        blocks[block] = blocks.get(block, 0.) + sq
    return {"norm": math.sqrt(sum(blocks.values())),
            "blocks": {k: math.sqrt(v) for k, v in blocks.items()}}


def finish(output, name, implementation, lr, rows, validation, initial):
    output = Path(output)
    if output.name != name or name not in START:
        raise ValueError("Invalid optimizer match output")
    status = ("measured" if len(rows) == EXPECTED_UPDATES and [r["update"] for r in rows] == list(range(1, EXPECTED_UPDATES+1))
              and validation and validation[-1]["step"] == EXPECTED_UPDATES
              and all(math.isfinite(r["norm"]) for r in rows) else "incomplete")
    checkpoint(output, name, implementation=implementation, native_lr=lr,
               status=status, steps=rows, validation=validation, initial_fingerprint=initial)
    return refresh(output.parent)


def record_terminal(output, case, status, reason, progress=None):
    output = Path(output)
    if case not in START or output.name != case or not (output.parent / MANIFEST).is_file():
        return
    checkpoint(output, case, status=status, reason=reason, progress=progress)
    refresh(output.parent)


def refresh(group):
    group = Path(group)
    manifest = json.loads((group / MANIFEST).read_text(encoding="utf-8"))
    runs = []
    for name in [r["__optimizer_match"] for r in manifest["settings"]]:
        p = group / name / AUDIT
        runs.append(json.loads(p.read_text(encoding="utf-8")) if p.is_file()
                    else {"case": name, "status": "not_completed"})
    ref = runs[0]
    repeat = next((r for r in runs if r["case"] == "adamw-repeat"), {})
    def paired(a,b):
        return (a.get("initial_fingerprint") == b.get("initial_fingerprint")
                and bool(a.get("initial_fingerprint"))
                and len(a.get("steps", [])) == len(b.get("steps", [])) == EXPECTED_UPDATES
                and all(x.get("input_hash") and x["input_hash"] == y.get("input_hash")
                        for x,y in zip(a["steps"], b["steps"])))
    repeat_ok = (ref["status"] == repeat.get("status") == "measured" and paired(ref,repeat)
                 and all(math.isclose(x["norm"], y["norm"], rel_tol=.01, abs_tol=1e-9)
                         for x,y in zip(ref["steps"], repeat["steps"])))
    ref_ok = repeat_ok
    for r in runs:
        if r["status"] != "measured" or not ref_ok:
            continue
        if r["initial_fingerprint"] != ref["initial_fingerprint"]:
            r["status"] = "initialization_mismatch"
            continue
        if not paired(ref,r):
            r["status"] = "input_mismatch"
            continue
        r["paired_ratios"] = [x["norm"]/y["norm"] if y["norm"] > 0 else None
                              for x,y in zip(r["steps"],ref["steps"])]
        r["post_refresh_updates"] = r["steps"][11:13]
        # Skip first update: SOAP may initialize its preconditioner without moving.
        base = statistics.median(x["norm"] for x in ref["steps"][1:8])
        value = statistics.median(x["norm"] for x in r["steps"][1:8])
        vals = [x["norm"] for x in r["steps"][1:]]
        qs = statistics.quantiles(vals, n=10, method="inclusive")
        r["movement_summary"] = {"p10": qs[0], "median": statistics.median(vals),
                                 "p90": qs[-1], "max": max(vals),
                                 "post_warmup_median": statistics.median(vals[7:])}
        r["pilot_only"] = True
        if base <= 0 or value <= 0:
            r["status"] = "zero_movement"
            continue
        ratio = value / base
        r["ratio"] = ratio
        r["status"] = "matched" if .9 <= ratio <= 1.1 else "not_matched"  # pilot only; independent seed required
        if r["case"] not in SPECIAL:
            r["suggested_lr"] = r["native_lr"] if r["status"] == "matched" else r["native_lr"] * min(4., max(.25, 1. / ratio))
        else:
            r["specialized_review"] = True
    report = {**manifest, "runs": runs, "adamw_repeat_verified": repeat_ok,
              "deferred": {"soap-1024": "duplicate axes on expected recipe",
                           "soap": "estimated 88.4 GiB GPU state",
                           "soap-chunked": "same persistent state; parity unresolved",
                           "soap-offloaded": "requires approximately 91.5 GiB available RAM",
                           "soap-compact64": "conditional on compact32 results",
                           "soap-compact128": "conditional on compact64 results"}}
    _write(group / REPORT, report)
    return report
