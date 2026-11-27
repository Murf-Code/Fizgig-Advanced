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
    "lion8bit": .000025, "soap": .00025, "polora": .02, "lora_tsd": .005,
    "prodigy": 1., "automagic3": .00025, "automagic4": .00025,
}
SPECIAL = {"prodigy", "automagic3", "automagic4", "polora", "lora_tsd"}
MANIFEST = "optimizer-match-manifest.json"
REPORT = "optimizer-match-report.json"
AUDIT = "optimizer-match-audit.json"


def settings(group, previous=None):
    group = Path(group)
    rates = dict(START)
    if previous:
        for r in previous["runs"]:
            name = r["case"]
            if name in rates and name not in SPECIAL and name != "adamw":
                proposed = r.get("suggested_lr")
                if proposed is not None and math.isfinite(proposed) and proposed > 0:
                    rates[name] = proposed
    return [{**COMMON, "LEARNING_RATE": str(lr), "OPTIMIZER_TYPE": name,
             "OPTIMIZER_ARGS": "" if name in SPECIAL else "weight_decay=0.01",
             "KREA2_SELECTIVE_BLOCK_CHECK": False, "KREA2_SELECTIVE_COMPENSATION": False,
             "__optimizer_match": name, "LORA_NAME": f"{group.name}-{name}",
             "LORA_OUTPUT_DIR": str(group / name)} for name, lr in rates.items()]


def attach(plan, values, family, resume=""):
    name = values["__optimizer_match"]
    expected = next((r for r in settings(Path(values["LORA_OUTPUT_DIR"]).parent)
                     if r["__optimizer_match"] == name), None)
    if not expected or family != "krea2" or resume or values.get("RESUME_TRAINING"):
        raise ValueError("Optimizer match needs a fresh Krea 2 run")
    lr = float(values["LEARNING_RATE"])
    if not math.isfinite(lr) or lr <= 0 or (name == "adamw" and lr != START[name]):
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
    sq = float(((left.T @ left) * (right @ right.T)).sum()) * scale**2
    if not math.isfinite(sq):
        raise RuntimeError("Nonfinite effective LoRA movement")
    if sq < 0:
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
    status = ("measured" if len(rows) == 8 and [r["update"] for r in rows] == list(range(1, 9))
              and validation and validation[-1]["step"] == EXPECTED_UPDATES
              and all(math.isfinite(r["norm"]) for r in rows) else "incomplete")
    _write(output / AUDIT, {"case": name, "implementation": implementation, "native_lr": lr,
                           "status": status, "steps": rows, "validation": validation,
                           "initial_fingerprint": initial})
    return refresh(output.parent)


def refresh(group):
    group = Path(group)
    manifest = json.loads((group / MANIFEST).read_text(encoding="utf-8"))
    runs = []
    for name in START:
        p = group / name / AUDIT
        runs.append(json.loads(p.read_text(encoding="utf-8")) if p.is_file()
                    else {"case": name, "status": "not_completed"})
    ref = runs[0]
    ref_ok = ref["status"] == "measured"
    for r in runs:
        if r["status"] != "measured" or not ref_ok:
            continue
        if r["initial_fingerprint"] != ref["initial_fingerprint"]:
            r["status"] = "initialization_mismatch"
            continue
        # Skip first update: SOAP may initialize its preconditioner without moving.
        base = statistics.median(x["norm"] for x in ref["steps"][1:])
        value = statistics.median(x["norm"] for x in r["steps"][1:])
        if base <= 0 or value <= 0:
            r["status"] = "zero_movement"
            continue
        ratio = value / base
        r["ratio"] = ratio
        r["status"] = "matched" if .9 <= ratio <= 1.1 else "not_matched"
        if r["case"] not in SPECIAL:
            r["suggested_lr"] = r["native_lr"] if r["status"] == "matched" else r["native_lr"] * min(4., max(.25, 1. / ratio))
        else:
            r["specialized_review"] = True
    report = {**manifest, "runs": runs}
    _write(group / REPORT, report)
    return report
