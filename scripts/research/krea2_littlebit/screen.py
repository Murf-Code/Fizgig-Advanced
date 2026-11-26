"""CPU projection screening for an external, separately licensed LittleBit pilot."""
import argparse
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open


def rms(x):
    return x.float().square().mean().sqrt().item()


def compare(actual, reference):
    a, b = actual.float().flatten(), reference.float().flatten()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("nonfinite probe output")
    denom = b.norm().item()
    return {
        "relative_error": (a - b).norm().item() / denom if denom else None,
        "cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        if denom and a.norm().item() else None,
    }


def evaluate(fn, x, dy):
    x = x.detach().requires_grad_(True)
    y = fn(x)
    if y.shape != dy.shape:
        raise ValueError("output and cotangent shape mismatch")
    dx, = torch.autograd.grad(y, x, dy.to(y.dtype))
    return y.detach(), dx.detach()


def read_rows(path, key, m, n, rows, seed):
    if path:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            x, dy = f.get_tensor(key + ".x"), f.get_tensor(key + ".dy")
        if x.ndim != 2 or dy.ndim != 2 or x.shape != (len(dy), n) or dy.shape[1] != m:
            raise ValueError(f"{key}: invalid activation shapes")
        if not len(x) or not x.is_floating_point() or not dy.is_floating_point():
            raise ValueError("expected nonempty floating activation rows")
        return x[:rows].bfloat16(), dy[:rows].bfloat16()
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, n, generator=g).bfloat16(),
            torch.randn(rows, m, generator=g).bfloat16())


def teacher_delta(path, key, x):
    stem = "lora_unet_" + key.removesuffix(".weight").replace(".", "_")
    with safe_open(str(path), framework="pt", device="cpu") as f:
        names = set(f.keys())
        if stem + ".lora_down.weight" not in names:
            return None
        a = f.get_tensor(stem + ".lora_down.weight").float()
        b = f.get_tensor(stem + ".lora_up.weight").float()
        alpha = float(f.get_tensor(stem + ".alpha"))
    return rms(((x.float() @ a.T) @ b.T) * (alpha / len(a)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", required=True, type=Path)
    p.add_argument("--raw", required=True, type=Path)
    p.add_argument("--key", action="append", required=True)
    p.add_argument("--rank", action="append", type=int)
    p.add_argument("--qat-steps", default=0, type=int)
    p.add_argument("--rows", default=128, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--threads", default=4, type=int)
    p.add_argument("--activations", type=Path)
    p.add_argument("--teacher", type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    if a.out.exists():
        p.error("refusing to overwrite output")
    if min(a.rows, a.threads) <= 0 or a.qat_steps < 0:
        p.error("invalid row, thread or QAT count")
    ranks = a.rank or [128, 512]
    if any(r < 2 or r % 2 for r in ranks):
        p.error("LittleBit ranks must be positive even integers")
    repo = Path(__file__).resolve().parents[3]
    pilot = a.pilot.resolve()
    if not all((pilot / name).is_file() for name in
               ("compact.py", "transfer_layer.py", "NOTICE.md", "Samsung-LICENSE.txt")):
        p.error("expected extracted external pilot with licensing notices")
    sys.path.insert(0, str(pilot))
    from compact import encode, linear, validate
    from transfer_layer import import_file, stock_layer, sha256
    from compress import ELIGIBLE, FORMAT
    int8_path = repo / "src/fizgig/modules/int8_train.py"
    int8 = import_file("screen_fizgig_int8", int8_path)
    torch.set_num_threads(a.threads)
    records = []
    with safe_open(str(a.raw), framework="pt", device="cpu") as f:
        if FORMAT in (f.metadata() or {}) or any(k.endswith((".weight_scale", ".scale_weight", ".comfy_quant")) for k in f.keys()):
            p.error("expected original unquantized Raw BF16")
        for key in dict.fromkeys(a.key):
            if key not in ELIGIBLE or key not in f.keys():
                p.error(f"not an eligible Krea2 block projection: {key}")
            w = f.get_tensor(key)
            if w.dtype != torch.bfloat16 or w.ndim != 2 or not torch.isfinite(w).all():
                p.error(f"{key}: expected finite Raw BF16 matrix")
            m, n = w.shape
            x, dy = read_rows(a.activations, key, m, n, a.rows, a.seed)
            raw = evaluate(lambda v: torch.nn.functional.linear(v, w), x, dy)
            native = stock_layer(w, int8)
            stock = evaluate(native, x, dy)
            stock_bytes = native.weight.nbytes + native._int8_wscale.nbytes
            delta = teacher_delta(a.teacher, key, x) if a.teacher else None
            plan = [("q4", None), ("q6", None)]
            plan += [(c, r) for r in ranks if r // 2 <= min(m, n)
                     for c in ("littlebit", "littlebit2")]
            for codec, rank in plan:
                packed, spec = encode(w, codec, rank=rank or 128,
                                      qat_steps=a.qat_steps if rank else 0)
                validate(packed, spec)
                y, dx = evaluate(lambda v: linear(v, packed, spec), x, dy)
                drift = rms(y.float() - stock[0].float())
                record = {
                    "key": key, "codec": codec, "rank": rank, "shape": [m, n],
                    "rows": len(x), "encoded_bytes": packed.nbytes,
                    "int8_bytes": stock_bytes, "storage_fraction": packed.nbytes / stock_bytes,
                    "forward_vs_int8": compare(y, stock[0]),
                    "input_gradient_vs_int8": compare(dx, stock[1]),
                    "forward_vs_raw": compare(y, raw[0]),
                    "input_gradient_vs_raw": compare(dx, raw[1]),
                    "int8_forward_vs_raw": compare(stock[0], raw[0]),
                    "int8_input_gradient_vs_raw": compare(stock[1], raw[1]),
                    "output_drift_over_teacher_rms": drift / delta if delta else None,
                }
                records.append(record)
                print(json.dumps(record, allow_nan=False), flush=True)
                del packed, y, dx
            del native, stock, raw, w
    report = {
        "mode": "supplied_activation_rows" if a.activations else "synthetic_diagnostic",
        "raw": str(a.raw.resolve()), "raw_bytes": a.raw.stat().st_size,
        "raw_sha256": sha256(a.raw), "int8_source_sha256": sha256(int8_path),
        "activation_sha256": sha256(a.activations) if a.activations else None,
        "teacher_sha256": sha256(a.teacher) if a.teacher else None,
        "pilot_compact_sha256": sha256(pilot / "compact.py"),
        "pilot_littlebit_sha256": sha256(pilot / "littlebit.py"),
        "torch": torch.__version__, "seed": a.seed, "qat_steps": a.qat_steps,
        "records": records,
    }
    with a.out.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
