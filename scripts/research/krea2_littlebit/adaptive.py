"""Adaptive frozen Krea2 projection screening; no trainer mutation or archive export."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch


def error(a, b):
    a, b = a.float(), b.float()
    return float((a-b).norm() / b.norm().clamp_min(1e-12))


def probe(fn, rows):
    x, dy = rows
    x = x.detach().requires_grad_(True)
    y = fn(x)
    dx, = torch.autograd.grad(y, x, dy.to(y.dtype))
    return y.detach(), dx.detach()


def scores(fn, rows, ref):
    y, dx = probe(fn, rows)
    return [error(y, ref[0]), error(dx, ref[1])]


def spectrum(w, rank):
    # Randomized subspace spectrum; exact total Frobenius energy, approximate top energy.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        _, s, _ = torch.svd_lowrank(w.float(), q=min(rank, *w.shape), niter=3)
    energy = w.float().square().sum().clamp_min(1e-30)
    return {"sampled_rank": len(s), "captured_energy": float(s.square().sum()/energy),
            "stable_rank_estimate": float(energy/s[0].square().clamp_min(1e-30))}


def adaptive(w, stock, cal, held, codec, geometry, ranks, threshold, group=64):
    """Select using calibration only; held-out failure rejects rather than retunes."""
    m, n = w.shape
    stock_bytes = m*n + 4*m
    cal_ref, held_ref = probe(stock, cal), probe(stock, held)
    raw = lambda x: torch.nn.functional.linear(x, w.to(x.dtype))
    # Diagonal activation weighting approximates E[xx^T] without forming a covariance.
    scale = cal[0].float().square().mean(0).sqrt().clamp_min(1e-6)
    scale = (scale / scale.mean().clamp_min(1e-6)).clamp(0.125, 8).bfloat16()
    plans = [(c, None, False) for c in ("q4", "q6") if n % group == 0]
    for r in sorted(set(ranks)):
        for c in ("littlebit", "littlebit2", "svd"):
            if (c == "svd" and r <= min(m,n)) or (c != "svd" and r % 2 == 0 and r//2 <= min(m,n)):
                plans += [(c, r, False), (c, r, True)]
    records, best = [], None
    for c, r, weighted in plans:
        target = w.float()*scale.float() if weighted else w
        packed, spec = codec.encode(target, c, rank=r or 128, group=group)
        codec.validate(packed, spec)
        # Include the runtime input scale. Do not fold it into binary factors silently.
        size = packed.nbytes + (scale.nbytes if weighted else 0)
        def fn(x):
            return codec.linear(x/scale.to(x.dtype) if weighted else x, packed, spec)
        err = scores(fn, cal, cal_ref)
        row = {"codec": c, "rank": r, "activation_weighted": weighted,
               "bytes": size, "calibration_error": err,
               "calibration_pass": size < stock_bytes and all(e <= threshold for e in err)}
        records.append(row)
        if row["calibration_pass"] and (best is None or size < best[0]["bytes"]):
            best = (row, packed, spec)
    result = {"shape": [m,n], "int8_bytes": stock_bytes,
              "spectrum": spectrum(w, max(ranks)), "candidates": records,
              "int8_error_vs_raw_heldout": scores(stock, held, probe(raw, held)),
              "selected": "int8", "selected_bytes": stock_bytes}
    if best is not None:
        row, packed, spec = best
        weighted = row["activation_weighted"]
        fn = lambda x: codec.linear(x/scale.to(x.dtype) if weighted else x, packed, spec)
        err = scores(fn, held, held_ref)
        result["heldout_candidate"] = dict(row, heldout_error=err)
        if spec["codec"] in ("littlebit", "littlebit2", "svd"):
            a,b = codec.factors(packed, spec)
            if weighted:
                a = a.float()/scale.float()
            met = geometry.metrics(a,b)
            result["heldout_candidate"]["factor_effective_rank"] = float(met["effective_rank"])
        if all(e <= threshold for e in err):
            result["selected"] = row
            result["selected_bytes"] = row["bytes"]
    return result


def main():
    from safetensors import safe_open
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", required=True, type=Path)
    p.add_argument("--raw", required=True, type=Path)
    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--heldout", required=True, type=Path)
    p.add_argument("--key", action="append", help="repeatable; default all eligible keys")
    p.add_argument("--rank", type=int, action="append")
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    ranks = a.rank or [128,256,512]
    if min(ranks) <= 0 or min(a.rows,a.threads) <= 0 or not 0 < a.threshold < 1:
        p.error("invalid rank, row, thread or threshold")
    if a.out.exists():
        p.error("output exists")
    pilot = a.pilot.resolve()
    if not all((pilot/name).is_file() for name in ("compact.py","NOTICE.md","Samsung-LICENSE.txt")):
        p.error("expected external pilot with license notices")
    sys.path.insert(0, str(pilot))
    import compact
    from compress import ELIGIBLE, FORMAT
    from transfer_layer import import_file, stock_layer, sha256
    repo = Path(__file__).resolve().parents[3]
    int8 = import_file("adaptive_native_int8", repo/"src/fizgig/modules/int8_train.py")
    geometry = import_file("adaptive_geometry", repo/"src/fizgig/training/matrix_geometry.py")
    torch.set_num_threads(a.threads)
    hashes = {k:sha256(v) for k,v in (("raw",a.raw),("calibration",a.calibration),("heldout",a.heldout))}
    if hashes["calibration"] == hashes["heldout"]:
        p.error("calibration and held-out archives must differ")
    records = {}
    with safe_open(str(a.raw),framework="pt",device="cpu") as f, safe_open(str(a.calibration),framework="pt",device="cpu") as c, safe_open(str(a.heldout),framework="pt",device="cpu") as h:
        if FORMAT in (f.metadata() or {}) or any(k.endswith((".weight_scale",".scale_weight",".comfy_quant")) for k in f.keys()):
            p.error("expected original Raw BF16 checkpoint")
        keys = list(dict.fromkeys(a.key)) if a.key else sorted(set(f.keys()) & ELIGIBLE)
        if not keys:
            p.error("no eligible keys")
        for k in keys:
            if k not in ELIGIBLE or k not in f.keys():
                p.error(f"unknown eligible key {k}")
            w = f.get_tensor(k)
            if w.dtype != torch.bfloat16 or w.ndim != 2 or not torch.isfinite(w).all():
                p.error(f"{k}: expected finite BF16 matrix")
            def rows(src):
                x,dy = (src.get_slice(k+s)[:a.rows].bfloat16() for s in (".x",".dy"))
                if x.ndim != 2 or dy.ndim != 2 or not len(x) or x.shape != (len(dy),w.shape[1]) or dy.shape[1] != w.shape[0] or not torch.isfinite(x).all() or not torch.isfinite(dy).all():
                    raise ValueError(f"invalid activation rows: {k}")
                return x,dy
            cal,held = rows(c),rows(h)
            # Detect exact row reuse even if archive hashes differ.
            if any(torch.equal(x,y) for x in cal[0] for y in held[0]):
                raise ValueError(f"overlapping calibration/heldout input row: {k}")
            records[k] = adaptive(w,stock_layer(w,int8),cal,held,compact,geometry,ranks,a.threshold)
            print(k, records[k]["selected"], flush=True)
    with a.out.open("x",encoding="utf-8") as f:
        json.dump({"threshold":a.threshold,"hashes":hashes,"records":records,
                   "selected_bytes":sum(r["selected_bytes"] for r in records.values()),
                   "int8_bytes":sum(r["int8_bytes"] for r in records.values()),
                   "scope":"eligible projection weights only; not full-model VRAM"},f,indent=2,allow_nan=False)


if __name__ == "__main__":
    main()
