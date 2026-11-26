"""Report full-size stock-versus-stock reproducibility without a quality gate."""
import argparse
import json
from pathlib import Path
import torch
from safetensors import safe_open
from fit_queue import atomic_json


def compare(a,b):
    with safe_open(str(a),framework="pt",device="cpu") as f,safe_open(str(b),framework="pt",device="cpu") as g:
        if set(f.keys())!=set(g.keys()): raise ValueError("repeat inventory mismatch")
        rows={}
        for k in f.keys():
            x,y=f.get_tensor(k),g.get_tensor(k)
            if x.shape!=y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all(): raise ValueError("bad repeat tensor")
            d=float((x.float()-y.float()).norm()); ref=float(x.float().norm())
            rows[k]={"bitwise":torch.equal(x,y),"difference_norm":d,"reference_norm":ref,
                     "relative_l2":d/ref if ref else None}
        return rows


def main():
    p=argparse.ArgumentParser()
    for name in ("stock","repeat","out"): p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    st=json.loads((a.stock/"trace.json").read_text()); rt=json.loads((a.repeat/"trace.json").read_text())
    for key in ("initial_sha256","expected_modules","completed_updates"):
        if st[key]!=rt[key]: raise ValueError("stock repeat mismatch "+key)
    if [s["input_sha256"] for s in st["steps"]]!=[s["input_sha256"] for s in rt["steps"]]:
        raise ValueError("different repeat inputs")
    if set(st["checkpoints"])!=set(rt["checkpoints"]) or len(st["calibration"])!=len(rt["calibration"]):
        raise ValueError("different repeat horizons")
    captures={}
    for s,r in zip(st["calibration"],rt["calibration"]):
        if s["step"]!=r["step"] or s["tokens"]!=r["tokens"]: raise ValueError("different repeat sample rows")
        captures[str(s["step"])]=compare(a.stock/s["file"],a.repeat/r["file"])
    horizons={}
    for step in st["checkpoints"]:
        name=f"task-grad-{int(step):06d}.safetensors"
        horizons[step]={"gradients":compare(a.stock/name,a.repeat/name),
                        "exports":compare(a.stock/st["checkpoints"][step],a.repeat/rt["checkpoints"][step])}
    report={"status":"completed","captures":captures,"horizons":horizons,
            "loss_differences":[r["loss"]-s["loss"] for s,r in zip(st["steps"],rt["steps"])],
            "note":"Report repeatability; finite nonzero CUDA differences are not a learning failure."}
    atomic_json(a.out,report)


if __name__=="__main__":
    main()
