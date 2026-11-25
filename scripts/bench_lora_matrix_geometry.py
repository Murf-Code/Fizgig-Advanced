"""Standalone synthetic LoRA optimizer screening (no model downloads).

Run from the repository root:
  python scripts/bench_lora_matrix_geometry.py --device cpu
  python scripts/bench_lora_matrix_geometry.py --device cuda --rank 64
Results are synthetic, not diffusion quality or fair LR calibration claims.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fizgig.training.matrix_geometry import metrics, delta_norm, balanced_factors
from fizgig.training.muon import BatchedMuon
from fizgig.training.lora_tsd import LoRATSD
from fizgig.training.polora.optim import Polora


def validate(device):
    g = torch.Generator(device=device).manual_seed(123)
    for dtype in (torch.float64, torch.float32, torch.bfloat16):
        for m,n,r in ((43,57,8),(5,7,9),(19,23,6)):
            a = torch.randn(r,n,device=device,generator=g).to(dtype)
            b = torch.randn(m,r,device=device,generator=g).to(dtype)
            if r == 6:
                b[:,1:] = 0
            work = torch.float64 if dtype == torch.float64 else torch.float32
            a,b = a.to(work),b.to(work)
            full = b @ a
            tol = 1e-10 if work == torch.float64 else 2e-5
            s = metrics(a,b)["singular_values"]
            torch.testing.assert_close(s, torch.linalg.svdvals(full)[:s.numel()], rtol=tol, atol=tol*full.norm().item())
            an,bn = balanced_factors(a,b)
            torch.testing.assert_close(bn@an,full,rtol=tol,atol=tol*full.norm().item())
            torch.testing.assert_close(bn.T@bn,an@an.T,rtol=tol,atol=tol*full.norm().item())
            q = torch.linalg.qr(torch.randn(r,r,device=device,dtype=work,generator=g))[0]
            torch.testing.assert_close((b@q.T)@(q@a),full,rtol=tol,atol=tol*full.norm().item())
            a1,b1 = (a+0.003).to(dtype).to(work),(b-0.004).to(dtype).to(work)
            ref = (b1@a1-full).norm()
            torch.testing.assert_close(delta_norm(a,b,a1,b1),ref,rtol=1e-3,atol=tol*full.norm().item())
            z = torch.zeros_like(b)
            assert metrics(a,z)["effective_rank"].item() == 0
    return "passed"


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def serial_metrics(a,b):
    return {k: (v.detach().cpu().tolist() if v.ndim else
                (float(v) if torch.isfinite(v) else None))
            for k,v in metrics(a,b).items()}


def make_optimizer(name,a,b,lr):
    if name == "adamw":
        return torch.optim.AdamW([a,b],lr=lr,weight_decay=0)
    if name == "muon":
        return BatchedMuon([a,b],lr=lr,weight_decay=0,adjust_lr_fn="match_rms_adamw")
    if name == "polora":
        return Polora(pairs=[(a,b)],lr=lr,compile=False,magnitude_scale=1.0)
    return LoRATSD([(a,b)],lr=lr,balance=name=="tsd-balanced",log_every=0)


def run(name, a0, b0, x, y, xv, yv, lr, steps):
    a,b = (torch.nn.Parameter(t.clone()) for t in (a0,b0))
    opt = make_optimizer(name,a,b,lr)
    device=a.device
    sync(device)
    if device.type=="cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter()
    update_time=0.0
    snapshots=[]
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss=(x@a.T@b.T-y).square().mean()
        loss.backward()
        sync(device)
        t=time.perf_counter()
        opt.step()
        sync(device)
        update_time+=time.perf_counter()-t
        if step==0 or (step+1)%max(1,steps//5)==0 or step+1==steps:
            with torch.no_grad():
                val=(xv@a.T@b.T-yv).square().mean()
                if not torch.isfinite(val):
                    raise FloatingPointError("nonfinite validation loss")
                snapshots.append({"step":step+1,"validation_loss":float(val),
                                  "geometry":serial_metrics(a,b)})
    sync(device)
    return {"optimizer":name,"lr":lr,"seconds_including_diagnostics":time.perf_counter()-start,
            "optimizer_seconds_synchronized":update_time,
            "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
            "final_update_from_initial":float(delta_norm(a0,b0,a,b)),
            "snapshots":snapshots}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device",default="cpu")
    p.add_argument("--rank",type=int,default=16)
    p.add_argument("--width",type=int,default=128)
    p.add_argument("--steps",type=int,default=40)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output",type=Path,default=Path("matrix-geometry-synthetic.json"))
    args=p.parse_args()
    if not 1<=args.rank<=args.width or args.steps<1:
        p.error("require 1 <= rank <= width and positive steps")
    device=torch.device(args.device)
    if device.type=="cpu":
        torch.set_num_threads(1)
    validation=validate(device)
    g=torch.Generator(device=device).manual_seed(args.seed)
    def rnd(*shape): return torch.randn(*shape,device=device,generator=g)
    r,w=args.rank,args.width
    a0=rnd(r,w)/math.sqrt(w)
    b0=torch.zeros(w,r,device=device)
    target_a=rnd(r,w)/math.sqrt(w)
    target_b=rnd(w,r)/math.sqrt(r)
    x,xv=rnd(256,w),rnd(128,w)
    y,yv=x@target_a.T@target_b.T,xv@target_a.T@target_b.T
    results=[]
    # A small LR sweep, not a claim that the rates are equivalent.
    for name in ("adamw","muon","polora","tsd-balanced","tsd-unbalanced"):
        for lr in (1e-4,1e-3,1e-2):
            try:
                result=run(name,a0,b0,x,y,xv,yv,lr,args.steps)
                print(name,lr,result["snapshots"][-1]["validation_loss"],flush=True)
            except Exception as e:
                result={"optimizer":name,"lr":lr,"error":repr(e)}
                print(result,flush=True)
            results.append(result)
    report={"schema":1,"scope":"synthetic FP32 factors; rounded-factor diagnostics also tested",
            "validation":validation,"torch":torch.__version__,"device":str(device),
            "seed":args.seed,"rank":r,"width":w,"steps":args.steps,"results":results}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print("Report:",args.output)


if __name__=="__main__":
    main()
