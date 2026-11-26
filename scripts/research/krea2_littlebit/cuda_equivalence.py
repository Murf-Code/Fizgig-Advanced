"""CUDA numerical preflight: actual 28-block Krea2, reduced random widths.
Not a compression-quality or production-size memory experiment.
"""
import argparse
import gc
import math
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from fit_queue import atomic_json
from native_codec import encode,install,ELIGIBLE


def norm_error(a,b):
    if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("invalid comparison tensors")
    a,b=a.double(),b.double()
    diff=float((a-b).norm()); ref=float(b.norm())
    return {"relative_l2":diff/max(ref,1e-12),"absolute_l2":diff,"reference_l2":ref,
            "bitwise":torch.equal(a,b)}


def run(swaps):
    from fizgig.families.registry import get
    from fizgig.families.lora import FamilyLoRA,TRAINABLE
    from fizgig.krea2.model import SingleMMDiTConfig,SingleStreamDiT
    from fizgig.krea2.sampling import prepare
    from fizgig.training.optimizers import create_optimizer
    dev=torch.device("cuda:0")
    torch.manual_seed(812)
    config=SingleMMDiTConfig(features=256,tdim=64,txtdim=128,heads=4,kvheads=2,
                    multiplier=2,layers=28,patch=2,channels=16,txtlayers=2,txtheads=2,txtkvheads=2)
    model=SingleStreamDiT(config,attn_mode="torch",split_attn=False).bfloat16().requires_grad_(False)
    driver=get("krea2").load_driver()
    driver.set_options({})
    layers={k+".weight":m for k,m in model.named_modules() if k+".weight" in ELIGIBLE}
    if set(layers)!=ELIGIBLE: raise RuntimeError("incomplete reduced quant inventory")
    # Independent, distinct weights per block. All use the actual packed codec.
    payload_hashes=[]
    from research_trace import fingerprint
    for k,m in layers.items():
        p,s=encode(m.weight.detach(),"q6-had",rank=0,group=64)
        install(m,p,s)
        if k.endswith(".attn.wk.weight"): payload_hashes.append(fingerprint(p))
    if len(set(payload_hashes))!=28: raise RuntimeError("test requires distinct block weights")
    del layers,m,p
    model._int8_quantized=True
    if swaps:
        driver.enable_block_swap(model,swaps,dev,True,send_only=False)
    else:
        model.to(dev)
    model.enable_gradient_checkpointing()
    net=FamilyLoRA(model,driver,device=dev)
    torch.manual_seed(394)
    net.add_trainable(4,4)
    if set(net.wrapped)!=set(driver.lora_target_names(None)): raise RuntimeError("missing targets")
    optimizer,_=create_optimizer("muon",net.parameters(),.00025)
    model.train()
    records=[]; handles=[]; observed={}
    # Observe one projection in every main block; record output and exact input cotangent.
    def fwd(k):
        def hook(m,args,y):
            if torch.is_grad_enabled() and k+".y" not in observed:
                observed[k+".y"]=y.detach().cpu().clone()
        return hook
    def bwd(k):
        def hook(m,gi,go):
            if gi[0] is None: raise RuntimeError("missing layer input gradient")
            observed[k+".dx"]=gi[0].detach().cpu().clone()
        return hook
    for i in range(28):
        k=f"blocks.{i}.attn.wk"
        handles.extend([net.wrapped[k].register_forward_hook(fwd(k)),
                        net.wrapped[k].register_full_backward_hook(bwd(k))])
    gen=torch.Generator().manual_seed(381)
    try:
        for step in range(2):
            observed.clear()
            optimizer.zero_grad(set_to_none=True)
            lat=torch.randn(1,16,4,4,generator=gen).to(dev,torch.bfloat16)
            context=torch.randn(1,7,2,128,generator=gen).to(dev,torch.bfloat16)
            mask=torch.tensor([[1,1,1,1,1,0,0]],device=dev,dtype=torch.bool)
            img,pos,full_mask=prepare(lat,7,2,mask)
            img=img.detach().requires_grad_(True)
            t=torch.tensor([.25+.4*step],device=dev,dtype=torch.bfloat16)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                y=model(img,context,t,pos,full_mask)
            target=torch.randn(y.shape,generator=gen).to(dev)
            loss=F.mse_loss(y.float(),target)
            loss.backward()
            if len(observed)!=56: raise RuntimeError("incomplete observed projection pairs")
            result={"output":y.detach().cpu(),"image_input_grad":img.grad.detach().cpu(),**observed}
            for k,w in net.wrapped.items():
                for j,p in enumerate(w.adapters[TRAINABLE].parameters()):
                    if p.grad is None: raise RuntimeError(f"missing factor gradient {k}/{j}")
                    result[f"{k}.grad.{j}"]=p.grad.detach().cpu().clone()
            optimizer.step()
            torch.cuda.synchronize()
            # Effective rank-4 matrices are small in this reduced model. Compare BA directly.
            for k,w in net.wrapped.items():
                a,b=w.adapters[TRAINABLE]
                result[k+".BA"]=(b.weight.detach().float()@a.weight.detach().float()).cpu()
            records.append(result)
    finally:
        for h in handles: h.remove()
        if swaps:
            model.offloader.set_forward_only(True)
            model.offloader.remove_hooks()
            pool=getattr(model.offloader,"thread_pool",None)
            if pool: pool.shutdown(wait=True)
        torch.cuda.synchronize()
    return records


def compare_runs(a,b):
    if len(a)!=len(b): raise ValueError("step mismatch")
    out=[]
    for x,y in zip(a,b):
        if set(x)!=set(y): raise ValueError("tensor coverage mismatch")
        out.append({k:norm_error(x[k],y[k]) for k in sorted(x)})
    return out


def decide(repeat,swapped):
    # Numerical correctness guard, NOT a learning-retention threshold.
    failures=[]
    for i,(r,s) in enumerate(zip(repeat,swapped),1):
        for k,v in s.items():
            tolerance=max(1e-5,3*r[k]["relative_l2"])
            if v["relative_l2"]>tolerance and v["absolute_l2"]>1e-7:
                failures.append({"step":i,"tensor":k,"error":v,"resident_repeat":r[k],"tolerance":tolerance})
    return failures


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.repo/"src"))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    report={"status":"running","scope":"reduced random Krea2; rank4; two Muon updates; Q6 Hadamard",
            "resident_swaps":0,"swapped_swaps":22}
    try:
        if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
        report["device"]=torch.cuda.get_device_name(0)
        report["torch"]=torch.__version__
        base=run(0); gc.collect(); torch.cuda.empty_cache()
        again=run(0); gc.collect(); torch.cuda.empty_cache()
        exchanged=run(22); gc.collect(); torch.cuda.empty_cache()
        repeat=compare_runs(again,base); swap=compare_runs(exchanged,base)
        failures=decide(repeat,swap)
        report.update(resident_repeat=repeat,swapped=swap,failures=failures,
                      status="failed" if failures else "passed")
        return 1 if failures else 0
    except Exception as exc:
        report.update(status="error",error=str(exc))
        raise
    finally:
        atomic_json(a.out,report)


if __name__=="__main__":
    sys.exit(main())
