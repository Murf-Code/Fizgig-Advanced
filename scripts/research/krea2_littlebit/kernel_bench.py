"""V6 stage 1: real archived Q6 projections, synthetic full-token inputs.

No compression, training, cache publication or model-loader changes.
Run from checkout in the existing Fizgig venv. JSON records every attempt.
"""
import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time
import torch
from safetensors import safe_open
import native_codec as codec
import geometry_codec as geo
import grouped_q6 as grouped

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))


def error(a,b):
    a,b=a.float(),b.float()
    den=b.norm().item()
    return {"relative_l2":(a-b).norm().item()/max(den,1e-30),
            "max_abs":(a-b).abs().max().item(),
            "finite":bool(torch.isfinite(a).all().item())}


def measure(fn, warmup, repeat):
    torch.cuda.synchronize()
    for _ in range(warmup):
        y=fn()
        torch.cuda.synchronize()
        del y
    torch.cuda.reset_peak_memory_stats()
    base=torch.cuda.memory_allocated()
    reserved=torch.cuda.memory_reserved()
    times=[]
    for _ in range(repeat):
        torch.cuda.synchronize()
        start=time.perf_counter()
        y=fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter()-start)*1000)
        del y
    return {"median_ms":statistics.median(times),"samples_ms":times,
            "baseline_allocated_bytes":base,"baseline_reserved_bytes":reserved,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved(),
            "extra_allocated_bytes":torch.cuda.max_memory_allocated()-base}


def attempt(f,b,y,dx,args):
    try:
        fy=f()
        err_f=error(fy,y)
        del fy
        fx=b()
        err_b=error(fx,dx)
        del fx
        return {"status":"finished","forward_error":err_f,"backward_error":err_b,
                "forward":measure(f,args.warmup,args.repeat),
                "backward":measure(b,args.warmup,args.repeat)}
    except torch.cuda.OutOfMemoryError as e:
        return {"status":"cuda_oom","error":str(e)}


def clean():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive",type=Path,required=True)
    ap.add_argument("--raw",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--block",type=int,default=9,choices=range(28))
    ap.add_argument("--tokens",type=int,default=1280)
    ap.add_argument("--warmup",type=int,default=2)
    ap.add_argument("--repeat",type=int,default=5)
    ap.add_argument("--budget-gib",type=float,default=None,
                    help="Optional PyTorch allocator cap, not physical residency enforcement")
    args=ap.parse_args()
    if min(args.tokens,args.warmup,args.repeat)<=0:
        ap.error("positive tokens, warmup and repeat required")
    if args.out.exists():
        ap.error("output already exists; choose a fresh report path")
    if not torch.cuda.is_available():
        ap.error("CUDA required")
    torch.cuda.init()
    free,total=torch.cuda.mem_get_info()
    if args.budget_gib is not None:
        cap=int(args.budget_gib*2**30)
        if not 0<cap<=total:
            ap.error("allocator cap must be positive and no larger than physical VRAM")
        torch.cuda.set_per_process_memory_fraction(cap/total)
    print("Verifying raw checkpoint and reading existing archive; no recompression",flush=True)
    raw_hash=codec.sha256(args.raw)
    archive_hash=codec.sha256(args.archive)
    report={"scope":"single frozen projection; synthetic activations, real weights; no LoRA or swap",
            "raw_sha256":raw_hash,"archive_sha256":archive_hash,"device":torch.cuda.get_device_name(),
            "torch":torch.__version__,"cuda":torch.version.cuda,"initial_free_bytes":free,
            "physical_bytes":total,"allocator_budget_gib":args.budget_gib,
            "tokens":args.tokens,"seed":42,"block":args.block,"projections":[]}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    def save():
        tmp=args.out.with_suffix(args.out.suffix+".tmp")
        tmp.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
        tmp.replace(args.out)
    from fizgig.modules import int8_train
    with safe_open(str(args.archive),framework="pt",device="cpu") as a, safe_open(str(args.raw),framework="pt",device="cpu") as raw:
        if (a.metadata() or {}).get("source_sha256")!=raw_hash:
            raise ValueError("archive/raw checkpoint provenance mismatch")
        specs=codec.manifest(a)
        if set(specs)!=codec.ELIGIBLE or set(a.keys())!=codec.ELIGIBLE:
            raise ValueError("expected complete 224-projection archive")
        for family in codec.MODULES:
            key=f"blocks.{args.block}.{family}.weight"
            s=specs[key]
            if s["codec"]!="q6-had":
                raise ValueError("Q6-only archive required")
            host=a.get_tensor(key)
            codec.validate(host,s)
            w=raw.get_tensor(key)
            if list(w.shape)!=s["shape"] or w.dtype!=torch.bfloat16:
                raise ValueError("unexpected raw shape/dtype")
            m,n=s["shape"]
            print(key,s["shape"],flush=True)
            torch.manual_seed(42)
            x=torch.randn(args.tokens,n,device="cuda",dtype=torch.bfloat16)
            dy=torch.randn(args.tokens,m,device="cuda",dtype=torch.bfloat16)
            p=host.cuda()
            del host
            yf=lambda:geo.forward(x,p,s,codec._unpack,codec.factors)
            xb=lambda:geo.backward(dy,p,s,codec._unpack,codec.factors)
            y,dx=yf(),xb()
            rec={"key":key,"shape":s["shape"],"q6_bytes":p.numel(),"arms":{}}
            report["projections"].append(rec)
            rec["arms"]["q6-reference"]=attempt(yf,xb,y,dx,args)
            save()
            for rows in sorted(set(min(r,((m+127)//128)*128) for r in (256,512,1024,2048,4096,((m+127)//128)*128))):
                for batched in (False,True):
                    name=f"q6-{'batched' if batched else 'grouped'}-{rows}"
                    clean()
                    rec["arms"][name]=attempt(lambda:grouped.forward(x,p,s,rows,batched),
                                               lambda:grouped.backward(dy,p,s,rows,batched),y,dx,args)
                    print(name,rec["arms"][name].get("status"),flush=True)
                    save()
            del p
            clean()
            # Native W8A8 control, actual Fizgig input-gradient implementation.
            layer=codec.stock_layer(w,int8_train).cuda()
            del w
            def stock_f():
                return layer(x)
            def stock_b():
                with torch.enable_grad():
                    z=x.detach().requires_grad_(True)
                    return torch.autograd.grad(layer(z),z,dy)[0]
            rec["arms"]["stock-int8"]=attempt(stock_f,stock_b,y,dx,args)
            # Stock errors are relative to Q6, not an equivalence gate.
            rec["arms"]["stock-int8"]["error_reference"]="q6-reference (different quantization)"
            save()
            del layer,x,dy,y,dx,yf,xb
            clean()
    report["status"]="completed"
    save()
    print(args.out,flush=True)


if __name__=="__main__":
    main()
