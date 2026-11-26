"""Process-isolated native Fizgig compact training and stock transfer scoring."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import torch
from safetensors import safe_open
import native_codec as codec


def compact_loader(archive,original):
    def load(driver,path,device,precision="bf16",blocks_to_swap=0,supports_backward=True,send_only=False):
        if driver.__class__.__name__!="Krea2Driver" or precision!="int8" or send_only:
            raise ValueError("compact test requires Krea2, explicit INT8 fallback and classic swap")
        from fizgig.families import quant
        swap=int(blocks_to_swap)
        if not 0<=swap<=driver.max_blocks_to_swap():
            raise ValueError("invalid swap count")
        with safe_open(str(archive),framework="pt",device="cpu") as packed:
            specs=codec.manifest(packed)
            if not specs or set(specs)!=set(packed.keys()) or codec.sha256(path)!=(packed.metadata() or {}).get("source_sha256"):
                raise ValueError("compact archive/source identity mismatch")
            layouts=[{k.split(".",2)[2]:v for k,v in specs.items() if k.startswith(f"blocks.{i}.")} for i in range(28)]
            if any(l!=layouts[0] for l in layouts):
                raise ValueError("classic swap requires identical projection layouts")
            got=driver.stream_dit(path)
            if got is None:
                raise ValueError("raw streaming loader unavailable")
            model,reader,dtype=got
            target=dict(quant._targets(model,driver))
            if not set(specs)<={n+".weight" for n in target}:
                raise ValueError("compact target mismatch")
            with reader as raw:
                keys=set(raw.keys())
                if keys!=set(model.state_dict()):
                    raise ValueError("model layout mismatch")
                rest={k:raw.get_tensor(k,device=torch.device("cpu"),dtype=dtype)
                      for k in keys if k.rpartition(".")[0] not in target}
                model.load_state_dict(rest,strict=False,assign=True)
                del rest
                for name,module in target.items():
                    key=name+".weight"
                    if key in specs:
                        if name+".bias" in keys:
                            module.bias=torch.nn.Parameter(raw.get_tensor(name+".bias",device=torch.device("cpu"),dtype=dtype),requires_grad=False)
                        codec.install(module,packed.get_tensor(key).clone(),specs[key])
                        if not swap:
                            module.to(device)
                    else:
                        values={k.rpartition(".")[2]:raw.get_tensor(k,device=torch.device("cpu"),dtype=dtype)
                                for k in (key,name+".bias") if k in keys}
                        module.load_state_dict(values,strict=True,assign=True)
                        quant._quantize_one(module,driver,"int8",torch.device(device),"cpu" if swap else None)
                if any(v.is_meta for v in model.state_dict().values()):
                    raise ValueError("incomplete compact model")
        model.eval().requires_grad_(False)
        model._int8_quantized=True
        if swap:
            driver.enable_block_swap(model,swap,device,supports_backward,send_only=False)
        else:
            quant.move(model,device)
        print(f"Independent compact base: {len(specs)} packed projections loaded",flush=True)
        return model,swap
    return load


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("train","evaluate"))
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--archive",type=Path)
    p.add_argument("--case",default="legacy")
    p.add_argument("--stock-trace",type=Path)
    p.add_argument("--attempt-result",type=Path)
    p.add_argument("--dit",type=Path)
    p.add_argument("--snapshot",type=Path)
    p.add_argument("--stock-lora",type=Path)
    p.add_argument("--compressed-lora",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--swaps",type=int,default=22)
    p.add_argument("--deployment",choices=("int8","bf16","compact"),default="int8")
    p.add_argument("--split",choices=("development","confirmation","all"),default="development")
    p.add_argument("--paired-horizons",action="store_true")
    argv=sys.argv[1:]
    cut=argv.index("--") if "--" in argv else len(argv)
    a=p.parse_args(argv[:cut])
    tail=argv[cut:]
    sys.path.insert(0,str(a.repo.resolve()/"src"))
    from fizgig.families import quant
    if a.mode=="train":
        # parse_known_args keeps everything after the explicit delimiter.
        if tail[:1]!=["--"] or "--finetune" in tail or "--resume" in tail:
            p.error("expected archive and fresh LoRA trainer arguments after --")
        from fizgig.families import train
        original=quant.load_base
        if a.archive:
            quant.load_base=compact_loader(a.archive,original)
        observer=None
        if a.case=="timing":
            from timing_trace import TimingTrace
            observer=TimingTrace()
            train._LITTLEBIT_TRACE=observer
        elif a.case!="legacy":
            from research_trace import Trace
            if a.case!="stock" and not a.stock_trace:
                raise ValueError("paired stock trace required")
            observer=Trace(a.case,a.stock_trace)
            train._LITTLEBIT_TRACE=observer
        # Record allocator maxima across the trainer's per-epoch resets.
        from fit_queue import atomic_json, OOM_EXIT
        started = time.monotonic()
        metrics = {"status":"running", "peak_allocated_bytes":0, "peak_reserved_bytes":0}
        monitor=None
        reset = torch.cuda.reset_peak_memory_stats
        def collect(device=None):
            if torch.cuda.is_initialized():
                metrics["peak_allocated_bytes"] = max(metrics["peak_allocated_bytes"], torch.cuda.max_memory_allocated(device))
                metrics["peak_reserved_bytes"] = max(metrics["peak_reserved_bytes"], torch.cuda.max_memory_reserved(device))
        def tracking_reset(device=None):
            collect(device)
            return reset(device)
        torch.cuda.reset_peak_memory_stats = tracking_reset
        rc = 0
        try:
            if observer is not None:
                # Avoid treating WDDM paging beyond physical VRAM as GPU fit.
                # This caps the PyTorch allocator, not all driver allocations.
                free,total=torch.cuda.mem_get_info()
                metrics.update(initial_free_bytes=free,physical_vram_bytes=total,
                               device_name=torch.cuda.get_device_name(),torch_version=torch.__version__)
                if not os.environ.get("FIZGIG_SIM_VRAM_GB"):
                    budget=max(1,int(free-512*1024**2))
                    torch.cuda.set_per_process_memory_fraction(min(1.0,budget/total))
                    metrics["allocator_budget_bytes"]=budget
                else:
                    metrics["allocator_budget_source"]="existing FIZGIG_SIM_VRAM_GB"
                from wddm_memory import WddmMemory
                monitor=WddmMemory()
            sys.argv=[sys.argv[0],*tail[1:]]
            train.main()
            if observer is not None:
                observer.finish()
            metrics["status"] = "finished"
        except torch.cuda.OutOfMemoryError as exc:
            metrics.update(status="cuda_oom", error=str(exc))
            traceback.print_exc()
            rc = OOM_EXIT
        except BaseException as exc:
            metrics.update(status="error", error=str(exc))
            raise
        finally:
            try:
                collect()
            finally:
                torch.cuda.reset_peak_memory_stats = reset
                quant.load_base = original
                train._LITTLEBIT_TRACE=None
                if monitor is not None:
                    try:
                        metrics["wddm_memory"]=monitor.stop()
                    except Exception as exc:
                        metrics["wddm_memory"]={"status":"unavailable","reason":str(exc)}
                metrics["seconds"] = time.monotonic()-started
                if a.attempt_result:
                    atomic_json(a.attempt_result,metrics)
        return rc
    if tail or not all((a.dit,a.snapshot,a.stock_lora,a.compressed_lora,a.out)):
        p.error("evaluation requires stock base, snapshot, both adapters and output")
    from fizgig.families.registry import get
    from fizgig.families.lora import FamilyLoRA
    from fizgig.training.val_loss import FixedValidation
    snap=torch.load(a.snapshot,map_location="cpu",weights_only=True)
    if snap.get("version")!=1 or not snap.get("wanted") or not snap.get("ts"):
        raise ValueError("invalid validation snapshot")
    wanted=snap["wanted"]
    if set(snap["batches"])!=set(wanted):
        raise ValueError("incomplete snapshot")
    val=FixedValidation(wanted,images=len(wanted),timesteps=len(snap["ts"]),seed=int(snap["seed"]))
    val.wanted=list(wanted)
    val.groups=snap.get("groups") or [wanted]
    val.ts=list(snap["ts"])
    # Preserve original image indices/seeds by scoring the same ordered list.
    # Development excludes reserved confirmation keys; do not run the reserved subset here.
    import hashlib
    order=sorted(wanted,key=lambda k:hashlib.sha256(("littlebit-split-v1:"+str(k)).encode()).digest())
    n=max(1,len(order)//2) if len(order)>=4 else len(order)
    dev=set(order[:n]); confirm=set(order[n:])
    subset=dev if a.split=="development" else confirm if a.split=="confirmation" else set(wanted)
    if not subset:
        raise ValueError("no untouched confirmation images available; use a larger dataset")
    val.batches={k:v for k,v in snap["batches"].items() if k in subset}
    driver=get("krea2").load_driver()
    driver.set_options({})
    device=torch.device("cuda:0")
    if a.deployment=="compact":
        if not a.archive: raise ValueError("compact evaluation needs training archive")
        model,swapped=compact_loader(a.archive,quant.load_base)(driver,str(a.dit),device,"int8",a.swaps)
    else:
        model,swapped=quant.load_base(driver,str(a.dit),device,a.deployment,a.swaps)
    net=FamilyLoRA(model,driver,device=device)
    # net.targets was built before wrappers replaced the original Linear modules.
    expected=set(driver.lora_target_names(None))
    if set(net.targets)!=expected or set(net.wrapped)!=expected or not expected:
        raise ValueError("deployment model missing expected standard LoRA targets")
    def score():
        row=val.run(driver,model,device,net,swapped=bool(swapped))
        if row is None or not math.isfinite(row[0]):
            raise ValueError("invalid validation result")
        return {"loss":row[0],"per_timestep":row[1],"per_image":val.last_per_image,"seconds":row[2]}
    from paired_metrics import paired
    from task_diagnostics import movement, gradients
    baseline=score()
    def pair(stock_path,candidate_path,geometry=False,calibration=None):
        scores={"no_adapter":baseline}; factors=[]
        for label,path in (("stock_trained",stock_path),("compressed_trained",candidate_path)):
            read=net.read_file(str(path))
            if set(read)!=expected or any(v[0]!="lora" for v in read.values()):
                raise ValueError(f"{label}: missing={sorted(expected-set(read))}; extra={sorted(set(read)-expected)}")
            if geometry: factors.append(read)
            if not net.add_file(str(path),label,strength=1.0):
                raise ValueError("adapter did not load")
            try:
                covered={k for k,v in net.wrapped.items() if label in v.adapters}
                if covered!=expected:
                    raise ValueError(f"{label}: missing reloaded modules {sorted(expected-covered)}")
                scores[label]=score()
            finally:
                net.remove(label)
        result={"scores":scores,**paired(scores)}
        if geometry:
            result["effective_updates"]=movement(*factors,calibration)
        return result

    horizons={}
    if a.paired_horizons:
        if a.split!="development":
            raise ValueError("horizon selection is development-only")
        sroot,croot=a.stock_lora.parent,a.compressed_lora.parent
        st=json.loads((sroot/"trace.json").read_text(encoding="utf-8"))
        ct=json.loads((croot/"trace.json").read_text(encoding="utf-8"))
        if st["initial_sha256"]!=ct["initial_sha256"] or st["expected_modules"]!=ct["expected_modules"] or set(st["expected_modules"])!=expected:
            raise ValueError("training manifest/initialization mismatch")
        if [r["input_sha256"] for r in st["steps"]]!=[r["input_sha256"] for r in ct["steps"]] or st["completed_updates"]!=ct["completed_updates"]:
            raise ValueError("different optimizer trajectories' input schedule")
        if set(st["checkpoints"])!=set(ct["checkpoints"]):
            raise ValueError("missing matched horizons")
        from contextlib import ExitStack
        with ExitStack() as stack:
            cf=[stack.enter_context(safe_open(str(sroot/r["file"]),framework="pt",device="cpu")) for r in st["calibration"]]
            for step in sorted(st["checkpoints"],key=int):
                row=pair(sroot/st["checkpoints"][step],croot/ct["checkpoints"][step],
                         geometry=a.deployment=="int8",calibration=cf)
                if a.deployment=="int8":
                    filename=f"task-grad-{int(step):06d}.safetensors"
                    row["task_factor_gradients"]=gradients(sroot/filename,croot/filename,expected)
                    row["gradient_scope"]="pre-clipping factor gradients; step 1 same initialization, later steps different adapter states"
                horizons[step]=row
            # Final export is scored independently and geometry uses all stock captures.
            result=pair(a.stock_lora,a.compressed_lora,geometry=a.deployment=="int8",calibration=cf)
        final_step=str(st["completed_updates"])
        result["completed_updates"]=int(final_step)
    else:
        result=pair(a.stock_lora,a.compressed_lora)
    result.update(horizons=horizons,deployment=a.deployment,split=a.split,
                  deployment_swaps=swapped,expected_modules=sorted(expected),
                  development_images=sorted(dev),reserved_confirmation_images=sorted(confirm),
                  snapshot_sha256=codec.sha256(a.snapshot),source_sha256=codec.sha256(a.dit))
    with a.out.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,allow_nan=False)


if __name__=="__main__":
    sys.exit(main())
