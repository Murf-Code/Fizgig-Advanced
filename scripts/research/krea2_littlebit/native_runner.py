"""Process-isolated native Fizgig compact training and stock transfer scoring."""
import argparse
import json
import math
from pathlib import Path
import sys
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
    p.add_argument("--dit",type=Path)
    p.add_argument("--snapshot",type=Path)
    p.add_argument("--stock-lora",type=Path)
    p.add_argument("--compressed-lora",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--swaps",type=int,default=22)
    argv=sys.argv[1:]
    cut=argv.index("--") if "--" in argv else len(argv)
    a=p.parse_args(argv[:cut])
    tail=argv[cut:]
    sys.path.insert(0,str(a.repo.resolve()/"src"))
    from fizgig.families import quant
    if a.mode=="train":
        # parse_known_args keeps everything after the explicit delimiter.
        if not a.archive or tail[:1]!=["--"] or "--finetune" in tail or "--resume" in tail:
            p.error("expected archive and fresh LoRA trainer arguments after --")
        from fizgig.families import train
        original=quant.load_base
        quant.load_base=compact_loader(a.archive,original)
        try:
            sys.argv=[sys.argv[0],*tail[1:]]
            train.main()
        finally:
            quant.load_base=original
        return
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
    val.batches=snap["batches"]
    driver=get("krea2").load_driver()
    driver.set_options({})
    device=torch.device("cuda:0")
    model,swapped=quant.load_base(driver,str(a.dit),device,"int8",a.swaps)
    net=FamilyLoRA(model,driver,device=device)
    def score():
        row=val.run(driver,model,device,net,swapped=bool(swapped))
        if row is None or not math.isfinite(row[0]):
            raise ValueError("invalid validation result")
        return {"loss":row[0],"per_timestep":row[1],"seconds":row[2]}
    scores={"no_adapter":score()}
    coverage=[]
    for label,path in (("stock_trained",a.stock_lora),("compressed_trained",a.compressed_lora)):
        if not net.add_file(str(path),label,strength=1.0):
            raise ValueError("adapter did not load")
        try:
            coverage.append({k for k,v in net.wrapped.items() if label in v.adapters})
            scores[label]=score()
        finally:
            net.remove(label)
    if coverage[0]!=coverage[1]:
        raise ValueError("different LoRA module coverage")
    base=scores["no_adapter"]["loss"]
    gain=base-scores["stock_trained"]["loss"]
    candidate=base-scores["compressed_trained"]["loss"]
    result={"scores":scores,"stock_gain":gain,"candidate_gain":candidate,
            "retention":candidate/gain if gain>1e-8 else None,
            "snapshot_sha256":codec.sha256(a.snapshot),"source_sha256":codec.sha256(a.dit)}
    with a.out.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,allow_nan=False)


if __name__=="__main__":
    main()
