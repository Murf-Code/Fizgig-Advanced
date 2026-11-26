"""Conservative raw-weight gate for the uniform-layout LittleBit queue."""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import tomllib

import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/"src"))
from fizgig.web import littlebit_benchmark as bench
from screen import evaluate, compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("raw","out","dataset"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--case",choices=tuple(bench.CASES),required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    report=a.out/"prepare.json"
    if report.exists():
        p.error("fresh comparison required; prepare.json already exists")
    result={"case":a.case,"status":"checking","gate":"synthetic rows, native INT8 reference",
            "threshold":0.05,"seeds":[42,314159],"rows":64,"projections":{}}
    try:
        from native_codec import ELIGIBLE, MODULES, FORMAT, convert, encode, linear, validate, stock_layer, import_file
        from geometry_codec import compatible, HQQ, HYBRID
        int8=import_file("littlebit_queue_int8",REPO/"src/fizgig/modules/int8_train.py")
        torch.set_num_threads(4)
        hold=Path(tomllib.loads(a.dataset.read_text(encoding="utf-8"))["general"]["holdout_file"])
        result["holdout_sha256"]=bench.sha(hold)
        result["raw_sha256"]=bench.sha(a.raw)
        with safe_open(str(a.raw),framework="pt",device="cpu") as f:
            keys=set(f.keys())
            if not ELIGIBLE <= keys or FORMAT in (f.metadata() or {}) or any(
                k.endswith((".weight_scale",".scale_weight",".comfy_quant")) for k in keys):
                raise ValueError("Expected complete original Krea2 Raw checkpoint")
            if any(f.get_slice(k).get_dtype()!="BF16" for k in ELIGIBLE):
                raise ValueError("Expected raw BF16 main-block weights")
            stock_sizes = {k: math.prod(f.get_slice(k).get_shape()) + 4*f.get_slice(k).get_shape()[0]
                           for k in ELIGIBLE}
            result["int8_target_bytes"] = sum(stock_sizes.values())
            result["int8_target_bytes_per_block"] = [sum(v for k,v in stock_sizes.items() if k.startswith(f"blocks.{i}.")) for i in range(28)]
        if a.case=="stock":
            result["status"]="stock_verified"
            return 0
        root=a.out.parent
        stock=root/"stock"
        stock_name=root.name+"-stock"
        prior=bench.read_json(stock/"prepare.json")
        summary=bench.read_json(root/"littlebit-report.json")
        if (not prior or prior.get("status")!="stock_verified" or not summary or
            summary["cases"]["stock"]["status"]!="finished" or
            not (stock/f"{stock_name}.safetensors").is_file() or
            not (stock/f".fizgig_val_{stock_name}.pt").is_file()):
            result["status"]="blocked"
            raise ValueError("Stock control must finish with LoRA and validation snapshot first")
        if any(prior.get(k)!=result[k] for k in ("holdout_sha256","raw_sha256")):
            raise ValueError("Raw checkpoint or validation holdout changed between cases")
        codec,rank=bench.CASES[a.case]
        if codec in HQQ | HYBRID:
            from fizgig.minimax.hqq4 import hqq_available
            if not hqq_available():
                result["status"]="blocked"
                raise ValueError("HQQ dependency unavailable: run the Fizgig updater to install hqq==0.2.8.post1")
        result.update(codec=codec,rank=rank)
        accepted=[]
        totals={}
        with safe_open(str(a.raw),framework="pt",device="cpu") as f:
            for suffix in MODULES:
                rows=[]
                result["projections"][suffix]=rows
                good=True
                for block in range(28):
                    key=f"blocks.{block}.{suffix}.weight"
                    w=f.get_tensor(key)
                    m,n=w.shape
                    if not compatible(codec,m,n,rank):
                        rows.append({"key":key,"reason":"incompatible dimensions"})
                        good=False
                        break
                    pck,spec=encode(w,codec,rank=rank,group=64)
                    validate(pck,spec)
                    native=stock_layer(w,int8)
                    stock_bytes=native.weight.nbytes+native._int8_wscale.nbytes
                    packed_bytes=pck.nbytes
                    row={"key":key,"packed_bytes":packed_bytes,"int8_bytes":stock_bytes,
                         "bytes_per_weight":packed_bytes/(m*n),"probes":[]}
                    rows.append(row)
                    good=packed_bytes<stock_bytes
                    for seed in result["seeds"]:
                        g=torch.Generator().manual_seed(seed)
                        x=torch.randn(64,n,generator=g).bfloat16()
                        dy=torch.randn(64,m,generator=g).bfloat16()
                        y,dx=evaluate(lambda t:linear(t,pck,spec),x,dy)
                        ref,rdx=evaluate(native,x,dy)
                        forward,backward=compare(y,ref),compare(dx,rdx)
                        fw,bw=forward["relative_error"],backward["relative_error"]
                        row["probes"].append({"seed":seed,"forward":fw,"backward":bw,
                                              "forward_cosine":forward["cosine"],"backward_cosine":backward["cosine"]})
                        good=good and fw is not None and bw is not None and fw<=0.05 and bw<=0.05
                    print(key,codec,"pass" if good else "reject",row["probes"],flush=True)
                    del w,pck,native
                    if not good:
                        break
                if good and len(rows)==28:
                    accepted.append(f"blocks.*.{suffix}.weight")
                    totals[suffix]={"packed_bytes":sum(r["packed_bytes"] for r in rows),
                                    "int8_bytes":sum(r["int8_bytes"] for r in rows)}
        result["accepted_patterns"]=accepted
        result["accepted_storage"]=totals
        saved = sum(v["int8_bytes"]-v["packed_bytes"] for v in totals.values())
        result["estimated_target_bytes_after_compression"] = result["int8_target_bytes"]-saved
        result["target_bytes_saved"] = saved
        result["storage_scope"] = "main-block eligible linear weights and scales only; excludes other weights and training overhead"
        if not accepted:
            result["status"]="rejected"
            print("No uniform projection family passed; training deliberately skipped.",flush=True)
            return 3
        if shutil.disk_usage(a.out).free < sum(v["packed_bytes"] for v in totals.values())+2*1024**3:
            raise ValueError("Insufficient disk space for a separate compact archive")
        archive=a.out/"compact.safetensors"
        specs=convert(a.raw,archive,accepted,codec,rank,64)
        if len(specs)!=28*len(accepted):
            raise ValueError("Unexpected compact archive coverage")
        result.update(status="ready",archive_bytes=archive.stat().st_size,
                      compact_projections=len(specs))
        return 0
    except Exception as exc:
        if result["status"] not in ("blocked","rejected"):
            result["status"]="error"
        result["error"]=str(exc)
        raise
    finally:
        bench.atomic_json(report,result)


if __name__=="__main__":
    sys.exit(main())
