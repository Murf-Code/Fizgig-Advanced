"""Short-queue real calibration and uniform-family mixed layout preparation."""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time
import tomllib
from contextlib import ExitStack
import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/"src"))
from fizgig.web import littlebit_benchmark as bench
from native_codec import ELIGIBLE,MODULES,FORMAT,encode,linear,validate,stock_layer,import_file
from geometry_codec import compatible,HQQ,HYBRID
from screen import evaluate,compare
from mixed_selector import choose


def publish(archive,payload,specs,source_hash):
    if not payload or set(payload)!=set(specs) or archive.exists():
        raise ValueError("empty, inconsistent or existing archive")
    tmp=archive.with_suffix(".tmp")
    try:
        save_file(payload,str(tmp),metadata={FORMAT:json.dumps(specs),"source_sha256":source_hash})
        tmp.replace(archive)
    finally:
        tmp.unlink(missing_ok=True)


def mixed(root,out,result):
    choices={suffix:[{"codec":"int8","bytes":sum(result["int8_sizes"][f"blocks.{i}.{suffix}.weight"] for i in range(28)), "error":0.0}] for suffix in MODULES}
    for case in bench.CASES:
        if case in ("stock","nf4","mixed"): continue
        record=bench.read_json(root/case/"prepare.json")
        if not record or record.get("status")!="ready": continue
        if any(record.get(k)!=result[k] for k in ("holdout_sha256","raw_sha256","calibration_sha256")):
            raise ValueError("mixed precision source identity mismatch")
        for suffix,row in record["families"].items():
            if row["eligible"]:
                choices[suffix].append({"codec":case,"bytes":row["packed_bytes"],"error":row["proxy"]})
    plan=choose(choices)
    result["mixed_plan"]=plan
    payload={}; specs={}
    with ExitStack() as stack:
        readers={}
        for suffix,row in plan["families"].items():
            case=row["codec"]
            if case=="int8": continue
            if case not in readers:
                f=stack.enter_context(safe_open(str(root/case/"compact.safetensors"),framework="pt",device="cpu"))
                meta=f.metadata() or {}
                if meta.get("source_sha256")!=result["raw_sha256"]: raise ValueError("mixed archive identity mismatch")
                readers[case]=(f,json.loads(meta[FORMAT]))
            f,manifest=readers[case]
            for block in range(28):
                key=f"blocks.{block}.{suffix}.weight"
                p=f.get_tensor(key); spec=manifest[key]; validate(p,spec)
                payload[key]=p; specs[key]=spec
        if not payload:
            result["status"]="rejected"
            return 3
        if shutil.disk_usage(out).free < sum(p.nbytes for p in payload.values())+2*1024**3:
            raise ValueError("insufficient space for mixed archive")
        publish(out/"compact.safetensors",payload,specs,result["raw_sha256"])
    result.update(status="ready",compact_projections=len(specs),estimated_target_bytes_after_compression=plan["chosen_bytes"],
                  target_bytes_saved=plan["baseline_bytes"]-plan["chosen_bytes"],archive_bytes=(out/"compact.safetensors").stat().st_size)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("raw","out","dataset"): p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--case",choices=tuple(bench.CASES),required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    report=a.out/"prepare.json"
    if report.exists(): p.error("fresh comparison required")
    started=time.monotonic()
    result={"case":a.case,"status":"checking","screen":"real stock training activation/cotangent rows; no error cutoff",
            "storage_scope":"224 eligible main-block weights and scales only; not total VRAM","projections":{},"families":{}}
    try:
        torch.set_num_threads(4)
        hold=Path(tomllib.loads(a.dataset.read_text(encoding="utf-8"))["general"]["holdout_file"])
        result["holdout_sha256"]=bench.sha(hold)
        result["raw_sha256"]=bench.sha(a.raw)
        with safe_open(str(a.raw),framework="pt",device="cpu") as f:
            if not ELIGIBLE<=set(f.keys()) or any(f.get_slice(k).get_dtype()!="BF16" for k in ELIGIBLE):
                raise ValueError("expected original Raw BF16 weights")
            sizes={k:math.prod(f.get_slice(k).get_shape())+4*f.get_slice(k).get_shape()[0] for k in ELIGIBLE}
        result["int8_sizes"]=sizes; result["int8_target_bytes"]=sum(sizes.values())
        if a.case=="stock":
            result["status"]="stock_verified"; return 0
        root=a.out.parent; stock=root/"stock"; prior=bench.read_json(stock/"prepare.json")
        summary=bench.read_json(root/"littlebit-report.json")
        trace=bench.read_json(stock/"trace.json")
        if not prior or prior["status"]!="stock_verified" or not summary or summary["cases"]["stock"]["status"]!="finished" or not trace or not trace.get("calibration"):
            result["status"]="blocked"; raise ValueError("completed stock training and real calibration required")
        if any(prior[k]!=result[k] for k in ("holdout_sha256","raw_sha256")):
            raise ValueError("raw or holdout identity changed")
        calibration=[stock/r["file"] for r in trace["calibration"]]
        result["calibration_sha256"]=[bench.sha(f) for f in calibration]
        result["calibration_steps"]=[r["step"] for r in trace["calibration"]]
        result["calibration_timesteps"]=[trace["steps"][i-1]["timestep"] for i in result["calibration_steps"]]
        if a.case=="nf4":
            result["status"]="native_control"; return 0
        if a.case=="mixed": return mixed(root,a.out,result)
        name,rank=bench.CASES[a.case]
        if name in HQQ|HYBRID:
            from fizgig.minimax.hqq4 import hqq_available
            if not hqq_available():
                result["status"]="blocked"; raise ValueError("HQQ unavailable: run Fizgig updater")
        int8=import_file("littlebit_queue_int8",REPO/"src/fizgig/modules/int8_train.py")
        result.update(codec=name,rank=rank)
        accepted={}; specs={}
        with ExitStack() as stack:
            f=stack.enter_context(safe_open(str(a.raw),framework="pt",device="cpu"))
            probes=[stack.enter_context(safe_open(str(path),framework="pt",device="cpu")) for path in calibration]
            for suffix in MODULES:
                rows=[]; payload={}; manifest={}; result["projections"][suffix]=rows
                for block in range(28):
                    key=f"blocks.{block}.{suffix}.weight"; w=f.get_tensor(key); m,n=w.shape
                    if not compatible(name,m,n,rank):
                        rows.append({"key":key,"reason":"incompatible dimensions"}); continue
                    pck,spec=encode(w,name,rank=rank,group=64); validate(pck,spec)
                    native=stock_layer(w,int8)
                    row={"key":key,"packed_bytes":pck.nbytes,"int8_bytes":sizes[key],"probes":[]}
                    rows.append(row)
                    for index,probe in enumerate(probes):
                        x=probe.get_tensor(key+".x").bfloat16()
                        dy=probe.get_tensor(key+".dy").bfloat16()
                        y,dx=evaluate(lambda t:linear(t,pck,spec),x,dy)
                        ref,rdx=evaluate(native,x,dy)
                        fw,bw=compare(y,ref),compare(dx,rdx)
                        row["probes"].append({"step":result["calibration_steps"][index],
                            "forward_relative_error":fw["relative_error"],"input_gradient_relative_error":bw["relative_error"],
                            "forward_cosine":fw["cosine"],"input_gradient_cosine":bw["cosine"]})
                    if pck.nbytes<sizes[key] and all(v["forward_relative_error"] is not None and v["input_gradient_relative_error"] is not None for v in row["probes"]):
                        payload[key]=pck; manifest[key]=spec
                    print(key,name,row["probes"],flush=True)
                    del w,native
                good=len(payload)==28
                if good:
                    accepted.update(payload); specs.update(manifest)
                    values=[.5*(p["forward_relative_error"]**2+p["input_gradient_relative_error"]**2) for row in rows for p in row["probes"]]
                    result["families"][suffix]={"eligible":True,"packed_bytes":sum(r["packed_bytes"] for r in rows),
                                               "int8_bytes":sum(r["int8_bytes"] for r in rows),"proxy":sum(values)/len(values)}
                else:
                    result["families"][suffix]={"eligible":False,"reason":"requires 28 compatible, smaller, finite projections"}
        if not accepted:
            result["status"]="rejected"; return 3
        total=sum(p.nbytes for p in accepted.values())
        if shutil.disk_usage(a.out).free < total+2*1024**3:
            raise ValueError("insufficient disk space for compact archive")
        if bench.sha(a.raw)!=result["raw_sha256"]: raise ValueError("raw source changed")
        publish(a.out/"compact.safetensors",accepted,specs,result["raw_sha256"])
        saved=sum(sizes[k]-p.nbytes for k,p in accepted.items())
        result.update(status="ready",compact_projections=len(specs),archive_bytes=(a.out/"compact.safetensors").stat().st_size,
                      target_bytes_saved=saved,estimated_target_bytes_after_compression=sum(sizes.values())-saved)
        return 0
    except Exception as exc:
        if result["status"] not in ("blocked","rejected"): result["status"]="error"
        result["error"]=str(exc); raise
    finally:
        result["seconds"]=time.monotonic()-started
        bench.atomic_json(report,result)


if __name__=="__main__":
    sys.exit(main())
