"""Prepare reusable full eligible-projection QH archives from Raw BF16."""
import argparse
import json
import time
import os
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file,load_file
import native_codec as codec
import qh_runtime as qh
from fit_queue import atomic_json
from binary_rank_screen import KEYS


def import_targeted(report_path,report,key,s):
    row=next(r for r in report["projections"] if r["key"]==key)
    candidate=row["candidates"]["mix-2.5-targeted"]
    ar=candidate["archive"]
    path=report_path.parent/ar["file"]
    if path.resolve().parent!=report_path.parent.resolve() or codec.sha256(path)!=ar["sha256"]:
        raise ValueError("targeted archive identity mismatch: "+key)
    with safe_open(str(path),framework="pt",device="cpu") as f:
        meta=f.metadata()
        if (meta["key"]!=key or json.loads(meta["shape"])!=s["shape"]
            or meta["protocol"]!="qh-allocation-v1" or meta["bits"]!="mixed-2-3"
            or meta["group"]!="64" or meta["transform"]!="sylvester-h64"
            or meta["codebook"]!="odd-symmetric" or int(meta["q3_rows"])!=s["q3_rows"]):
            raise ValueError("targeted archive metadata mismatch")
        if set(f.keys())!={"codes2","codes3","scales","row_mask"}:
            raise ValueError("targeted archive fields mismatch")
        t={k:f.get_tensor(k) for k in f.keys()}
    a,b,c,d=qh.lengths(s)
    m,n=s["shape"]
    if (t["scales"].dtype!=torch.bfloat16 or t["scales"].shape!=(m,n//64)
        or any(t[k].dtype!=torch.uint8 or t[k].ndim!=1 or t[k].numel()!=length
               for k,length in (("codes2",a),("codes3",b),("row_mask",d)))
        or sum(v.nbytes for v in t.values())!=ar["payload_bytes"]):
        raise ValueError("targeted archive tensor mismatch")
    p=torch.cat([t["codes2"],t["codes3"],t["scales"].contiguous().view(torch.uint8).flatten(),t["row_mask"]])
    codec.validate(p,s)
    return p


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--allocation-report",type=Path,required=True)
    ap.add_argument("--codec",choices=sorted(qh.NAMES),action="append")
    ap.add_argument("--threads",type=int,default=8)
    args=ap.parse_args()
    if args.threads<1: ap.error("positive threads required")
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True,exist_ok=True)
    names=args.codec or ["qh2","qh3","qh-mix25"]
    if len(set(names))!=len(names): ap.error("duplicate codec")
    result_path=args.out/"prepare-report.json"
    if result_path.exists() or any((args.out/(name+".safetensors")).exists() for name in names):
        ap.error("fresh output directory required")
    start=time.monotonic()
    report={"status":"running","protocol":"qh-packed-preparation-v1","archives":{},"torch":str(torch.__version__)}
    def save():
        report["seconds"]=time.monotonic()-start
        atomic_json(result_path,report)
    try:
        raw_hash=codec.sha256(args.raw)
        allocation=json.loads(args.allocation_report.read_text(encoding="utf-8"))
        if (allocation.get("status")!="completed" or allocation.get("protocol")!="qh-allocation-v1"
            or allocation.get("raw_sha256")!=raw_hash or allocation.get("confirmation_scored") is not False
            or {r["key"] for r in allocation["projections"] if r["status"]=="finished"}!=set(KEYS)):
            raise ValueError("completed matching QH-4B report required")
        report.update(raw_sha256=raw_hash,allocation_sha256=codec.sha256(args.allocation_report),
                      raw_required_for_noneligible=True)
        with safe_open(str(args.raw),framework="pt",device="cpu") as f:
            if not codec.ELIGIBLE<=set(f.keys()): raise ValueError("missing eligible Krea2 projections")
            inventory={k:{"shape":list(f.get_slice(k).get_shape()),"dtype":f.get_slice(k).get_dtype()} for k in sorted(codec.ELIGIBLE)}
            for k,v in inventory.items():
                if v["dtype"]!="BF16" or len(v["shape"])!=2: raise ValueError("Raw BF16 required: "+k)
            report["eligible_inventory"]=inventory
            report["int8_eligible_bytes"]=sum(m*n+4*m for m,n in (v["shape"] for v in inventory.values()))
            for name in names:
                specs={k:qh.spec(name,*v["shape"]) for k,v in inventory.items()}
                layouts=[{k.split(".",2)[2]:v for k,v in specs.items() if k.startswith(f"blocks.{i}.")} for i in range(28)]
                if any(layout!=layouts[0] for layout in layouts): raise ValueError("incompatible classic swap layout")
                payload={}; stats={}
                record=report["archives"][name]={"status":"running","projections":stats}
                save()
                for i,(key,s) in enumerate(specs.items(),1):
                    t0=time.monotonic()
                    if name=="qh-mix25" and key in KEYS:
                        p=import_targeted(args.allocation_report,allocation,key,s)
                        policy="task-targeted-locked"
                    else:
                        p,actual=codec.encode(f.get_tensor(key),name,128,64)
                        if actual!=s: raise ValueError("unexpected codec layout")
                        policy="weight-only" if name=="qh-mix25" else "uniform"
                    codec.validate(p,s)
                    payload[key]=p
                    stats[key]={"policy":policy,"bytes":p.nbytes,"seconds":time.monotonic()-t0}
                    print(name,f"{i}/{len(specs)}",key,policy,flush=True)
                    save()
                path=args.out/(name+".safetensors")
                tmp=path.with_suffix(".tmp")
                try:
                    save_file(payload,str(tmp),metadata={codec.FORMAT:json.dumps(specs),"source_sha256":raw_hash,
                        "qh_protocol":"qh-packed-preparation-v1","allocation_sha256":report["allocation_sha256"]})
                    with safe_open(str(tmp),framework="pt",device="cpu") as check:
                        if set(check.keys())!=set(payload) or codec.manifest(check)!=specs:
                            raise ValueError("archive manifest mismatch")
                        for key,p in payload.items():
                            if not torch.equal(check.get_tensor(key),p): raise ValueError("archive payload mismatch: "+key)
                    os.link(tmp,path)
                finally:
                    tmp.unlink(missing_ok=True)
                record.update(status="finished",file=path.name,sha256=codec.sha256(path),file_bytes=path.stat().st_size,
                              payload_bytes=sum(p.nbytes for p in payload.values()))
                del payload
                save()
        if codec.sha256(args.raw)!=raw_hash: raise ValueError("raw checkpoint changed")
        report["status"]="completed"
    except Exception as e:
        import traceback
        report.update(status="error",error=str(e),traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__=="__main__":
    main()
