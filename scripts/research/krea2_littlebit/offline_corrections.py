"""Replay V5 real stock captures offline; no new LoRA training or runtime patches."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
import torch
from safetensors import safe_open
import native_codec as codec
import private_correction as pc
from fit_queue import atomic_json

PAIRS=(( "blocks.0.attn.wk.weight","blocks.1.attn.wk.weight"),
       ("blocks.18.attn.wk.weight","blocks.20.attn.wk.weight"),
       ("blocks.0.mlp.up.weight","blocks.1.mlp.up.weight"))


def read_capture(stock,key,record,trace):
    step=next(s for s in trace["steps"] if s["step"]==record["step"])
    path=stock/record["file"]
    if path.resolve().parent!=stock.resolve(): raise ValueError("invalid capture path")
    with safe_open(str(path),framework="pt",device="cpu") as f:
        x,g=f.get_tensor(key+".x"),f.get_tensor(key+".dy")
    tokens=record["tokens"]
    ni,nt=tokens["sampled_image"],tokens["sampled_text"]
    if len(x)!=len(g) or len(x)!=ni+nt or not torch.isfinite(x).all() or not torch.isfinite(g).all():
        raise ValueError("bad capture")
    return {"x":x,"dy":g,"sampled_image":ni,"step":record["step"],
            "timestep":step["timestep"],"item":step["items"][0]}


def run_pair(weights,keys,stock,trace,split,int8_module,steps):
    m,n=weights[0].shape
    if any(w.shape!=(m,n) or w.dtype!=torch.bfloat16 for w in weights):
        raise ValueError("expected equal-shape raw BF16 pair")
    budget=pc.ranks(m,n)
    refs=[codec.stock_layer(w,int8_module) for w in weights]
    captures={name:[[read_capture(stock,k,r,trace) for r in rs] for k in keys]
              for name,rs in split.items() if name!="confirmation"}
    report={"keys":list(keys),**budget,"candidates":{}}
    # All variants live as small BF16 factors plus actual packed residuals.
    # Retain only the candidates' payloads until their development selection.
    models={}
    def add(name,res,aa=None,bb=None,fit=None):
        nbytes=sum(p.nbytes for p,s in res)+pc.DESCRIPTOR_BYTES
        if aa is not None: nbytes+=sum(x.nbytes for x in aa+bb)
        if nbytes>budget["budget_bytes"]: raise ValueError("candidate exceeded byte cap")
        row={"charged_bytes":nbytes,"unused_bytes":budget["budget_bytes"]-nbytes,
             "rank":aa[0].shape[0] if aa else 0,"weight_error":[],"fit":fit,
             "active_layer_payload_bytes":[],"dense_equivalent_macs_per_token":[]}
        for i,w in enumerate(weights):
            p,s=res[i]
            a=aa[0 if len(aa)==1 else i] if aa else None
            b=bb[i] if bb else None
            row["weight_error"].append(pc.weight_error(w,p,s,a,b))
            row["active_layer_payload_bytes"].append(p.nbytes+(a.nbytes+b.nbytes if a is not None else 0))
            row["dense_equivalent_macs_per_token"].append(m*n+(a.shape[0]*(m+n) if a is not None else 0))
        models[name]=(res,aa,bb)
        report["candidates"][name]=row
    def evaluate(name,subset):
        res,aa,bb=models[name]
        rows=[]
        for i,w in enumerate(weights):
            p,s=res[i]
            a=aa[0 if len(aa)==1 else i] if aa else None
            b=bb[i] if bb else None
            for c in captures[subset][i]:
                rows.append({"key":keys[i],"step":c["step"],"item":c["item"],"timestep":c["timestep"],
                             "errors":pc.score(w,p,s,c,refs[i],a,b)})
        report["candidates"][name][subset]={"rows":rows,"raw_proxy":pc.aggregate(rows)}
    for name in ("q5-had","q6-had"):
        res=[codec.encode(w,name,rank=0) for w in weights]
        add(name,res)
        evaluate(name,"development")
    def value(name):
        v=report["candidates"][name]["development"]["raw_proxy"]
        return float("inf") if v is None else v
    # At most the best private and shared candidate are retained in host RAM.
    # All development measurements remain in the report.
    for shared in (False,True):
        label="shared" if shared else "private"
        limit=budget[label+"_rank"]
        if not limit: continue
        best=None
        for rank in sorted({max(1,limit//2),limit}):
            aa,bb=pc.svd_factors(weights,rank,shared)
            res=[pc.residual(w,aa[0 if shared else i],bb[i]) for i,w in enumerate(weights)]
            name=label+f"-svd-r{rank}-q5"
            add(name,res,aa,bb)
            evaluate(name,"development")
            targets=[[pc.fit_target(w,*res[i],c) for c in captures["fit"][i]] for i,w in enumerate(weights)]
            fa,fb,fit=pc.fit_factors(aa,bb,targets,steps)
            fitted=label+f"-task-r{rank}-q5"
            add(fitted,res,fa,fb,fit)
            evaluate(fitted,"development")
            eligible=[k for k in (best,name,fitted) if k is not None]
            best=min(eligible,key=lambda k:(value(k),report["candidates"][k]["charged_bytes"],k))
            for k in eligible:
                if k!=best: del models[k]
            del targets,aa,bb,fa,fb,res
    private=[k for k in models if k.startswith("private-")]
    winner=min(models,key=lambda k:(value(k),report["candidates"][k]["charged_bytes"],k))
    if value(winner)==float("inf"): raise ValueError("no valid development signal")
    best_private=min(private,key=lambda k:(value(k),k)) if private else None
    # Freeze selection before evaluating the reserved capture group.
    finalists=list(dict.fromkeys(k for k in ("q6-had",best_private,winner) if k is not None))
    report.update(development_winner=winner,best_private=best_private,confirmation_candidates=finalists)
    captures["confirmation"]=[[read_capture(stock,k,r,trace) for r in split["confirmation"]] for k in keys]
    for name in finalists: evaluate(name,"confirmation")
    report["scope"]="local operator diagnostics, not portable LoRA quality; no promotion gate"
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw",type=Path,required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--stock",type=Path,help="V5 comparison's stock directory")
    group.add_argument("--comparisons",type=Path,help="find latest completed V5 stock under this directory")
    p.add_argument("--out",type=Path,required=True,help="fresh JSON report path")
    p.add_argument("--fit-steps",type=int,default=40)
    p.add_argument("--threads",type=int,default=4)
    a=p.parse_args()
    if a.comparisons:
        stocks=sorted(a.comparisons.glob("littlebit-*/stock/trace.json"),key=lambda p:p.stat().st_mtime,reverse=True)
        for t in stocks:
            tr=json.loads(t.read_text(encoding="utf-8"))
            if (tr.get("case")=="stock" and tr.get("completed_updates") and tr.get("calibration")
                    and all("tokens" in r for r in tr["calibration"]) and (t.parent/"prepare.json").is_file()):
                a.stock=t.parent
                break
        if a.stock is None: p.error("no completed stock trace found")
    if a.out.exists(): p.error("fresh output path required")
    if a.fit_steps<1 or a.threads<1: p.error("positive fit steps and threads required")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads)
    started=time.monotonic()
    result={"status":"running","protocol":"private-correction-v1","pairs":[],"fit_steps":a.fit_steps,
            "stock_directory":str(a.stock.resolve()),"rank_fractions":[0.5,1.0],
            "capture_teacher":"native INT8 training trajectory; local references Raw BF16 and CPU native INT8",
            "limitations":"image-key-disjoint stock training captures, not independent LoRA holdout; tiny sample; CPU only"}
    try:
        trace=json.loads((a.stock/"trace.json").read_text(encoding="utf-8"))
        prep=json.loads((a.stock/"prepare.json").read_text(encoding="utf-8"))
        if trace["case"]!="stock" or not trace.get("completed_updates") or prep.get("status")!="stock_verified":
            raise ValueError("completed V5 stock arm required")
        result["raw_sha256"]=codec.sha256(a.raw)
        if result["raw_sha256"]!=prep["raw_sha256"]: raise ValueError("raw checkpoint mismatch")
        split=pc.split_captures(trace)
        result["split"]={k:[{"step":r["step"],"file":r["file"],"sha256":codec.sha256(a.stock/r["file"]),
                            "item":next(s["items"][0] for s in trace["steps"] if s["step"]==r["step"])} for r in v] for k,v in split.items()}
        atomic_json(a.out,result)
        repo=Path(__file__).resolve().parents[3]
        int8=codec.import_file("correction_int8",repo/"src/fizgig/modules/int8_train.py")
        with safe_open(str(a.raw),framework="pt",device="cpu") as f:
            for keys in PAIRS:
                print("Offline correction:",keys,flush=True)
                weights=[f.get_tensor(k) for k in keys]
                row=run_pair(weights,keys,a.stock,trace,split,int8,a.fit_steps)
                result["pairs"].append(row)
                atomic_json(a.out,result)
                del weights
        if codec.sha256(a.raw)!=result["raw_sha256"]: raise ValueError("raw changed during experiment")
        for records in result["split"].values():
            for r in records:
                if codec.sha256(a.stock/r["file"])!=r["sha256"]:
                    raise ValueError("calibration file changed during experiment")
        result["status"]="completed"
    except Exception as exc:
        result.update(status="error",error=str(exc))
        raise
    finally:
        result["seconds"]=time.monotonic()-started
        atomic_json(a.out,result)


if __name__=="__main__":
    main()
