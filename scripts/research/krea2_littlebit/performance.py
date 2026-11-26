"""Separate native timed runs after the paired quality experiment."""
import argparse
import json
from pathlib import Path
import re
import sys
from fit_queue import arg,set_arg,atomic_json,run_sweep,SWAP_STEPS,OOM_EXIT


def compare_timing(a,b):
    if not a or not b or a.get("status")!="completed" or b.get("status")!="completed":
        return {"status":"insufficient_steps"}
    aa=[(s["items"],s["timestep"]) for s in a["steps"]]
    bb=[(s["items"],s["timestep"]) for s in b["steps"]]
    if aa!=bb:
        return {"status":"unpaired_input_schedule"}
    return {"status":"paired","stock_median_seconds":a["median_seconds"],
            "candidate_median_seconds":b["median_seconds"],
            "speed_ratio":a["median_seconds"]/b["median_seconds"]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("repo","archive","stock","candidate","out"):
        p.add_argument("--"+name,type=Path,required=True)
    argv=sys.argv[1:]; cut=argv.index("--")
    a=p.parse_args(argv[:cut]); original=argv[cut+1:]
    sys.path.insert(0,str(a.repo/"src"))
    stock_fit=json.loads((a.stock/"fit-report.json").read_text())
    cand_fit=json.loads((a.candidate/"fit-report.json").read_text())
    common=stock_fit["selected_swaps"]
    if cand_fit["selected_swaps"]!=common:
        raise ValueError("quality arms were not at common swaps")
    hold=(a.stock/"fit_attempts"/"holdout-original.json").read_bytes()
    config=Path(arg(original,"--dataset_config")).read_text(encoding="utf-8")
    root=a.out.parent/"performance"
    root.mkdir(exist_ok=False)
    report={"status":"running","common_swaps":common,"arms":{}}
    atomic_json(a.out,report)
    runner=Path(__file__).with_name("native_runner.py")
    def arm(label,schedule,compact):
        out=root/label
        out.mkdir()
        h=out/"holdout.json"; h.write_bytes(hold)
        d=out/"dataset.toml"
        line="holdout_file = "+json.dumps(h.as_posix(),ensure_ascii=False)
        text,n=re.subn(r"(?m)^holdout_file\s*=\s*[^\n]*$",lambda _:line,config)
        if n!=1: raise ValueError("invalid timing dataset config")
        d.write_text(text,encoding="utf-8")
        args=list(original)
        set_arg(args,"--dataset_config",d); set_arg(args,"--output_dir",out)
        set_arg(args,"--output_name","timing-"+label)
        command=[sys.executable,str(runner),"train","--repo",str(a.repo),"--case","timing"]
        if compact: command+=["--archive",str(a.archive)]
        try:
            rc=run_sweep([*command,"--",*args],out,schedule=schedule)
            fit=json.loads((out/"fit-report.json").read_text())
            data=json.loads((out/"timing.json").read_text()) if rc==0 else None
            report["arms"][label]={"exit_code":rc,"fit":fit,"timing":data}
            atomic_json(a.out,report)
            return rc,data
        except Exception as exc:
            report["arms"][label]={"status":"error","error":str(exc)}
            atomic_json(a.out,report)
            return 1,None
    try:
        sr,s=arm("stock-common",(common,),False)
        cr,c=arm("q6-common",(common,),True)
        report["common_comparison"]=compare_timing(s,c)
        lower=tuple(n for n in SWAP_STEPS if n<common)
        best=c
        if cr==0 and lower:
            br,b=arm("q6-residency",lower,True)
            if br==0: best=b
            elif br!=OOM_EXIT: best=None
        report["best_comparison"]=compare_timing(s,best)
        report["status"]="completed" if sr==cr==0 and report["common_comparison"]["status"]=="paired" and report["best_comparison"]["status"]=="paired" else "incomplete"
    except Exception as exc:
        report.update(status="error",error=str(exc))
    finally:
        atomic_json(a.out,report)
    # A timing failure is reported separately, not a failure of the completed quality arms.
    return 0


if __name__=="__main__":
    sys.exit(main())
