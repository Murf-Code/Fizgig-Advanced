"""H3: stock-only calibration collector; no compression or paired training arms."""
import hashlib
import math
from pathlib import Path
import torch
from safetensors.torch import save_file
from research_trace import Trace
from valid_tokens import valid_rows
from fit_queue import atomic_json

KEYS=("blocks.0.attn.wk.weight","blocks.14.attn.wk.weight","blocks.0.mlp.up.weight")
COUNT=80
ROWS=32
MAX_UPDATES=200
FIT_SIZES=(2,8,16,32,64)


def partition(records,steps):
    by_step={s["step"]:s for s in steps}
    ids=[by_step[r["step"]]["items"][0] for r in records]
    if len(records)!=COUNT or len(set(ids))!=COUNT:
        raise ValueError("need 80 distinct captured image IDs")
    order=sorted(zip(ids,records),key=lambda z:hashlib.sha256(("LB-H3-v1:"+z[0]).encode()).digest())
    # Hash order gives deterministic nested prefixes, not timestep-selected examples.
    return {k:[r["step"] for _,r in part] for k,part in
            (("fit",order[:64]),("development",order[64:72]),("confirmation",order[72:]))}


class CaptureBank(Trace):
    def __init__(self):
        super().__init__("stock")
        self.data.update(protocol="lb-h3-stock-bank-v1",horizons=[],capture_keys=list(KEYS),
                         requested_captures=COUNT,rows_per_capture=ROWS)
        self.seen=set()

    def start(self,net,out,seed):
        # Reuse the existing fresh-adapter/manifest checks. Install only
        # the three targeted projection hooks, not the other 221.
        super().start(net,out,seed)
        for h in self.handles: h.remove()
        self.handles.clear()
        for key in KEYS:
            self.handles.append(net.wrapped[key[:-7]].register_forward_hook(self.hook(key)))
        self.write()

    def begin(self,batch,latents,cond,gen):
        super().begin(batch,latents,cond,gen)
        items=self.data["steps"][-1]["items"]
        if len(items)!=1 or not items[0]:
            raise RuntimeError("capture bank requires batch size one and identified images")
        self.active=items[0] not in self.seen and len(self.seen)<COUNT

    def set_layout(self,image_tokens,text_mask):
        if self.active:
            self.layout=valid_rows(image_tokens,text_mask,ROWS)

    def pre_update(self):
        if self.active:
            expected={key+suffix for key in KEYS for suffix in (".x",".dy")}
            if set(self.rows)!=expected or self.layout is None:
                raise RuntimeError("incomplete H3 capture: "+str(sorted(expected-set(self.rows))))
            if any(not torch.isfinite(t).all() for t in self.rows.values()):
                raise RuntimeError("nonfinite H3 capture")
            step=len(self.data["steps"])
            filename=f"calibration-{step:06d}.safetensors"
            path=self.out/filename
            save_file(self.rows,str(path))
            with path.open("rb") as f:
                digest=hashlib.file_digest(f,"sha256").hexdigest()
            self.data["calibration"].append({"step":step,"file":filename,"tokens":self.layout[1],
                     "sha256":digest})
            self.seen.add(self.data["steps"][-1]["items"][0])
        self.active=False
        self.rows={}

    def end(self,info,loss):
        step=len(self.data["steps"])
        t=float(info.get("t",float("nan"))); lv=float(loss.detach())
        if not math.isfinite(t) or not 0<=t<=1 or not math.isfinite(lv):
            raise RuntimeError("invalid training capture timestep/loss")
        self.data["steps"][-1].update(timestep=t,loss=lv)
        self.write()
        if step%8==0 or len(self.seen)==COUNT:
            print(f"[LB-H3] {len(self.seen)}/{COUNT} unique images captured, update {step}",flush=True)
        return len(self.seen)>=COUNT or step>=MAX_UPDATES

    def finish(self):
        if len(self.seen)!=COUNT:
            raise RuntimeError(f"Only {len(self.seen)} distinct images captured; need {COUNT}. Use a larger diverse dataset.")
        self.data["split_steps"]=partition(self.data["calibration"],self.data["steps"])
        super().finish()
        atomic_json(self.out/"capture-bank.json",{
            "status":"completed","protocol":self.data["protocol"],"keys":list(KEYS),
            "split_steps":self.data["split_steps"],"steps":self.data["steps"],
            "calibration":self.data["calibration"],"fit_sizes":list(FIT_SIZES),
            "scope":"stock INT8 standard LoRA training signals; disjoint item IDs, not guaranteed source-image independence"})
