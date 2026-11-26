"""Opt-in native trainer observer installed only by the LittleBit worker."""
import hashlib
import json
from pathlib import Path
import random
import torch
from safetensors.torch import save_file
from fit_queue import atomic_json

HORIZONS=(1,16,200)
CALIBRATION_STEPS=(1,2,3,4)
ROWS=16


def fingerprint(value):
    h=hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            x=v.detach().cpu().contiguous()
            h.update(str((x.dtype,tuple(x.shape))).encode())
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v):
                add(k); add(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(str(len(v)).encode())
            for x in v: add(x)
        elif isinstance(v,(str,int,float,bool,type(None))):
            h.update(repr(v).encode()+b'\0')
        else:
            raise TypeError(f"unsupported trace object {type(v)}")
    add(value)
    return h.hexdigest()


class Trace:
    def __init__(self,case,stock_trace=None):
        self.case=case
        self.reference=json.loads(Path(stock_trace).read_text(encoding="utf-8")) if stock_trace else None
        self.data={"case":case,"horizons":list(HORIZONS),"steps":[],"calibration":[],"checkpoints":{}}
        self.active=False
        self.rows={}
        self.handles=[]

    def seed(self,seed):
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def start(self,net,out,seed):
        from fizgig.families.lora import TRAINABLE
        self.net=net; self.out=Path(out)
        self.targets={k:w.adapters[TRAINABLE] for k,w in net.wrapped.items() if TRAINABLE in w.adapters}
        expected=set(net.targets)
        if set(self.targets)!=expected or not expected:
            raise RuntimeError(f"incomplete trainable manifest: {sorted(expected-set(self.targets))}")
        if any(len(a)!=2 or a[1].weight.count_nonzero().item()!=0 for a in self.targets.values()):
            raise RuntimeError("expected fresh standard zero-up LoRA")
        self.data.update(seed=seed,expected_modules=sorted(expected),
                         initial_sha256=fingerprint(net.state_dict(torch.float32)))
        if self.reference and any(self.reference[k]!=self.data[k] for k in ("seed","expected_modules","initial_sha256")):
            raise RuntimeError("paired initial adapter or target manifest mismatch")
        if self.case=="stock":
            from native_codec import ELIGIBLE
            for k,w in net.wrapped.items():
                if k+".weight" in ELIGIBLE:
                    self.handles.append(w.register_forward_hook(self.hook(k+".weight")))
        self.write()

    def hook(self,key):
        def fn(module,args,y):
            if not self.active or not torch.is_grad_enabled() or not y.requires_grad:
                return
            x=args[0].detach().reshape(-1,args[0].shape[-1])
            if x.shape[0]!=y.numel()//y.shape[-1]:
                raise RuntimeError("activation/cotangent rows differ")
            idx=torch.linspace(0,len(x)-1,min(ROWS,len(x)),device=x.device).long()
            sample=x.index_select(0,idx).cpu().contiguous()
            step=len(self.data["steps"])
            def backward(dy):
                if self.active and len(self.data["steps"])==step and key+".x" not in self.rows:
                    grad=dy.detach().reshape(-1,dy.shape[-1]).index_select(0,idx).cpu().contiguous()
                    self.rows[key+".x"]=sample
                    self.rows[key+".dy"]=grad
            y.register_hook(backward)
        return fn

    def begin(self,batch,latents,cond,gen):
        i=len(self.data["steps"])
        if self.active: raise RuntimeError("unfinished calibration step")
        sig=fingerprint({"keys":list(map(str,batch.get("item_keys") or [])),
                         "latents":latents,"cond":cond,"noise_rng":gen.get_state()})
        if self.reference and (i>=len(self.reference["steps"]) or self.reference["steps"][i]["input_sha256"]!=sig):
            raise RuntimeError(f"paired batch/conditioning/noise RNG mismatch at update {i+1}")
        self.data["steps"].append({"step":i+1,"input_sha256":sig,
                                   "items":list(map(str,batch.get("item_keys") or []))})
        self.active=self.case=="stock" and i+1 in CALIBRATION_STEPS
        self.rows={}

    def pre_update(self):
        if not self.data["steps"]: raise RuntimeError("optimizer update without trace")
        step=len(self.data["steps"])
        if step in HORIZONS:
            grads={}
            for k,ad in self.targets.items():
                for i,p in enumerate(ad.parameters()):
                    if p.grad is None:
                        raise RuntimeError(f"missing task gradient: {k}/{i}")
                    grads[f"{k}.{i}"]=p.grad.detach().cpu().contiguous()
            save_file(grads,str(self.out/f"task-grad-{step:06d}.safetensors"))
        if self.active:
            from native_codec import ELIGIBLE
            missing=sorted(ELIGIBLE-{k[:-2] for k in self.rows if k.endswith(".x")})
            if missing or len(self.rows)!=2*len(ELIGIBLE):
                raise RuntimeError(f"incomplete real calibration capture: {missing}")
            filename=f"calibration-{step:06d}.safetensors"
            save_file(self.rows,str(self.out/filename))
            self.data["calibration"].append({"step":step,"file":filename})
        self.active=False
        self.rows={}

    def end(self,info,loss):
        step=len(self.data["steps"])
        self.data["steps"][-1].update(timestep=float(info.get("t",0.5)),loss=float(loss.detach()))
        if self.reference and self.reference["steps"][step-1]["timestep"]!=self.data["steps"][-1]["timestep"]:
            raise RuntimeError("paired timestep mismatch")
        if step in HORIZONS:
            filename=f"task-lora-{step:06d}.safetensors"
            self.net.save(str(self.out/filename))
            self.data["checkpoints"][str(step)]=filename
        self.write()
        return step>=HORIZONS[-1]

    def finish(self):
        if self.reference and len(self.reference["steps"])!=len(self.data["steps"]):
            raise RuntimeError("paired training lengths differ")
        if not self.data["steps"]:
            raise RuntimeError("no actual optimizer updates")
        self.data["completed_updates"]=len(self.data["steps"])
        self.write()
        for h in self.handles: h.remove()
        self.handles.clear()

    def write(self):
        atomic_json(self.out/"trace.json",self.data)
