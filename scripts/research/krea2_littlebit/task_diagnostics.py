"""Diagnostics of actual exported LoRA movement, without dense BA materialization."""
import math
from pathlib import Path
import torch
from safetensors import safe_open
from fizgig.training.matrix_geometry import delta_norm


def vector_compare(a,b):
    a,b=a.double(),b.double()
    aa=float(a.square().sum()); bb=float(b.square().sum()); ab=float((a*b).sum())
    return {"stock_norm":math.sqrt(aa),"candidate_norm":math.sqrt(bb),
            "difference_norm":math.sqrt(max(0.,aa+bb-2*ab)),
            "cosine":max(-1.,min(1.,ab/math.sqrt(aa*bb))) if aa>0 and bb>0 else None}


@torch.no_grad()
def movement(stock,candidate,calibration=None):
    """Compare exported factors in effective matrix space, including alpha/rank."""
    if set(stock)!=set(candidate):
        raise ValueError("adapter inventories differ")
    out={}
    for k,(kind,a,b,scale) in stock.items():
        other,c,d,scale2=candidate[k]
        if kind!="lora" or other!="lora" or a.shape!=c.shape or b.shape!=d.shape:
            raise ValueError(f"incompatible adapter {k}")
        a,b,c,d=a.float(),b.float()*scale,c.float(),d.float()*scale2
        sn=float(delta_norm(a,torch.zeros_like(b),a,b))
        cn=float(delta_norm(c,torch.zeros_like(d),c,d))
        dn=float(delta_norm(a,b,c,d))
        cosine=(sn*sn+cn*cn-dn*dn)/(2*sn*cn) if sn and cn else None
        row={"stock_norm":sn,"candidate_norm":cn,"difference_norm":dn,
             "effective_update_cosine":max(-1.,min(1.,cosine)) if cosine is not None else None}
        key=k+".weight.x"
        if calibration is not None and key in calibration.keys():
            x=calibration.get_tensor(key).float()
            row["on_stock_calibration_rows"]=vector_compare((x@a.T)@b.T,(x@c.T)@d.T)
        out[k]=row
    return out


def gradients(stock,candidate,expected):
    with safe_open(str(stock),framework="pt",device="cpu") as f, safe_open(str(candidate),framework="pt",device="cpu") as g:
        keys={f"{k}.{i}" for k in expected for i in (0,1)}
        if set(f.keys())!=keys or set(g.keys())!=keys:
            raise ValueError("incomplete task gradient manifest")
        return {k:vector_compare(f.get_tensor(k),g.get_tensor(k)) for k in sorted(keys)}
