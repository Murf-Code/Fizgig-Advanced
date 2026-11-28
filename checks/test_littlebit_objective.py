from pathlib import Path
import copy
import hashlib
import json
import sys
import pytest
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research/krea2_littlebit"))
import binary_qat_objective as h
import binary_qat_convergence as c
import qat_core as q


def example():
    torch.manual_seed(42)
    w=torch.randn(37,31)
    x,g=torch.randn(5,31),torch.randn(5,37)
    p=nn.ModuleList([q.Path(torch.randn(7,31),torch.randn(37,7)),
                     q.Path(torch.randn(3,31),torch.randn(37,3))])
    return p,w,[(x,g,x@w.T,g@w)]


def test_control_matches_h4():
    p,w,fit=example()
    a,b=copy.deepcopy(p),copy.deepcopy(p)
    c.optimize(a,w,fit,.00005,warmup=20,steps=50)
    h.optimize(b,w,fit,.00005,warmup=20,steps=50)
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))


def test_all_objectives_finite_and_invalid_rejected():
    p,w,fit=example()
    for name,weights in h.ARMS:
        a=copy.deepcopy(p)
        hist=[]
        h.optimize(a,w,fit,.00005,warmup=2,steps=5,
                   weights=weights,callback=lambda ps,s:hist.append(s))
        assert hist[-1]["sign_grad_norm"]>0
        assert all(torch.isfinite(t).all() for t in a.parameters())
    with pytest.raises(ValueError,match="objective weights"):
        h.optimize(p,w,fit,.00005,weights=(.2,.2,.2),steps=1)


def test_fresh_bank_rejects_overlap_and_duplicates(tmp_path,monkeypatch):
    raw=tmp_path/"raw"
    raw.write_bytes(b"raw")
    sha=hashlib.sha256(b"raw").hexdigest()
    def make(name,ids):
        d=tmp_path/name
        d.mkdir()
        tr={"case":"stock","completed_updates":3,
            "steps":[{"step":i,"items":[item]} for i,item in enumerate(ids)],
            "calibration":[{"step":i,"file":f"capture{i}"} for i in range(len(ids))]}
        for r in tr["calibration"]:
            (d/r["file"]).write_bytes((name+r["file"]).encode())
        (d/"trace.json").write_text(json.dumps(tr),encoding="utf-8")
        (d/"prepare.json").write_text(json.dumps({"status":"stock_verified","raw_sha256":sha}),encoding="utf-8")
        return d,tr
    old,tr=make("old",["a","b","c"])
    overlap,_=make("overlap",["a","d","e"])
    new,_=make("new",["d","e","f"])
    monkeypatch.setattr(h,"FRESH_STOCK",overlap)
    with pytest.raises(ValueError,match="overlaps"):
        h.fresh_bank(old,tr,raw)
    monkeypatch.setattr(h,"FRESH_STOCK",new)
    assert len(h.fresh_bank(old,tr,raw)[2])==3
    (new/"capture0").write_bytes((old/"capture0").read_bytes())
    with pytest.raises(ValueError,match="duplicate"):
        h.fresh_bank(old,tr,raw)
