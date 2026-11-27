from pathlib import Path
import copy
import math
import sys
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research/krea2_littlebit"))
import binary_qat_convergence as h
import binary_qat_stabilization as f
import qat_core as q


def test_schedule():
    lr=.0001
    assert h.rate(20,lr,20,"cosine",None)==0.
    assert h.rate(21,lr,20,"cosine",None)==lr
    assert h.rate(100,lr,20,"cosine",None)==lr
    assert math.isclose(h.rate(300,lr,20,"cosine",None),lr*.525)
    assert math.isclose(h.rate(500,lr,20,"cosine",None),lr*.05)
    assert h.rate(1000,lr,20,"cosine",None)==h.rate(500,lr,20,"cosine",None)
    assert h.rate(100,lr,20,"constant",100)==lr
    assert h.rate(101,lr,20,"constant",100)==0.


def example():
    torch.manual_seed(42)
    w=torch.randn(37,31)
    x,g=torch.randn(5,31),torch.randn(5,37)
    p=nn.ModuleList([q.Path(torch.randn(7,31),torch.randn(37,7)),
                     q.Path(torch.randn(3,31),torch.randn(37,3))])
    return p,w,[(x,g,x@w.T,g@w)]


def test_constant_prefix_matches_f5():
    p,w,fit=example()
    a,b=copy.deepcopy(p),copy.deepcopy(p)
    f.optimize(a,w,fit,.0001,warmup=20,steps=100)
    h.optimize(b,w,fit,.0001,warmup=20,steps=100,schedule="cosine")
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))


def test_freeze_keeps_shadow_signs_and_adam_from_moving():
    p,w,fit=example()
    snapshots={}
    def callback(ps,stats):
        snapshots[stats["step"]]=(
            [t.detach().clone() for p in ps for t in (p.a,p.b)],
            [t.detach().clone() for p in ps for t in (p.row,p.col,p.latent)],
            stats)
    h.optimize(p,w,fit,.0001,warmup=20,steps=105,freeze_after=100,callback=callback)
    a,b=snapshots[100],snapshots[105]
    assert all(torch.equal(x,y) for x,y in zip(a[0],b[0]))
    assert any(not torch.equal(x,y) for x,y in zip(a[1],b[1]))
    assert a[2]["cumulative_flips_per_element"]==b[2]["cumulative_flips_per_element"]
    assert b[2]["sign_grad_norm"]==0.
    assert b[2]["sign_lr_current"]==0.
    assert not b[2]["sign_active"]
