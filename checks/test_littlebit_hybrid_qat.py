from pathlib import Path
import copy
import sys
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research/krea2_littlebit"))
import hybrid_qat_screen as h
import binary_qat_stabilization as f
import qat_core as q
import mixed_factor_core as mix


def test_allocation():
    for m,n in ((1536,6144),(16384,6144),(37,31)):
        top=min(m,n,2048)
        for cap in (.5,1.):
            budget=int(cap*m*n//8)
            for pct in (25,50,75):
                alloc=h.allocation(m,n,budget,top,pct)
                if alloc is None:
                    assert m==37
                    continue
                r8,r,nb8=alloc
                assert r8+r<=top
                assert nb8==mix.pair_bytes(m,n,r8,8,8)
                assert nb8<=budget*pct//100
                assert nb8+q.size(m,n,r)<=budget


def test_fixed_path_fidelity_and_freeze():
    torch.manual_seed(123)
    m,n=37,31
    w=torch.randn(m,n)
    a,b=torch.randn(3,n)*.1,torch.randn(m,3)*.1
    fixed=(a,b,torch.ones(m),torch.ones(n),torch.ones(3))
    x,g=torch.randn(5,n),torch.randn(5,m)
    y,dx=q.actions(fixed,x,g)
    assert torch.allclose(y,x@(b@a).T,atol=1e-6)
    assert torch.allclose(dx,g@(b@a),atol=1e-6)
    original=[t.clone() for t in fixed]
    paths=nn.ModuleList([q.Path(torch.randn(7,n),torch.randn(m,7))])
    history=[]
    h.optimize(paths,w,[(x,g,x@w.T,g@w)],.0001,warmup=20,steps=21,
               callback=lambda ps,stats:history.append(stats),fixed=fixed)
    assert history[0]["net_flip_fraction"]==[0.,0.]
    assert history[1]["sign_grad_norm"]>0
    assert all(torch.equal(a,b) for a,b in zip(fixed,original))


def test_pure_binary_matches_lb_f():
    torch.manual_seed(42)
    w=torch.randn(37,31)
    x,g=torch.randn(5,31),torch.randn(5,37)
    fit=[(x,g,x@w.T,g@w)]
    p=nn.ModuleList([q.Path(torch.randn(7,31),torch.randn(37,7))])
    a,b=copy.deepcopy(p),copy.deepcopy(p)
    f.optimize(a,w,fit,.0001,steps=21,warmup=20)
    h.optimize(b,w,fit,.0001,steps=21,warmup=20)
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
