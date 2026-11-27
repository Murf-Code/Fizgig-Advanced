from pathlib import Path
import copy
import sys
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research/krea2_littlebit"))
import qat_core as q
import binary_qat_stabilization as f


def test_scale_only_and_checkpoint_grad():
    torch.manual_seed(42)
    a,b=torch.randn(7,31),torch.randn(37,7)
    w=b@a
    x,g=torch.randn(5,31),torch.randn(5,37)
    fit=[(x,g,x@w.T,g@w)]
    initial=nn.ModuleList([q.Path(a,b)])
    ps=copy.deepcopy(initial)
    captured=[]
    def callback(paths,stats):
        assert torch.is_grad_enabled()
        z=x.detach().requires_grad_(True)
        y=q.actions(paths[0].components(),z,g)[0]
        dx,=torch.autograd.grad(y,z,g)
        assert torch.isfinite(dx).all()
        captured.append(stats)
    f.optimize(ps,w,fit,0.,steps=2,callback=callback)
    assert captured[-1]["step"]==2
    assert captured[-1]["net_flip_fraction"]==[0.,0.]
    assert captured[-1]["cumulative_flips_per_element"]==[0.,0.]
    assert torch.equal(ps[0].a,initial[0].a)
    assert torch.equal(ps[0].b,initial[0].b)
    assert not torch.equal(ps[0].latent,initial[0].latent)
    assert ps[0].a.requires_grad


def test_warmup_and_matched_sampling():
    torch.manual_seed(43)
    a,b=torch.randn(7,31),torch.randn(37,7)
    w=b@a
    x,g=torch.randn(5,31),torch.randn(5,37)
    fit=[(x,g,x@w.T,g@w)]
    initial=nn.ModuleList([q.Path(a,b)])
    results=[]
    for _ in range(2):
        ps=copy.deepcopy(initial)
        history=[]
        f.optimize(ps,w,fit,.0001,warmup=20,steps=21,
                   callback=lambda p,s:history.append(s))
        assert history[0]["step"]==20
        assert history[0]["net_flip_fraction"]==[0.,0.]
        assert history[1]["step"]==21
        assert history[1]["sign_grad_norm"]>0
        results.append([v.detach().clone() for v in ps.parameters()])
    assert all(torch.equal(a,b) for a,b in zip(*results))


def test_zero_energy_rejected():
    import pytest
    a,b=torch.randn(3,7),torch.randn(5,3)
    x,g=torch.zeros(2,7),torch.zeros(2,5)
    with pytest.raises(ValueError,match="zero teacher energy"):
        f.optimize(nn.ModuleList([q.Path(a,b)]),b@a,
                   [(x,g,g,x)],.001,steps=1)
