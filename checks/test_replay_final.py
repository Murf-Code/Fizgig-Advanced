import math
import torch
import pytest

from fizgig.training.replay_final import active_blocks, multiplier
from fizgig.training.muon import BatchedMuon, Muon

def test_global_rotation():
    for seed in (42,43):
        masks = [active_blocks(28,k,seed==43) for k in range(1,66)]
        assert all(len(a)==14 for a in masks)
        assert all(a.isdisjoint(b) and len(a|b)==28 for a,b in zip(masks,masks[1:]))
        assert sorted(sum(i in a for a in masks) for i in range(28)) == [32]*14+[33]*14

@pytest.mark.parametrize("cls",[BatchedMuon,Muon])
def test_multiplier_and_skipped_state(cls):
    g=torch.Generator().manual_seed(3)
    initial=torch.randn(4,8,generator=g)
    grad=torch.randn(4,8,generator=g)
    data=[]
    for c in (1.,math.sqrt(2)):
        a,b=[torch.nn.Parameter(initial.clone()) for _ in range(2)]
        opt=cls([a,b],lr=.001,weight_decay=.01)
        a.grad=grad.clone();b.grad=grad.clone()
        opt.step()
        before=a.detach().clone()
        skip=b.detach().clone()
        mom=opt.state[b]["momentum_buffer"].clone()
        b.grad=None;a.grad=grad.clone()
        rows=[]
        opt.step(learning_multiplier=c,proposal_sink=lambda p,u,alpha,f: rows.append((u[0].clone(),alpha)))
        assert torch.equal(b,skip)
        assert torch.equal(opt.state[b]["momentum_buffer"],mom)
        assert len(rows)==1
        assert torch.allclose(a,before*(1-.001*.01)+rows[0][0]*rows[0][1])
        data.append((rows[0],opt.state[a]["momentum_buffer"].clone()))
    assert torch.equal(data[0][0][0],data[1][0][0])
    assert data[1][0][1]==pytest.approx(math.sqrt(2)*data[0][0][1])
    assert torch.equal(data[0][1],data[1][1])
    assert multiplier("compensated",14)==math.sqrt(2)


def test_full_ema_for_initially_suspended_factors():
    from fizgig.training.ema import EMAWeights
    a,b=[torch.nn.Parameter(torch.tensor([1.])) for _ in range(2)]
    b.requires_grad_(False)
    class Network:
        def parameters(self):
            return [a,b]
    ema=EMAWeights(Network(),.98,include_inactive=True)
    assert len(ema.params)==2
    with torch.no_grad():
        b.requires_grad_(True)
        b.add_(2.)
    ema.update()
    raw=b.clone()
    ema.swap_in()
    assert 1 < b.item() < 3
    ema.swap_out()
    assert torch.equal(b,raw)


def test_dynamic_multiplier_is_bounded_and_full_probe_is_native():
    from fizgig.training.replay_final import DYNAMIC_CAP
    assert multiplier("dynamic",28)==1.
    assert multiplier("dynamic",14)==math.sqrt(2)
    assert multiplier("dynamic",1)==DYNAMIC_CAP==1.5
    assert all(1.<=multiplier("dynamic",n)<=1.5 for n in range(1,29))
    assert all(multiplier("dynamic",n)>=multiplier("dynamic",n+1) for n in range(1,28))
    assert all(multiplier("existing",n)==1. for n in range(1,29))
    with pytest.raises(ValueError):
        multiplier("dynamic",0)
