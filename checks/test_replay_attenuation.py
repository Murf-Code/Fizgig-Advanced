import pytest
import torch
from fizgig.training.muon import BatchedMuon
from fizgig.training.replay_forks import Pair
from fizgig.training.replay_attenuation import allocated_step, rotation


def build():
    g = torch.Generator().manual_seed(7)
    pairs = []
    for i in range(4):
        a = torch.nn.Parameter(torch.randn(2,12,generator=g)*.1)
        b = torch.nn.Parameter(torch.randn(12,2,generator=g)*.1)
        for p in (a,b):
            p.grad = torch.randn(p.shape,generator=g)
        pairs.append(Pair(str(i),i,a,b,1.))
    opt = BatchedMuon([p for x in pairs for p in (x.A,x.B)],lr=.001)
    return pairs,opt


def test_rotation_balanced():
    a,b = rotation(range(28),1),rotation(range(28),2)
    assert sum(v==.5 for v in a.values()) == 14
    assert all(a[i]+b[i] == 1.5 for i in a)


def test_allocation_same_parent_exact_and_momentum_preserved():
    results, buffers = {}, {}
    for mode in ("full","rotate","uniform","compensated"):
        pairs,opt=build()
        results[mode]=allocated_step(pairs,opt,mode,1)
        buffers[mode]=[opt.state[p]["momentum_buffer"].clone() for x in pairs for p in (x.A,x.B)]
        assert results[mode]["relative_gap"] <= .001
    assert results["uniform"]["actual_movement"] == pytest.approx(results["rotate"]["actual_movement"],rel=.001)
    assert results["compensated"]["actual_movement"] == pytest.approx(results["full"]["actual_movement"],rel=.001)
    assert all(torch.equal(a,b) for mode in buffers for a,b in zip(buffers[mode],buffers["full"]))
    pairs,opt=build()
    opt.step()
    p2,o2=build()
    allocated_step(p2,o2,"full",1)
    assert all(torch.equal(a,b) for x,y in zip(pairs,p2) for a,b in ((x.A,y.A),(x.B,y.B)))
