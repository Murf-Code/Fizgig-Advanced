import random
import copy

import numpy as np
import pytest
import torch

from fizgig.training.muon import BatchedMuon, Muon
from fizgig.training.replay_forks import MovementCurve, Pair, capture, match


def test_first_branch_stops_before_cancellation():
    c = MovementCurve([(0,np.array([0.]),np.array([-6.]),np.array([9.]))],{0:1.})
    assert c.first_increasing_end() == pytest.approx(1/3)
    assert c.norm(2/3) == pytest.approx(0, abs=1e-12)
    assert c.solve(.5,c.first_increasing_end()) < 1/3


def test_decay_floor_and_initial_decrease():
    c = MovementCurve([(0,np.array([1.]),np.array([-2.]),np.array([0.]))],{0:1.})
    with pytest.raises(ValueError,match='initially decreases'):
        c.first_increasing_end()
    c = MovementCurve([(0,np.array([1.]),np.array([2.]),np.array([0.]))],{0:1.})
    assert c.solve(2,1) == pytest.approx(.5)
    with pytest.raises(ValueError,match='unreachable'):
        c.solve(.5,1)


@pytest.mark.parametrize('cls',[BatchedMuon,Muon])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('decay',[0.,.01])
def test_native_fork_and_rollback(cls,dtype,decay):
    torch.manual_seed(42)
    gen=torch.Generator().manual_seed(123)
    pairs=[Pair(str(i),i,torch.nn.Parameter(torch.randn(3,9,dtype=dtype)),
                torch.nn.Parameter(torch.randn(7,3,dtype=dtype)),.5) for i in range(3)]
    params=[p for pair in pairs for p in (pair.A,pair.B)]
    opt=cls(params,lr=.00045,weight_decay=decay,adjust_lr_fn='match_rms_adamw')
    for p in params: p.grad=torch.randn_like(p)
    opt.step()
    for p in params: p.grad=torch.randn_like(p)
    old=[p.detach().clone() for p in params]
    buffers=[opt.state[p]['momentum_buffer'].clone() for p in params]
    ref_params=[torch.nn.Parameter(p.detach().clone()) for p in params]
    ref=cls(ref_params,lr=.00045,weight_decay=decay,adjust_lr_fn='match_rms_adamw')
    ref.load_state_dict(copy.deepcopy(opt.state_dict()))
    for p,r in zip(params,ref_params): r.grad=p.grad.clone()
    ref.step()
    py=random.getstate(); np_state=np.random.get_state(); rng=torch.get_rng_state(); gstate=gen.get_state()
    with capture(pairs,opt,generators=(gen,)) as fork:
        post=[opt.state[p]['momentum_buffer'].clone() for p in params]
        for m in (0.,.5,1.):
            multipliers={i:m for i in range(3)}
            actual,_=fork.achieved(multipliers)
            dense=sum(float((pair.scale*(pair.B.detach().double()@pair.A.detach().double()-old[2*i+1].double()@old[2*i].double())).square().sum()) for i,pair in enumerate(pairs))**.5
            assert actual == pytest.approx(dense,rel=1e-10,abs=1e-12)
            assert all(torch.equal(opt.state[p]['momentum_buffer'],b) for p,b in zip(params,post))
            if m==1.:
                assert all(torch.equal(p,r) for p,r in zip(params,ref_params))
                assert all(torch.equal(opt.state[p]['momentum_buffer'],ref.state[r]['momentum_buffer']) for p,r in zip(params,ref_params))
        alloc={'uniform':{0:1.,1:1.,2:1.},'static':{0:.5,1:1.,2:1.},'adaptive':{0:.5,1:1.,2:1.}}
        result=match(fork,alloc,minimum=0,atol=1e-8,rtol=.001)
        if dtype==torch.float32 and decay==0.:
            assert result['matched'],result
        if 'arms' in result:
            assert result['arms']['static']==result['arms']['adaptive']
        random.random(); np.random.rand(); torch.rand(2); torch.rand(2,generator=gen)
    assert all(torch.equal(p,o) for p,o in zip(params,old))
    assert all(torch.equal(opt.state[p]['momentum_buffer'],b) for p,b in zip(params,buffers))
    assert random.getstate()==py and np.array_equal(np.random.get_state()[1],np_state[1])
    assert torch.equal(torch.get_rng_state(),rng) and torch.equal(gen.get_state(),gstate)


def test_exception_restores_parent_and_empty_state():
    A=torch.nn.Parameter(torch.ones(2,3)); B=torch.nn.Parameter(torch.zeros(4,2))
    A.grad=torch.ones_like(A); B.grad=torch.ones_like(B)
    opt=BatchedMuon([A,B])
    with pytest.raises(RuntimeError,match='scoring failed'):
        with capture([Pair('zero',0,A,B,1.)],opt) as fork:
            fork.apply({0:.5})
            raise RuntimeError('scoring failed')
    assert torch.equal(A,torch.ones_like(A)) and torch.equal(B,torch.zeros_like(B))
    assert not opt.state
