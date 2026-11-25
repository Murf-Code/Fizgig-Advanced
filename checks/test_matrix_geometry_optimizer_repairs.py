"""Research-branch regressions based on Grok's independent round-2 audit."""
import copy
import math
import pytest
import torch
from fizgig.training.lora_tsd import LoRATSD
from fizgig.training.polora.optim import Polora


@pytest.mark.parametrize("rank", [1,2,4,16])
@pytest.mark.parametrize("dtype", [torch.float32,torch.bfloat16])
@pytest.mark.parametrize("start,scale", [(0.,1.),(1e-9,4.)])
def test_tsd_stored_cap(rank,dtype,start,scale):
    g=torch.Generator().manual_seed(19+rank)
    m,n,r=96,48,16
    a=torch.nn.Parameter((torch.randn(r,n,generator=g)/math.sqrt(n)).to(dtype))
    b=torch.nn.Parameter((start*torch.randn(m,r,generator=g)).to(dtype))
    opt=LoRATSD([(a,b)],lr=0.02,max_delta_norm=0.1,log_every=0)
    opt.set_adapter_scales([scale])
    teacher=torch.randn(m,n,generator=g,dtype=torch.float64)/math.sqrt(n)
    basis=torch.randn(rank,n,generator=g,dtype=torch.float64)
    for _ in range(8):
        aa,bb=a.detach().double(),b.detach().double()
        c=torch.randn(4,rank,generator=g,dtype=torch.float64)@basis
        gw=(scale*bb@aa-teacher)@c.T@c/4
        a.grad=(scale*bb.T@gw).to(dtype)
        b.grad=(scale*gw@aa.T).to(dtype)
        opt.step()
        change=scale*(b.detach().double()@a.detach().double()-bb@aa).norm()
        assert torch.isfinite(change)
        assert change <= 0.1+1e-8
    assert b.detach().norm()>0


def test_tsd_resume_with_bf16_momentum():
    g=torch.Generator().manual_seed(4)
    a=torch.nn.Parameter(torch.randn(4,20,generator=g).bfloat16())
    b=torch.nn.Parameter(torch.zeros(32,4,dtype=a.dtype))
    o=LoRATSD([(a,b)],log_every=0)
    o.set_adapter_scales([2.])
    def step(opt,aa,bb,ga,gb):
        aa.grad,bb.grad=ga.clone(),gb.clone()
        opt.step()
    ga,gb=torch.randn(a.shape,generator=g).bfloat16(),torch.randn(b.shape,generator=g).bfloat16()
    step(o,a,b,ga,gb)
    ar,br=torch.nn.Parameter(a.detach().clone()),torch.nn.Parameter(b.detach().clone())
    resumed=LoRATSD([(ar,br)],log_every=0)
    resumed.set_adapter_scales([2.])
    resumed.load_state_dict(copy.deepcopy(o.state_dict()))
    assert resumed.state[br]["momentum_buffer"].dtype==torch.float32
    step(o,a,b,ga,gb)
    step(resumed,ar,br,ga,gb)
    torch.testing.assert_close(a,ar,rtol=0,atol=0)
    torch.testing.assert_close(b,br,rtol=0,atol=0)


def test_polora_chunk_skip_and_checkpoint():
    gen=torch.Generator().manual_seed(41)
    pairs=[(torch.nn.Parameter(torch.randn(4,24,generator=gen)),
            torch.nn.Parameter(torch.randn(32,4,generator=gen))) for _ in range(5)]
    other=[tuple(torch.nn.Parameter(p.detach().clone()) for p in pair) for pair in pairs]
    full=Polora(pairs=pairs,compile=False,chunk_memory_mib=0)
    small=Polora(pairs=other,compile=False,chunk_memory_mib=0.0001)
    full.record_every=small.record_every=1
    for step in range(4):
        for i,(p,q) in enumerate(zip(pairs,other)):
            for x,y in zip(p,q):
                grad=None if step%2 and i==2 else torch.randn(x.shape,generator=gen)
                x.grad=grad
                y.grad=None if grad is None else grad.clone()
        full.step()
        small.step()
        for p,q in zip(pairs,other):
            for x,y in zip(p,q):
                torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-6)
        for i in full.pair_state:
            for key in full.pair_state[i]:
                torch.testing.assert_close(full.pair_state[i][key],small.pair_state[i][key],rtol=2e-5,atol=2e-6)
        for key in ("sigma_A","sigma_B"):
            assert small.last_norms[key]==pytest.approx(full.last_norms[key],rel=2e-5)
        if step==1:
            # Change chunk budget across resume; serialized state layout is unchanged.
            sd=copy.deepcopy(small.state_dict())
            small=Polora(pairs=other,compile=False,chunk_memory_mib=0.002)
            small.load_state_dict(sd)
            small.record_every=1
