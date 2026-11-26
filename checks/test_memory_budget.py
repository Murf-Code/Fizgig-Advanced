import pytest
from fizgig.training.memory_budget import MemoryBudget, ScratchGroup, MIB


def test_disjoint_groups_use_max_not_sum():
    p=MemoryBudget(reserve_mib=0)
    groups=[ScratchGroup(0,20,30,100),ScratchGroup(1,10,40,90)]
    r=p.plan(180,groups,pending_state_bytes=40,outstanding_bytes=30)
    assert r.safe_bytes==100 and r.chunks==(3,2) and r.caps==(100,90)
    assert 30*r.chunks[0]<=r.selected_bytes and 40*r.chunks[1]<=r.selected_bytes


def test_pressure_shrinks_immediately_grows_conservatively():
    p=MemoryBudget(reserve_mib=0,growth_steps=3);g=[ScratchGroup(0,20,10,100)]
    assert p.plan(100,g).chunks==(10,)
    assert p.plan(30,g).chunks==(3,)
    assert p.plan(80,g).chunks==(3,)
    assert p.plan(60,g).chunks==(3,)
    assert p.plan(90,g).chunks==(6,)
    with pytest.raises(MemoryError): p.plan(5,g)
    assert p.last_plan is None


def test_refuses_single_item_even_if_group_cap_is_smaller():
    p=MemoryBudget(reserve_mib=1)
    with pytest.raises(MemoryError):p.plan(10*MIB,[ScratchGroup(0,1,4*MIB,3*MIB)])
    with pytest.raises(ValueError):p.plan(10*MIB,[ScratchGroup(0,1,1,0)])


@pytest.mark.parametrize('storage',['fp32','bf16','int8','splitmax'])
def test_muon_preflight_and_fixed_equivalence(storage):
    import torch
    from fizgig.training.muon_me import MuonME
    from fizgig.training.muon_me_compressed import CompressedMuonME
    cls=MuonME if storage=='fp32' else CompressedMuonME
    kw={} if storage=='fp32' else {'momentum_storage':storage}
    a=torch.nn.Parameter(torch.ones(8,8));b=torch.nn.Parameter(a.detach().clone())
    fixed=cls([a],**kw);auto=cls([b],auto_memory_budget=True,memory_reserve_mib=0,**kw)
    a.grad=torch.ones_like(a);b.grad=a.grad.clone()
    auto._memory_free_bytes=lambda device:0
    with pytest.raises(MemoryError):auto.step()
    assert not auto.state and torch.equal(a,b)
    auto._memory_free_bytes=lambda device:1024*MIB
    # After a pressure refusal, growth requires consecutive sufficient snapshots.
    for _ in range(2):
        with pytest.raises(MemoryError):auto.step()
    fixed.step();auto.step()
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    plan=next(iter(auto.memory_plans.values()))
    assert plan.pending_state_bytes>0
    fixed.step();auto.step()
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert next(iter(auto.memory_plans.values())).pending_state_bytes==0
