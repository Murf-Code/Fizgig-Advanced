"""Adapted from Grok's independent Round 5 adversarial suite (fizgig_round5.zip).
Run alongside test_polora_kernel_efficiency.py. CPU bitwise gate; CUDA gate is separate.
"""
import io
import random
import pytest
import torch
from fizgig.training.polora.optim import Polora
from _polora_reference_2323a51.optim import Polora as Reference

SHAPES = ([(8,48,96,torch.float32)]*6 + [(8,96,24,torch.float32)]*5
          + [(8,32,32,torch.float32)]*4 + [(4,4,40,torch.float32)]*3
          + [(1,16,12,torch.float32)]*2 + [(6,40,56,torch.bfloat16)]*5
          + [(8,20,300,torch.float32)])
KW = dict(lr=.02, magnitude_scale=3., compile=False, log_norms_every=3)

@pytest.fixture(autouse=True)
def precision():
    old = torch.get_float32_matmul_precision()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    det = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(det, warn_only=warn)
        torch.set_float32_matmul_precision(old)
        torch.backends.cuda.matmul.allow_tf32 = tf32

def pairs(shapes=SHAPES):
    g = torch.Generator().manual_seed(0)
    return [(torch.nn.Parameter((torch.randn(r,i,generator=g)/i**.5).to(dt)),
             torch.nn.Parameter(torch.zeros(o,r,dtype=dt))) for r,i,o,dt in shapes]

def bits(a,b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    dt = {torch.float32:torch.int32, torch.bfloat16:torch.int16}[a.dtype]
    assert torch.equal(a.detach().contiguous().view(dt), b.detach().contiguous().view(dt))

def same(p,o,q,ref):
    for pp,qq in zip(p,q):
        for a,b in zip(pp,qq):
            bits(a,b)
    assert o.pair_state.keys() == ref.pair_state.keys()
    for i,st in o.pair_state.items():
        assert st.keys() == ref.pair_state[i].keys()
        for k,v in st.items():
            bits(v,ref.pair_state[i][k])
    assert o.last_norms == ref.last_norms
    assert o._step_count == ref._step_count

def budget(c,r=8,i=20,o=300):
    return (c+.25)*4*(16*r*(i+o)+32*r*r+8*(i+o))/2**20

def mask(t,n):
    rng = random.Random(3000+t)
    m = ([True]*n if t%6==0 else [k%2==0 for k in range(n)] if t%6==1
         else [k not in (0,6,11,15) for k in range(n)] if t%6==3
         else [k>=n-3 for k in range(n)] if t%6==4
         else [rng.random()<.6 for _ in range(n)])
    if not any(m): m[0]=True
    return m

def run(o,p,t,active=None):
    g = torch.Generator().manual_seed(10000+t)
    active = mask(t,len(p)) if active is None else active
    for k,(a,b) in enumerate(p):
        for v in (a,b):
            grad = (torch.randn(v.shape,generator=g)*.01).to(v.dtype)
            v.grad = grad if active[k] else None
    o.step()

@pytest.mark.parametrize("mem",[0.,budget(1),budget(2),budget(3)])
def test_mixed_groups_and_masks(mem):
    p,q=pairs(),pairs()
    o,ref=Polora(pairs=p,chunk_memory_mib=mem,**KW),Reference(pairs=q,chunk_memory_mib=mem,**KW)
    o.record_every=ref.record_every=2
    for t in range(1,19):
        o.set_timestep_scale(.7 if t%4==0 else 1.)
        ref.set_timestep_scale(.7 if t%4==0 else 1.)
        run(o,p,t); run(ref,q,t); same(p,o,q,ref)

@pytest.mark.parametrize("src,dst",[(Polora,Polora),(Reference,Polora),(Polora,Reference)])
def test_fresh_serialized_checkpoint(src,dst):
    kw=dict(chunk_memory_mib=budget(2),**KW)
    p,q=pairs(),pairs()
    o,ref=src(pairs=p,**kw),Reference(pairs=q,**kw)
    for t in range(1,7): run(o,p,t); run(ref,q,t)
    buf=io.BytesIO()
    torch.save({"state":o.state_dict(),"params":[tuple(x.detach() for x in pr) for pr in p]},buf)
    buf.seek(0)
    ck=torch.load(buf,weights_only=True)
    fresh=[tuple(torch.nn.Parameter(x.clone()) for x in pr) for pr in ck["params"]]
    resumed=dst(pairs=fresh,**kw)
    resumed.load_state_dict(ck["state"])
    for g in resumed._groups:
        for j,i in enumerate(g.idxs):
            for key in ("M_A","M_B","Q","P"):
                assert resumed.pair_state[i][key].data_ptr()==getattr(g,key)[j].data_ptr()
            for key,arr,off in (("v_sigma_A",g.vA,0),("v_sigma_DA",g.vA,len(g.idxs)),
                                ("v_sigma_B",g.vB,0),("v_sigma_DB",g.vB,len(g.idxs))):
                assert resumed.pair_state[i][key].data_ptr()==arr[j+off].data_ptr()
    for t in range(7,13):
        run(resumed,fresh,t); run(ref,q,t); same(fresh,resumed,q,ref)

def test_lru_cross_group_and_eviction():
    p=pairs([(2,8,12,torch.float32)]*40+[(2,12,8,torch.float32)]*41)
    o=Polora(pairs=p,compile=False)
    g,h=o._groups
    a=o._chunk_index(0,g,[0,2])
    b=o._chunk_index(1,h,[0,2])
    assert a[1].tolist()==[0,2,40,42]
    assert b[1].tolist()==[0,2,41,43]
    expected={}
    from collections import OrderedDict
    expected=OrderedDict(o._chunk_indices)
    rng=random.Random(11)
    for _ in range(1100):
        part=tuple(sorted(rng.sample(range(40),5)))
        idx,vidx=o._chunk_index(0,g,list(part))
        assert idx.tolist()==list(part)
        assert vidx.tolist()==list(part)+[x+40 for x in part]
        expected[(0,part)]=(idx,vidx)
        expected.move_to_end((0,part))
        if len(expected)>512: expected.popitem(last=False)
        assert list(expected)==list(o._chunk_indices)
    assert len(o._chunk_indices)==512
    key=next(iter(o._chunk_indices))
    old=o._chunk_indices[key]
    assert o._chunk_index(key[0],o._groups[key[0]],list(key[1])) is old
    assert next(reversed(o._chunk_indices))==key

def test_layout_change_through_step():
    p,q=pairs(),pairs()
    kw=dict(chunk_memory_mib=budget(2),**KW)
    o,ref=Polora(pairs=p,**kw),Reference(pairs=q,**kw)
    for t in range(1,5): run(o,p,t); run(ref,q,t)
    assert o._chunk_indices
    old=tuple(o._chunk_indices.values())
    for pp in (p,q):
        for x in pp[21]: x.data=x.data.float()
    for t in range(5,11): run(o,p,t); run(ref,q,t); same(p,o,q,ref)
    assert not any(v is prev for v in o._chunk_indices.values() for prev in old)

def test_selective_real_autograd():
    torch.manual_seed(0)
    d=24
    p,q=pairs([(4,d,d,torch.float32),(4,d,2*d,torch.float32),(4,2*d,d,torch.float32)]*6),pairs([(4,d,d,torch.float32),(4,d,2*d,torch.float32),(4,2*d,d,torch.float32)]*6)
    kw=dict(chunk_memory_mib=budget(2,4,d,2*d),**KW)
    o,ref=Polora(pairs=p,**kw),Reference(pairs=q,**kw)
    x=torch.randn(16,d)
    windows=[set(range(6)),{0,2,3,5},{1,4},set(range(6)),{5}]
    for t in range(40):
        for pp,opt in ((p,o),(q,ref)):
            active=windows[t//8]
            for j,pr in enumerate(pp):
                for v in pr: v.requires_grad_(j//3 in active)
            opt.zero_grad(set_to_none=True)
            h=x
            for j in range(0,len(pp),3):
                (a,b),(c,e),(f,g)=pp[j:j+3]
                h=h+torch.tanh(h@a.T@b.T)
                u=h@c.T@e.T+.1*torch.cat((h,h),dim=1)
                h=h+.5*torch.tanh(u@f.T@g.T)
            h.square().mean().backward()
            opt.step()
        same(p,o,q,ref)
    assert any(part[-1]-part[0]!=len(part)-1 for _,part in o._chunk_indices)
