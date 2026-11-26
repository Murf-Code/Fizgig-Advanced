import copy, io
import pytest
import torch
from fizgig.training.muon_me import MuonME
from fizgig.training.muon_me_compressed import CompressedMuonME, encode, decode, validate, state_bytes

@pytest.mark.parametrize('mode',['bf16','int8','splitmax'])
@pytest.mark.parametrize('nesterov',[True,False])
def test_first_step_and_resume(mode,nesterov):
    rng=torch.Generator().manual_seed(13)
    ps=[torch.nn.Parameter(torch.randn(s,generator=rng)) for s in [(7,9),(7,9),(9,7),(3,5)]]
    qs=[torch.nn.Parameter(p.detach().clone()) for p in ps]
    ref=MuonME(ps,weight_decay=.01,nesterov=nesterov)
    opt=CompressedMuonME(qs,momentum_storage=mode,weight_decay=.01,nesterov=nesterov)
    for step in range(5):
        for p,q in zip(ps,qs):
            p.grad=torch.randn(p.shape,generator=rng);q.grad=p.grad.clone()
        ref.step();opt.step()
        if step==0:
            for p,q in zip(ps,qs):torch.testing.assert_close(p,q,rtol=0,atol=0)
    checkpoint=io.BytesIO();torch.save(opt.state_dict(),checkpoint);checkpoint.seek(0)
    saved=torch.load(checkpoint,weights_only=True,map_location='cpu')
    rs=[torch.nn.Parameter(q.detach().clone()) for q in qs]
    resumed=CompressedMuonME(rs,momentum_storage=mode,weight_decay=.01,nesterov=nesterov)
    resumed.load_state_dict(saved)
    for q,r in zip(qs,rs):
        a,b=opt.state[q],resumed.state[r]
        for k,v in a.items():
            if torch.is_tensor(v):assert torch.equal(v,b[k]) and v.dtype==b[k].dtype and v.data_ptr()!=b[k].data_ptr()
        assert sum(v.numel()*v.element_size() for v in a.values() if torch.is_tensor(v))==state_bytes(q.shape,mode,64)
        q.grad=torch.randn(q.shape,generator=rng);r.grad=q.grad.clone()
    opt.step();resumed.step()
    for q,r in zip(qs,rs):torch.testing.assert_close(q,r,atol=0,rtol=0)
    bad=copy.deepcopy(saved);st=next(iter(bad['state'].values()))
    st['shape']=(999,999)
    previous=resumed.state
    with pytest.raises(ValueError):resumed.load_state_dict(bad)
    assert resumed.state is previous

def test_splitmax_and_padding():
    x=torch.full((1,65),.1);x[0,0]=100
    st=encode(x,'splitmax',64);validate(st,x.shape,'splitmax',64)
    torch.testing.assert_close(decode(st,x.shape,'splitmax',64),x)
    st['index'][-1]=63
    with pytest.raises(ValueError):validate(st,x.shape,'splitmax',64)

def test_strict_budget_before_mutation():
    p=torch.nn.Parameter(torch.ones(3,3));q=torch.nn.Parameter(torch.ones(64,64))
    opt=CompressedMuonME([p,q],momentum_storage='int8',chunk_memory_mib=.01)
    for x in (p,q):x.grad=torch.ones_like(x)
    with pytest.raises(MemoryError):opt.step()
    assert not opt.state and torch.equal(p,torch.ones_like(p))

def test_format_mismatch_and_dtype():
    p=torch.nn.Parameter(torch.ones(4,4));p.grad=torch.ones_like(p)
    a=CompressedMuonME([p],momentum_storage='int8');a.step()
    b=CompressedMuonME([p],momentum_storage='bf16')
    with pytest.raises(ValueError):b.load_state_dict(a.state_dict())
    with pytest.raises(ValueError):CompressedMuonME([torch.nn.Parameter(p.bfloat16())])
