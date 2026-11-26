import pytest
import torch
from fizgig.training.optimizers import create_optimizer, available_optimizers
from fizgig.training.muon_me import MuonME

def test_factory_and_shape_local():
    assert "muon_me" in available_optimizers(probe=False)
    rng = torch.Generator().manual_seed(42)
    a = [torch.nn.Parameter(torch.randn((8,16),generator=rng)) for _ in range(3)]
    b = [torch.nn.Parameter(p.detach().clone()) for p in a]
    args = "adjust_lr_fn=match_rms_adamw weight_decay=0 momentum=0.95 nesterov=True ns_steps=5"
    ref,_ = create_optimizer("muon", a, .0002,args)
    opt,label = create_optimizer("muon_me", b,.0002,args+" chunk_memory_mib=0")
    assert isinstance(opt, MuonME) and label.startswith("muon_me")
    assert opt.param_groups[0]["chunk_memory_mib"] == 0
    for _ in range(4):
        for x,y in zip(a,b):
            x.grad=torch.randn(x.shape,generator=rng); y.grad=x.grad.clone()
        ref.step(); opt.step()
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    assert all(torch.equal(ref.state[x]["momentum_buffer"],opt.state[y]["momentum_buffer"]) for x,y in zip(a,b))

def test_invalid_no_fallback():
    with pytest.raises(ValueError):
        create_optimizer("muon_me",[torch.nn.Parameter(torch.ones(4))],.0002)
    with pytest.raises(ValueError):
        create_optimizer("muon_me",[torch.nn.Parameter(torch.ones(4,4))],.0002,"chunk_memory_mib=-1")

def test_state_resume():
    p=torch.nn.Parameter(torch.ones(4,8))
    opt,_=create_optimizer("muon_me",[p],.0002,"chunk_memory_mib=128")
    p.grad=torch.ones_like(p); opt.step()
    q=torch.nn.Parameter(p.detach().clone())
    resumed,_=create_optimizer("muon_me",[q],.0002,"chunk_memory_mib=128")
    resumed.load_state_dict(opt.state_dict())
    q.grad=p.grad.clone()
    # load_state_dict may reuse tensors on the same device, clone the buffer to model disk resume.
    resumed.state[q]["momentum_buffer"]=resumed.state[q]["momentum_buffer"].clone()
    opt.step(); resumed.step()
    assert torch.equal(p,q)
