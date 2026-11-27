"""Compatibility tests for real native optimizer steps, not a quality benchmark."""
import copy
import math
import pytest
import torch

from fizgig.training.selective_compensation import compensated_step, validate_optimizer


def _state_equal(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if torch.is_tensor(a[k]):
            assert torch.equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


def _make(name, params, device, mode):
    groups = [{"params": [params[0], params[1]], "lr": .002, "weight_decay": .13},
              {"params": [params[2]], "lr": .001, "weight_decay": .07}]
    if name == "adamw":
        return torch.optim.AdamW(groups, foreach=mode=="foreach", fused=mode=="fused")
    bnb = pytest.importorskip("bitsandbytes")
    cls = bnb.optim.AdamW8bit if name=="adamw8bit" else bnb.optim.PagedAdamW8bit
    return cls(groups, min_8bit_size=4096)


@pytest.mark.parametrize("name,device,mode", [
    ("adamw","cpu","scalar"), ("adamw","cpu","foreach"),
    ("adamw","cuda","fused"), ("adamw","cuda","foreach"),
    ("adamw8bit","cuda","8bit"), ("pagedadamw8bit","cuda","paged"),
])
def test_native_step_and_resume(name,device,mode):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    gen = torch.Generator().manual_seed(31)
    # Exercise BNB 32-bit fallback, quantized state and (large factor) paging.
    shapes = [(32,256),(512,256),(4,8)]
    init = [torch.randn(s,generator=gen).to(device) for s in shapes]
    grads = [[torch.randn(s,generator=gen).to(device) for s in shapes] for _ in range(5)]
    all_params = [[torch.nn.Parameter(t.clone()) for t in init] for _ in range(3)]
    opts = [_make(name,p,device,mode) for p in all_params]
    for opt in opts:
        validate_optimizer(opt,name)
    for ps,opt in zip(all_params,opts):
        for p,g in zip(ps,grads[0]): p.grad = g.clone()
        compensated_step(opt,name,1.)
    for ps in all_params[1:]:
        assert all(torch.equal(p,q) for p,q in zip(ps,all_params[0]))
    if name != "adamw":
        assert opts[0].state[all_params[0][0]]["state1"].dtype == torch.uint8
        assert opts[0].state[all_params[0][2]]["state1"].dtype == torch.float32
    frozen = [ps[1].detach().clone() for ps in all_params]
    frozen_states = [copy.deepcopy(opt.state[ps[1]]) for ps,opt in zip(all_params,opts)]
    original = [ [(g["lr"],g["weight_decay"]) for g in opt.param_groups] for opt in opts ]
    schedulers = [torch.optim.lr_scheduler.StepLR(opt,step_size=1,gamma=.9) for opt in opts]
    c = math.sqrt(2)
    before = [p.detach().clone() for p in all_params[0]]
    for ps in all_params:
        for i,p in enumerate(ps): p.grad = None if i==1 else grads[1][i].clone()
    # Three identical states: native step, compensated step, and independent
    # native oracle using explicitly specified LR/decay values.
    opts[0].step()
    compensated_step(opts[1],name,c)
    for g in opts[2].param_groups:
        g["lr"] *= c
        g["weight_decay"] /= c
    opts[2].step()
    for g,(lr,wd) in zip(opts[2].param_groups,original[2]):
        g["lr"],g["weight_decay"] = lr,wd
    for i in (0,2):
        assert torch.equal(all_params[1][i],all_params[2][i])
        _state_equal(opts[1].state[all_params[1][i]],opts[0].state[all_params[0][i]])
        # BNB backend may apply the same decay after the update rather than before.
        # In either ordering, the affine difference from the decay-only endpoint scales.
        lr,wd = original[0][0 if i==0 else 1]
        decayed = before[i] * (1-lr*wd)
        u = all_params[0][i].detach() - decayed
        v = all_params[1][i].detach() - decayed
        assert torch.allclose(v,c*u,atol=2e-6,rtol=.003)
    for j,(ps,opt,sch) in enumerate(zip(all_params,opts,schedulers)):
        assert torch.equal(ps[1],frozen[j])
        _state_equal(opt.state[ps[1]],frozen_states[j])
        assert [(g["lr"],g["weight_decay"]) for g in opt.param_groups]==original[j]
        sch.step()
        assert all(math.isclose(g["lr"],lr*.9) for g,(lr,wd) in zip(opt.param_groups,original[j]))
    # Serialize/reload, first resume into a reduced-mask step, then reactivate.
    # Use serialization, not shared live tensor aliases.
    import io
    buf=io.BytesIO()
    torch.save(opts[1].state_dict(),buf)
    buf.seek(0)
    sd=torch.load(buf,weights_only=False)
    new_params=[torch.nn.Parameter(p.detach().clone()) for p in all_params[1]]
    new_opt=_make(name,new_params,device,mode)
    new_opt.load_state_dict(sd)
    validate_optimizer(new_opt,name)
    for k,scale in ((2,c),(3,1.)):
        for ps,opt in ((all_params[1],opts[1]),(new_params,new_opt)):
            for i,p in enumerate(ps):
                p.grad = None if k==2 and i==1 else grads[k][i].clone()
            compensated_step(opt,name,scale)
        for p,q in zip(all_params[1],new_params):
            assert torch.equal(p,q)
            _state_equal(opts[1].state[p],new_opt.state[q])


def test_restores_groups_on_failure(monkeypatch):
    p=torch.nn.Parameter(torch.ones(4))
    opt=torch.optim.AdamW([p],lr=.01,weight_decay=.02)
    def fail():
        assert opt.param_groups[0]["lr"]==.015
        assert math.isclose(opt.param_groups[0]["weight_decay"],.02/1.5)
        raise RuntimeError("step failed")
    monkeypatch.setattr(opt,"step",fail)
    with pytest.raises(RuntimeError,match="step failed"):
        compensated_step(opt,"adamw",1.5)
    assert opt.param_groups[0]["lr"]==.01
    assert opt.param_groups[0]["weight_decay"]==.02


def test_reject_wrong_optimizer():
    p=torch.nn.Parameter(torch.ones(4))
    with pytest.raises(RuntimeError):
        validate_optimizer(torch.optim.SGD([p],lr=.1),"adamw")
    with pytest.raises(RuntimeError):
        compensated_step(torch.optim.SGD([p],lr=.1),"sgd",1.)


def test_bnb_override_rejected_without_gpu():
    from types import SimpleNamespace
    p=torch.nn.Parameter(torch.ones(4))
    p.grad=torch.ones_like(p)
    group={"params":[p],"lr":.01,"weight_decay":.02}
    class Fake:
        initialized=True
        param_groups=[group]
        def get_config(self,gi,pi,g):
            return {"lr":.01,"weight_decay":g["weight_decay"]}
        def step(self):
            raise AssertionError("should not step with a fixed absolute override")
    with pytest.raises(RuntimeError,match="overrides"):
        compensated_step(Fake(),"adamw8bit",1.5)
    assert group["lr"]==.01 and group["weight_decay"]==.02
