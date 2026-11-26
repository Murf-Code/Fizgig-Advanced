import torch, pytest
from contextlib import nullcontext
from types import SimpleNamespace
from fizgig.training.replay_isolation import isolate,movement,precision
from fizgig.training.replay_evaluator import Case
from fizgig.training.replay_diagnostics import digest
from fizgig.training.replay_forks import Pair
from fizgig.training.muon import BatchedMuon
from fizgig.families.lora import TRAINABLE


def setup():
    g=torch.Generator().manual_seed(71);pairs=[];wrapped={}
    for i in range(28):
        a=torch.nn.Parameter(torch.randn(2,5,generator=g));b=torch.nn.Parameter(torch.randn(4,2,generator=g))
        a.grad=torch.randn(a.shape,generator=g);b.grad=torch.randn(b.shape,generator=g)
        pairs.append(Pair(str(i),i,a,b,1.));wrapped[str(i)]=SimpleNamespace(adapters={TRAINABLE:None},scales={TRAINABLE:1.})
    opt=BatchedMuon([p for pair in pairs for p in (pair.A,pair.B)],lr=.00045,weight_decay=.01,adjust_lr_fn='match_rms_adamw');opt.step()
    net=SimpleNamespace(wrapped=wrapped)
    panel=[]
    for unit in ('a','b'):
        for t in (.125,.875):
            x=torch.randn(5,generator=g);panel.append((Case('target',unit,'a',t,1,digest(x)),x,torch.zeros(4)))
    def predict(x):return sum((p.B@p.A)@x*net.wrapped[p.name].scales[TRAINABLE] for p in pairs)
    return pairs,opt,net,panel,predict


def run(s,predict=None):
    pairs,opt,net,panel,fn=s;report={}
    return isolate(pairs=pairs,optimizer=opt,net=net,dit=None,panel=panel,predict=predict or fn,guard=lambda f:f(),report=report,save=lambda:None,precision_context=lambda *a:nullcontext(),trace=False)


def state(s):
    pairs,opt,net,_,_=s
    return digest(([(p.A,p.B,p.A.grad,p.B.grad) for p in pairs],opt.state_dict(),[w.scales for w in net.wrapped.values()]))


def test_decay_isolation():
    s=setup();before=state(s);r=run(s)
    assert r['integrity_passed'] and state(s)==before
    assert len(r['factor_rounding'])==28 and len(r['native_decay'])==4
    assert all(x['mse']>0 for x in r['native_decay'])
    assert len(r['precision'])==3 and all(len(x)==2 for x in r['precision'].values())
    assert all(x['equal'] for rows in r['integrity'].values() for x in rows)


def test_hidden_base_state_detected():
    s=setup();before=state(s);flag=False
    def predict(x):
        nonlocal flag
        if all(w.scales[TRAINABLE]==0 for w in s[2].wrapped.values()):flag=True
        return s[-1](x)+(0.02 if flag else 0)
    with pytest.raises(RuntimeError,match='after_base'):run(s,predict)
    assert state(s)==before


def test_hidden_decay_state_detected():
    s=setup();before=state(s);calls=0
    def predict(x):
        nonlocal calls
        calls+=1
        return s[-1](x)+(0.02 if calls>=13 else 0)
    with pytest.raises(RuntimeError,match='after_decay'):run(s,predict)
    assert state(s)==before


def test_compact_geometry():
    a,b=torch.randn(3,9),torch.randn(7,3);c,d=a*.9999955,b*.9999955
    for dtype in (torch.float32,torch.bfloat16):
        aa,bb,cc,dd=[x.to(dtype) for x in (a,b,c,d)]
        dense=float((dd.double()@cc.double()-bb.double()@aa.double()).square().sum())
        assert movement(aa,bb,cc,dd,1.)==pytest.approx(dense,rel=1e-9,abs=1e-24)


def test_fp32_lora_autocast_and_restore():
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__();self.base=torch.nn.Linear(5,4,bias=False);self.adapters=torch.nn.ModuleDict({TRAINABLE:torch.nn.Sequential(torch.nn.Linear(5,2,bias=False),torch.nn.Linear(2,4,bias=False))});self.scales={TRAINABLE:1.}
        def forward(self,x):return self.base(x)+self.adapters[TRAINABLE](x)
    w=Wrapper();net=SimpleNamespace(wrapped={'w':w});x=torch.randn(3,5)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        baseline=w(x);base=w.base(x)
        with precision(net,False,True):
            with torch.autocast('cpu',enabled=False):expected=base+w.adapters[TRAINABLE](x).to(base.dtype)
            assert torch.equal(w(x),expected)
        assert torch.equal(w(x),baseline) and 'forward' not in w.__dict__
    with pytest.raises(RuntimeError):
        with precision(net,False,True):raise RuntimeError('fault')
    assert 'forward' not in w.__dict__
