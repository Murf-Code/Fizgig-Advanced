import torch, pytest
from types import SimpleNamespace
from fizgig.training.replay_live import run_pilot
from fizgig.training.replay_forks import Pair
from fizgig.training.muon import BatchedMuon
from fizgig.training.replay_diagnostics import digest
from fizgig.families.lora import TRAINABLE

class TinyDriver:
    def __init__(self,net,pairs,extra):self.net=net;self.pairs=pairs;self.extra=extra
    def batch_cond(self,batch,device):return {'text':batch['cond__text'].to(device)}
    def block_swap_mode(self,dit,inference):self.extra['inference']=inference
    def predict(self,dit,state,cond):
        out=state['xt'].clone()
        for p in self.pairs:out=out+(p.B@p.A).sum()*self.net.wrapped[p.name].scales[TRAINABLE]
        return out+cond['text']

def setup():
    g=torch.Generator().manual_seed(1);pairs=[];wrapped={}
    for b in range(28):
        for i in range(8):
            n=f'blocks.{b}.{i}';a=torch.nn.Parameter(torch.randn(2,3,generator=g)*.01);z=torch.nn.Parameter(torch.randn(3,2,generator=g)*.01)
            a.grad=torch.randn(a.shape,generator=g);z.grad=torch.randn(z.shape,generator=g)
            pairs.append(Pair(n,b,a,z,1.));wrapped[n]=SimpleNamespace(adapters={TRAINABLE:None},scales={TRAINABLE:1.})
    # Match the real FamilyLoRA layout: all targets wrapped, only main blocks trained.
    for name in ('first','tmlp.0','tmlp.2','txtmlp.1','txtmlp.3','tproj.1','last.linear'):
        wrapped[name]=SimpleNamespace(adapters=torch.nn.ModuleDict(),scales={})
    opt=BatchedMuon([p for x in pairs for p in (x.A,x.B)],lr=.00045,weight_decay=.01,adjust_lr_fn='match_rms_adamw')
    opt.step();net=SimpleNamespace(wrapped=wrapped);dit=torch.nn.Module();dit.config=SimpleNamespace(patch=2);dit.register_buffer('test',torch.zeros(1))
    extra={'inference':False};driver=TinyDriver(net,pairs,extra)
    val=SimpleNamespace(ready=True,wanted=['held'],batches={'held':{'latents':torch.randn(1,1,2,2,generator=g),'cond__text':torch.ones(1)}},ts=[.125,.875],seed=42)
    return pairs,opt,g,net,dit,extra,driver,val

def run(tmp_path,s):
    pairs,opt,g,net,dit,extra,driver,val=s
    return run_pilot(driver=driver,dit=dit,net=net,optimizer=opt,validator=val,device='cpu',swapped=True,generators=(g,),read_extra_state=lambda:extra,output=tmp_path/'report.json',step=8,pairs=pairs,patchify=lambda x,p:(x,None,None))

def test_pilot(tmp_path):
    s=setup();pairs,opt,g,net,dit,extra,driver,val=s
    before=digest(([(p.A,p.B) for p in pairs],opt.state_dict(),g.get_state()))
    r=run(tmp_path,s)
    assert r['status']=='completed' and r['matching']['matched'] and len(r['arms'])==3
    assert all(x['bitwise_equal'] for x in r['repeat_parent'])
    assert r['scope']=='development_target_only' and r['semantic_preservation'] is False
    assert digest(([(p.A,p.B) for p in pairs],opt.state_dict(),g.get_state()))==before
    assert dit.training and not extra['inference']
    assert all(w.scales==({TRAINABLE:1.} if TRAINABLE in w.adapters else {}) for w in net.wrapped.values())

def test_contamination_fails(tmp_path):
    s=setup();driver=s[-2];predict=driver.predict
    def bad(*a):
        s[4].test.add_(1);return predict(*a)
    driver.predict=bad
    with pytest.raises(RuntimeError,match='mutation'):run(tmp_path,s)
    assert s[4].training and not s[5]['inference']


@pytest.mark.parametrize('mixed',[False,True])
def test_frozen_adapter_rejected(tmp_path,mixed):
    s=setup();net=s[3];name=s[0][0].name if mixed else 'first'
    w=net.wrapped[name];w.adapters['context']=torch.nn.Identity();w.scales['context']=0.
    with pytest.raises(ValueError,match='no frozen/context adapters'):run(tmp_path,s)


def test_orphan_scale_rejected(tmp_path):
    s=setup();s[3].wrapped['first'].scales[TRAINABLE]=1.
    with pytest.raises(ValueError,match='adapter/scale mismatch'):run(tmp_path,s)
