from contextlib import nullcontext
import torch, pytest
from fizgig.training.replay_dual import allocations,evaluate_dual
from fizgig.training.replay_disk import disk_capture
from fizgig.training.replay_diagnostics import digest
from fizgig.training.replay_evaluator import Case
from test_replay_live import setup


def test_allocation_and_ties():
    g={b:1. for b in range(28)};s={b:1. for b in g};g[26]=.1;s[13]=10000.
    a,p,r=allocations(g,s)
    assert p['adaptive_low_rms']==26 and p['adaptive_movement']==13
    assert a['adaptive_low_rms']==a['static26'] and a['adaptive_movement']==a['static13']
    with pytest.raises(ValueError):allocations(g,{**s,13:float('nan')})


@pytest.mark.parametrize('bad',[False,True])
def test_dual(tmp_path,bad):
    pairs,opt,g,net,dit,extra,driver,val=setup()
    x=torch.ones(1,1,2,2);panel=[(Case('target','a','a',.125,1,digest(x)),x,torch.zeros_like(x))]
    scores={b:1. if b!=26 else .01 for b in range(28)}
    def state():return digest(([(p.A,p.B) for p in pairs],opt.state_dict(),g.get_state()))
    old=state();calls=0
    def predict(x):
        nonlocal calls
        calls+=1
        return driver.predict(dit,{'xt':x},{'text':torch.zeros(1)})+(1 if bad and calls>=4 else 0)
    with disk_capture(pairs,opt,generators=(g,),scratch=tmp_path) as fork:
        report={}
        def run():return evaluate_dual(fork,net,scores,panel,predict,lambda fn:fn(),report,lambda:None,lambda *a:nullcontext())
        if bad:
            with pytest.raises(RuntimeError,match='restoration failed'):run()
        else:
            r=run();assert r['integrity_passed'] and r['matching']['matched']
            assert r['aliases']['adaptive_low_rms']=='static26'
            assert set(r['evaluators'])=={'native','weight_only'}
            assert r['evaluators']['native']['arms']==r['evaluators']['weight_only']['arms']
            assert all(len(v['restored'])==len(r['endpoint_geometry']) for v in r['evaluators'].values())
    assert old==state()
