import numpy as np
import pytest
from fizgig.training.replay_diagnostics import localized,depth_shares,digest,guarded_forward
from fizgig.training.replay_evaluator import metrics,Case


def test_equal_mse_distinct_concentration():
    a=np.r_[10.,np.zeros(99)];b=np.ones(100)
    assert (a*a).mean()==(b*b).mean()==1
    assert localized(a)['top_energy_share']==1
    assert localized(b)['top_energy_share']==.01
    assert localized(a)['max_abs']==10

def test_zero_energy():
    r=localized(np.zeros(7));assert r['zero_energy'] and r['top_energy_share']==0 and r['top_k']==1

@pytest.mark.parametrize('bad',[0,-1,1.1,float('nan')])
def test_bad_fraction(bad):
    with pytest.raises(ValueError):localized([1],fraction=bad)

def test_depth_energy():
    r=depth_shares({i:1. for i in range(28)})
    assert r['depth_squared']=={'early':9.,'middle':10.,'late':9.}
    assert sum(r['depth_fraction'].values())==pytest.approx(1)

def test_incomplete_depth():
    with pytest.raises(ValueError):depth_shares({0:1})

def test_empty_depth_budget():
    assert all(v is None for v in depth_shares(dict.fromkeys(range(28),0))['depth_fraction'].values())

def test_dtype_shape_and_order():
    a=np.ones(4,np.float32);assert digest(a)!=digest(a.astype(np.float64));assert digest(a)!=digest(a.reshape(2,2))
    assert digest({'a':a,'b':2})==digest({'b':2,'a':a})

def test_torch_bf16_mutation():
    import torch
    a=torch.ones(4,dtype=torch.bfloat16);h=digest(a);a[0]=1.01;assert digest(a)!=h

def test_forward_repeat():
    x={'noise':np.ones(4)};s={'momentum':np.ones(4)}
    def run(z):return guarded_forward(lambda p:p['noise']*z,x,expected_input_hash=digest(x),read_state=lambda:s)
    a,b=run(2),run(3);assert np.array_equal(a,run(2)) and np.array_equal(b,run(3))

@pytest.mark.parametrize('key',['momentum','buffer','rng','scheduler','ema','selector','gradient','factor'])
def test_state_fault(key):
    x={'noise':np.ones(4)};s={key:np.ones(4)}
    def fail(p):s[key][0]+=1;return p['noise']
    with pytest.raises(RuntimeError,match='state mutated'):guarded_forward(fail,x,expected_input_hash=digest(x),read_state=lambda:s)

def test_input_fault():
    x={'noise':np.ones(4)}
    def fail(p):p['noise'][0]+=1;return p['noise']
    with pytest.raises(RuntimeError,match='input mutated'):guarded_forward(fail,x,expected_input_hash=digest(x),read_state=lambda:{})

def test_wrong_hash():
    with pytest.raises(ValueError,match='manifest'):guarded_forward(lambda x:x,{'noise':np.ones(4)},expected_input_hash='0'*64,read_state=lambda:{})

def test_bf16_hidden_update_null_is_zero():
    import torch
    p=torch.ones(32768); c=p-1e-4; y=np.zeros(p.shape)
    r=p.bfloat16().float().numpy(); q=c.bfloat16().float().numpy()
    case=Case('target','toy','p',.375,1,'0'*64)
    assert metrics(case,r,r,r,y)['target_gain']==0
    assert metrics(case,q,r,r,y)['target_gain']==0
    assert metrics(case,c.numpy(),p.numpy(),p.numpy(),y)['target_gain']>0
