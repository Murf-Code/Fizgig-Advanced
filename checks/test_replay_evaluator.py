import hashlib
import numpy as np
import pytest
from fizgig.training.replay_evaluator import *

def case(d='target',u='a',v='p',t=.125,n=1):
    return Case(d,u,v,t,n,hashlib.sha256(str((d,u,v,t,n)).encode()).hexdigest())

def panel(): return [case(d,d+str(u),v,t,n) for d in DOMAINS for u in range(2) for v in ('a','b') for t in (.125,.875) for n in (1,2)]

def test_identity():
    r=np.array([1.,2.]); m=metrics(case(),r,r,r,r)
    assert all(m[k]==0 for k in m if k!='base_energy')

def test_target_gain():
    m=metrics(case(),[.5],[0.],[0.],[1.]); assert m['target_gain']==.75

def test_gain_not_absolute_target_error():
    m=metrics(case(),[2.],[1.],[0.],[1.]); assert m['target_gain']==-1

def test_base_improvement_nonzero_parent_movement():
    m=metrics(case('near'),[1.],[2.],[0.]); assert m['parent_mse']==1 and m['base_increment']==-3

def test_base_same_radius_different_direction():
    m=metrics(case('far'),[0.,1.],[1.,0.],[0.,0.]); assert m['base_increment']==0 and m['parent_mse']==1

def test_scaling():
    m=metrics(case('near'),[2.,4.],[1.,2.],[1.,2.]); assert m['base_mse']==2.5

def test_shape_reject():
    with pytest.raises(ValueError): metrics(case(),[[1.],[2.]],[1.,2.],[1.,2.])
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_nonfinite(bad):
    with pytest.raises(ValueError): metrics(case(),[bad],[1.],[1.],[1.])

def test_missing_target():
    with pytest.raises(ValueError): metrics(case(),[1.],[1.],[1.])

def test_unexpected_target():
    with pytest.raises(ValueError): metrics(case('near'),[1.],[1.],[1.],[1.])

def test_tiny_change():
    p=np.array([1+1e-10]); r=np.ones(1); y=np.zeros(1)
    m=metrics(case(),p,r,r,y); assert m['target_gain']<0 and np.isclose(m['target_gain'],-2e-10,rtol=1e-5,atol=0)

def test_order_manifest():
    c=panel(); assert freeze(c)==freeze(c[::-1])

def test_duplicate_manifest():
    c=panel();
    with pytest.raises(ValueError): freeze(c+[c[0]])

def test_missing_domain():
    with pytest.raises(ValueError): freeze([c for c in panel() if c.domain!='far'])

def test_unbalanced():
    with pytest.raises(ValueError): freeze(panel()[1:])

def test_unit_leak():
    c=[case(d,'shared') for d in DOMAINS]
    with pytest.raises(ValueError): freeze(c)

def test_manifest_input_hash():
    c=panel(); d=c.copy(); d[0]=Case(**{**asdict(c[0]),'input_hash':'0'*64}); assert freeze(c)[1]!=freeze(d)[1]

def test_missing_candidate():
    with pytest.raises(ValueError): Evaluator(panel()).finish()

def test_candidate_duplicate():
    e=Evaluator(panel()); c=e.cases[0]; args=dict(candidate=[1.],parent=[1.],base=[1.],target=[1.] if c.domain=='target' else None); e.add(c,**args)
    with pytest.raises(ValueError): e.add(c,**args)

def test_aggregate_unit_weight():
    # One unit has two variants, other one; each unit still has 1/2 weight.
    c=[case(d,d+'1',v) for d in DOMAINS for v in ('a','b')]+[case(d,d+'2') for d in DOMAINS]
    e=Evaluator(c)
    for k in c: e.add(k,candidate=[2.] if k.unit.endswith('2') else [0.],parent=[0.],base=[0.],target=[1.] if k.domain=='target' else None)
    assert e.finish()['domains']['near']['mean']['base_mse']==2

def test_equal_weight_unequal_function():
    # Two parameter vectors equal norm; input only sees first coordinate.
    a=np.array([1.,0.]); b=np.array([0.,1.]); x=np.array([1.,0.]); assert np.linalg.norm(a)==np.linalg.norm(b)
    m=metrics(case('near'),[a@x],[0.],[0.]); n=metrics(case('near'),[b@x],[0.],[0.]); assert m['base_mse']==1 and n['base_mse']==0

def test_cancellation():
    a=np.array([1.,1.]); assert np.linalg.norm(a)**2>0
    assert metrics(case('far'),[a@np.array([1.,-1.])],[0.],[0.])['base_mse']==0

def test_local_vs_wide():
    a=metrics(case('near'),[2.,0.,0.,0.],[0.]*4,[0.]*4)
    b=metrics(case('near'),[1.]*4,[0.]*4,[0.]*4); assert a['base_mse']==b['base_mse']

def test_pareto_tradeoff():
    assert not pareto({'g':2,'p':0},{'g':1,'p':1}) and not pareto({'g':1,'p':1},{'g':2,'p':0})

def test_pareto_dominance():
    assert pareto({'g':2,'p':1},{'g':1,'p':1})

def test_paired_small_n():
    r=paired_summary([1,1,1]); assert r['sign_p_two_sided']==.25 and r['exploratory_ci95'] is None

def test_paired_constant():
    r=paired_summary([.2]*12); assert np.allclose(r['exploratory_ci95'],[.2,.2])

def test_paired_no_pseudorep():
    with pytest.raises(ValueError): paired_summary([[1,2],[3,4]])

def test_gate():
    m=dict(target_gain=.01,near_preserve=.001,far_preserve=.001)
    assert gate(dict(target_gain=.02,near_preserve=-.0005,far_preserve=0),m)
    assert not gate(dict(target_gain=.02,near_preserve=-.002,far_preserve=0),m)
    assert not gate(dict(target_gain=.005,near_preserve=1,far_preserve=1),m)

def result(p=0.,parent=0.):
    e=Evaluator(panel())
    for c in e.cases:
        e.add(c,candidate=[p],parent=[parent],base=[0.],target=[1.] if c.domain=='target' else None)
    return e.finish()

def test_same_reference_comparison():
    r=compare_candidates(result(.5),result(.25))
    assert r=={'target_gain':.3125,'near_preserve':-.1875,'far_preserve':-.1875}

def test_different_reference_reject():
    with pytest.raises(ValueError): compare_candidates(result(.5),result(.25,parent=.125))
