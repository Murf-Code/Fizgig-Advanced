import pytest, torch
from fizgig.training.memory_pair import PairTrace, fingerprint, product_norm, displacement, compare

def test_compact():
    torch.manual_seed(2)
    before=(torch.randn(3,7),torch.randn(5,3))
    after=tuple(x+torch.randn_like(x)*.01 for x in before)
    dense=after[1].double()@after[0].double()-before[1].double()@before[0].double()
    assert product_norm(displacement(before,after))==pytest.approx(float(dense.norm()),rel=1e-10)
    gs=tuple(torch.ones_like(x) for x in before)
    other=tuple(x+torch.randn_like(x)*.02 for x in before)
    m=compare(before,after,gs,(before,other,gs))
    target=after[1].double()@after[0].double()-other[1].double()@other[0].double()
    assert m['update_error']==pytest.approx(float(target.norm()),rel=1e-10)
    assert m['grad_relative']==0

def test_trace(tmp_path):
    def run(case, mismatch=False):
        a=torch.nn.Parameter(torch.ones(2,3)); b=torch.nn.Parameter(torch.zeros(4,2))
        out=tmp_path/case; out.mkdir()
        tr=PairTrace({'x':([a,b],1.)},out,case,2)
        for step in (1,2):
            tr.begin(step,['a'],torch.ones(2),{'x':torch.ones(1)},torch.ones(2)*(2 if mismatch else 1),torch.ones(1))
            a.grad=torch.ones_like(a); b.grad=torch.ones_like(b)
            tr.pre_update(torch.tensor(.1))
            with torch.no_grad(): a.add_(.01); b.add_(.02)
            rec=tr.end()
            assert rec['metrics']['x']['update_relative']<1e-12
        return tr.finish()
    assert run('pair-fp32-control')==run('pair-fp32-repeat')
    with pytest.raises(RuntimeError,match='actual input mismatch'): run('pair-bad',True)

def test_hash():
    assert fingerprint(torch.ones(2,dtype=torch.bfloat16))!=fingerprint(torch.ones(2))
    assert fingerprint({'x':[1,2]})!=fingerprint({'x':[2,1]})
