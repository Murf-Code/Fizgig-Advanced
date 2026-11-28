from contextlib import nullcontext
from types import SimpleNamespace
import pytest
import torch
from fizgig.training.fixed_probe import run_fixed, gradient_metrics


def test_metrics():
    assert gradient_metrics([torch.ones(2)],[torch.ones(2)])['relative']==0
    assert gradient_metrics([torch.ones(2)],[torch.zeros(2)])['relative'] is None
    assert gradient_metrics([torch.ones(2)],[-torch.ones(2)])['cosine']==-1


def test_fixed(tmp_path):
    class Driver:
        def training_loss(self,dit,lat,cond,gen,**kw):
            noise=torch.randn(lat.shape,generator=gen); t=torch.rand(1,generator=gen)
            self._geometry_pair_input(lat,cond,noise,t)
            y=(lat+noise*t)@dit.a.T@dit.b.T
            self._geometry_fixed_output(y)
            return (y-1).square().mean(),{}
    def run(case,change=False):
        a=torch.nn.Parameter(torch.ones(2,3)); b=torch.nn.Parameter(torch.zeros(4,2))
        opt=torch.optim.AdamW([a,b]); opt.step=lambda:pytest.fail('unexpected optimizer step')
        d=Driver(); rows=[]
        out=tmp_path/case;out.mkdir()
        meta=run_fixed(d,SimpleNamespace(a=a,b=b),{'x':([a,b],1.)},opt,torch.ones(1,3)*(2 if change else 1),{},['x'],case=case,output_dir=out,seed=42,min_t=0,max_t=1,phase=lambda *args:nullcontext(),emit=lambda mark,r:rows.append((mark,r)))
        assert meta['trials']==6 and meta['optimizer_updates']==0
        assert not opt.state and not b.count_nonzero() and a.eq(1).all()
        assert len(rows)==7 and all(r['output_matches_reference'] for _,r in rows[:-1])
        assert all(r['metrics']['x']['relative']==0 for _,r in rows[:-1])
        return meta
    assert run('fixed-control')==run('fixed-repeat')
    with pytest.raises(RuntimeError,match='fixed input mismatch'):run('fixed-sync',True)
