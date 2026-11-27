from pathlib import Path
import copy
import sys
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research/krea2_littlebit"))
import qat_core as q


def test_core():
    torch.manual_seed(42)
    a=torch.randn(7,31); b=torch.randn(37,7)
    aa,bb=q.rotate(a,b,4)
    assert torch.allclose(b@a,bb@aa,atol=2e-5)
    p=q.Path(aa,bb); c=p.components(); w=q.weight(c).detach()
    x=torch.randn(5,31); g=torch.randn(5,37)
    y,dx=q.actions(c,x,g)
    assert torch.allclose(y,x@w.T,atol=2e-5)
    assert torch.allclose(dx,g@w,atol=2e-5)
    for mode in ("ste","smooth"):
        z=torch.tensor([-.01,0.,.01],requires_grad=True)
        q.binary(z,mode).sum().backward()
        assert torch.equal(q.binary(z,mode),torch.tensor([-1.,1.,1.]))
        assert (z.grad>0).all()
        ps=nn.ModuleList([copy.deepcopy(p)])
        h=q.train(ps,b@a,[(x,g,x@(b@a).T,g@(b@a))],mode,steps=20,action=True,batch=16)
        assert h[-1]["sign_change_fraction"][0]>0
    for cap in (1,100,500,1000):
        for n in (1,2):
            rr=q.ranks(37,31,cap,20,n)
            assert not rr or sum(q.size(37,31,r) for r in rr)<=cap
