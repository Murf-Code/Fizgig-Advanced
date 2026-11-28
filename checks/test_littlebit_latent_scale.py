from pathlib import Path
import sys
import torch
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research/krea2_littlebit"))
import latent_scale_core as ls


def test_equations_and_solution():
    torch.manual_seed(42)
    w=torch.randn(13,23,dtype=torch.float64)
    x=torch.randn(11,23,dtype=w.dtype)
    g=torch.randn(11,13,dtype=w.dtype)
    a=torch.randn(7,23,dtype=w.dtype).sign()
    b=torch.randn(13,7,dtype=w.dtype).sign()
    a*=torch.rand(23,dtype=w.dtype)
    b*=torch.rand(13,1,dtype=w.dtype)
    c0=torch.randn(7,dtype=w.dtype)
    def features(t):
        return torch.cat([v.flatten()*weight**.5/teacher.norm() for v,weight,teacher in zip((t,x@t.T,g@t),ls.WEIGHTS,(w,x@w.T,g@w))])
    D=torch.stack([features(b[:,i,None]*a[i,None,:]) for i in range(7)],dim=1)
    y=features(w)
    H,rhs=ls.system(a,b,w,x,g)
    assert torch.allclose(H,D.T@D,rtol=1e-11,atol=1e-13)
    assert torch.allclose(rhs,D.T@y,rtol=1e-11,atol=1e-13)
    c,info=ls.solve(H,rhs,c0)
    assert torch.allclose(c,torch.linalg.lstsq(D,y).solution,rtol=1e-10,atol=1e-11)
    assert ls.metrics(a,b,c,w,x,g)['objective']==pytest.approx(float((D@c-y).square().sum()),rel=1e-12)
    assert info['relative_normal_residual']<1e-10
    assert ls.metrics(a,b,c,w,x,g)['objective']<=ls.metrics(a,b,c0,w,x,g)['objective']


def test_rank_deficient_and_zero_atom():
    A=torch.tensor([[1.,-1.],[1.,-1.],[0.,0.]],dtype=torch.float64)
    B=torch.tensor([[1.,1.,1.],[-1.,-1.,-1.]],dtype=A.dtype)
    c0=torch.tensor([1.,3.,5.],dtype=A.dtype)
    w=(B*torch.tensor([2.,4.,0.],dtype=A.dtype))@A
    H,rhs=ls.system(A,B,w,torch.eye(2,dtype=A.dtype),torch.eye(2,dtype=A.dtype))
    c,info=ls.solve(H,rhs,c0)
    assert info['active_atoms']==2 and info['retained_rank']==1
    assert torch.allclose(c,torch.tensor([2.,4.,5.],dtype=A.dtype),atol=1e-12)
    assert ls.metrics(A,B,c,w,torch.eye(2,dtype=A.dtype),torch.eye(2,dtype=A.dtype))['objective']<1e-25


def test_rounding_fixed_fields():
    a=torch.ones(2,3,dtype=torch.bfloat16); b=torch.ones(4,2,dtype=a.dtype)
    row=torch.ones(4,dtype=a.dtype); col=torch.ones(3,dtype=a.dtype)
    old=torch.ones(2,dtype=a.dtype)
    cs=[(a,b,row,col,old)]
    c=torch.tensor([-.31,.15],dtype=torch.float64)
    out=ls.rounded_components(cs,c)
    assert all(x is y for x,y in zip(cs[0][:-1],out[0][:-1]))
    assert torch.equal(out[0][-1],c.bfloat16())
    assert torch.equal(old,torch.ones_like(old))
