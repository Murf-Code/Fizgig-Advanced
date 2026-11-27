import math
import torch
from fizgig.training.selective_effective_score import pair_terms, block_score

def dense(a,b,a0,b0,g,s):
    pu=b@torch.linalg.pinv(b,rtol=1e-12)
    pv=torch.linalg.pinv(a,rtol=1e-12)@a
    z=pu@g+g@pv-pu@g@pv
    d=s*(b@a-b0@a0)
    p=pu@d+d@pv-pu@d@pv
    return dict(dot=float((z*d).sum()),d2=float(d.square().sum()),
                z2=float(z.square().sum()),p2=float(p.square().sum()))

def test_dense_and_gauges():
    gen=torch.Generator().manual_seed(42)
    def rand(*shape): return torch.randn(shape,generator=gen,dtype=torch.float64)
    for i in range(100):
        a,b,a0,b0,g=rand(3,7),rand(9,3),rand(3,7),rand(9,3),rand(9,7)
        if i%5==0: b.zero_()
        if i%5==1: a.zero_()
        if i%5==2: b[:,2]=b[:,0]+b[:,1]
        if i%5==3: a[2]=a[0]+a[1]
        s=.7
        expect=dense(a,b,a0,b0,g,s)
        r=rand(3,3)+4*torch.eye(3,dtype=a.dtype)
        r0=rand(3,3)+4*torch.eye(3,dtype=a.dtype)
        for x,y,x0,y0 in ((a,b,a0,b0),(torch.linalg.solve(r,a),b@r,
                                           torch.linalg.solve(r0,a0),b0@r0)):
            t=pair_terms(x,y,x0,y0,s*y.T@g,s*g@x.T,s)
            assert t["compatible"]
            for key,value in expect.items():
                assert math.isclose(t[key],value,rel_tol=1e-10,abs_tol=1e-10),(i,key,t[key],value)
            q=block_score([t])
            assert -1<=q["score"]<=1
            assert 0<=q["observability"]<=1

def test_unknown_and_unobservable():
    a=torch.tensor([[1.,0.]],dtype=torch.float64); b=a.T
    a0=torch.tensor([[0.,1.]],dtype=a.dtype); b0=a0.T
    g=torch.diag(torch.tensor([1.,2.],dtype=a.dtype))
    t=pair_terms(a,b,a0,b0,b.T@g,g@a.T,1.)
    q=block_score([t])
    assert math.isclose(q["score"],-1/math.sqrt(2))
    assert math.isclose(q["observability"],1/math.sqrt(2))
    assert block_score([pair_terms(a,b,a,b,b.T@g,g@a.T,1.)])["score"] is None
    assert block_score([pair_terms(a,b,a0,b0,10*b.T@g,g@a.T,1.)])["reason"]=="inconsistent_factor_gradients"
    assert block_score([None])["score"] is None

def test_near_checkpoint_and_direct_sum():
    gen=torch.Generator().manual_seed(10)
    a=torch.randn(4,9,generator=gen,dtype=torch.float64)
    b=torch.randn(11,4,generator=gen,dtype=torch.float64)
    g=torch.randn(11,9,generator=gen,dtype=torch.float64)
    t=pair_terms(a,b,a+1e-8,b-1e-8,b.T@g,g@a.T,1.)
    d=dense(a,b,a+1e-8,b-1e-8,g,1.)
    assert math.isclose(t["d2"],d["d2"],rel_tol=1e-6)
    q=block_score([t,t])
    assert math.isclose(q["score"],block_score([t])["score"])
