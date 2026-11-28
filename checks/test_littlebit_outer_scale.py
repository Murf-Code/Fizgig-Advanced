import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research/krea2_littlebit"))
from outer_scale_core import outer_system,pcg,rounded_outer

def test_outer_scale():
    torch.manual_seed(42)
    m,n,r=5,9,3; w=torch.randn(m,n,dtype=torch.float64); x=torch.randn(7,n,dtype=w.dtype); g=torch.randn(8,m,dtype=w.dtype)
    cs=[tuple(t.bfloat16() for t in (torch.randn(r,n).sign(),torch.randn(m,r).sign(),torch.rand(m)+.2,torch.rand(n)+.2,torch.rand(r)+.2)) for _ in range(2)]
    def dense(cs):
        return sum((b.double()*row.double()[:,None]*c.double())@(a.double()*col.double()) for a,b,row,col,c in cs)
    def embed(v):
        return torch.cat([a.flatten()*s for a,s in zip((v,x@v.T,g@v),(.2**.5/w.norm(),.4**.5/(x@w.T).norm(),.4**.5/(g@w).norm()))])
    for axis in (2,3):
        start,rhs,diag,op,mem=outer_system(cs,w,x,g,axis)
        cols=[]
        for i in range(len(start)):
            q=[list(t) for t in cs]; v=torch.zeros_like(start); v[i]=1
            for p,s in zip(q,v.chunk(2)): p[axis]=s
            cols.append(embed(dense(q)))
        M=torch.stack(cols,1); H=M.T@M; expected=M.T@embed(w-dense(cs))
        torch.testing.assert_close(rhs,expected); torch.testing.assert_close(diag,H.diag())
        torch.testing.assert_close(torch.stack([op(v) for v in torch.eye(len(start),dtype=w.dtype)],1),H)
        d,info=pcg(rhs,diag,op,maxiter=100,rtol=1e-10)
        assert info['converged']; torch.testing.assert_close(H@d,rhs,atol=1e-9,rtol=1e-9)
        fitted=rounded_outer(cs,start+d,axis)
        assert embed(w-dense(fitted)).norm()<embed(w-dense(cs)).norm()
        assert all(torch.equal(p[k],q[k]) for p,q in zip(cs,fitted) for k in range(5) if k!=axis)
    d,info=pcg(torch.tensor([0.,2.]),torch.tensor([0.,2.]),lambda v:torch.tensor([0.,2.])*v)
    torch.testing.assert_close(d,torch.tensor([0.,1.])); assert info['converged']
    try: outer_system(cs,w,x,g,3,1)
    except MemoryError: pass
    else: raise AssertionError('missing memory guard')
