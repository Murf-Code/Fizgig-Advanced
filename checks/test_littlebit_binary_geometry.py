from pathlib import Path
import sys
import torch
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research/krea2_littlebit"))
import binary_geometry_screen as h


def test_same_rank_and_payload():
    m,n=1536,6144
    cap=m*n//8
    ranks=h.allocations(m,n,cap,m)
    assert ranks=={50:(597,597),75:(896,298),25:(298,896)}
    assert all(sum(h.q.size(m,n,r) for r in pair)==1179348<=cap for pair in ranks.values())


def test_unwhitening_preserves_product():
    torch.manual_seed(42)
    w=torch.randn(19,31,dtype=torch.float64)
    g=torch.randn(11,19,dtype=torch.float64)
    L=h.backward_metric(w,g)
    u,s,vh=torch.linalg.svd(L.T@w,full_matrices=False)
    a,b=h.old.balanced(u,s,vh.T,7)
    target=torch.linalg.solve_triangular(L.T,b@a,upper=True)
    ar,br=h.unwhiten(a,b,L)
    assert torch.allclose((br@ar).double(),target,atol=2e-6,rtol=1e-5)
    assert torch.allclose(ar.norm(dim=1),br.norm(dim=0),atol=1e-6)
    with pytest.raises(ValueError,match="energies"):
        h.backward_metric(w,torch.zeros_like(g))


def test_residual_matches_packed_first_path():
    torch.manual_seed(42)
    w=torch.randn(19,31).bfloat16()
    path=h.q.Path(torch.randn(7,31),torch.randn(19,7))
    cs,_=h.e.packed([path],*w.shape)
    expected=w.float()-h.q.weight(tuple(t.float() for t in cs[0]))
    assert torch.equal(h.residual_target(w,path),expected)
