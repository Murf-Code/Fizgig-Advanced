from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research/krea2_littlebit"))
import mixed_factor_core as mix
import native_codec as codec


@pytest.mark.parametrize("bits",[1,2,3,4,8,16])
def test_precision(bits):
    torch.manual_seed(42)
    w=torch.randn(17,31); w[0]=0
    d,nb=mix.quant_rows(w,bits,codec._pack,codec._unpack,tile=5)
    assert nb==mix.factor_bytes(17,31,bits)
    assert d.shape==w.shape and d.dtype==torch.bfloat16
    assert torch.isfinite(d).all() and torch.count_nonzero(d[0])==0
    if bits==16: assert torch.equal(d,w.bfloat16())
    d2,_=mix.quant_rows(w,bits,codec._pack,codec._unpack,tile=5)
    assert torch.equal(d,d2)


@pytest.mark.parametrize("a,b",[(1,1),(1,2),(2,1),(2,2),(1,4),(4,1),(4,4),(8,8),(16,16)])
def test_budget(a,b):
    for cap in (1,100,1000,10000):
        r=mix.budget_rank(41,37,a,b,cap,29)
        assert r==0 or mix.pair_bytes(41,37,r,a,b)<=cap
        assert r==29 or mix.pair_bytes(41,37,r+1,a,b)>cap
