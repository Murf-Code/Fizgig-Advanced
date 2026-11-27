"""Exact decoding checks and grouped arithmetic parity on synthetic matrices."""
from pathlib import Path
import sys
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research/krea2_littlebit"))
import native_codec as codec
import geometry_codec as geo
import grouped_q6 as grouped


@pytest.mark.parametrize("m,n",[(128,64),(259,128),(513,256)])
@pytest.mark.parametrize("batched",[False,True])
def test_grouped_decode(m,n,batched):
    torch.manual_seed(42)
    p,s=codec.encode(torch.randn(m,n).bfloat16(),"q6-had")
    ref=torch.cat([geo._decode_tile(p,s,i,min(i+128,m),torch.bfloat16,codec._unpack)
                   for i in range(0,m,128)])
    assert torch.equal(grouped.decode(p,s,0,m,torch.bfloat16,batched),ref)
    if m>128:
        assert torch.equal(grouped.decode(p,s,128,m,torch.bfloat16,batched),ref[128:])
    x=torch.randn(7,n).bfloat16()
    dy=torch.randn(7,m).bfloat16()
    y=geo.forward(x,p,s,codec._unpack,codec.factors)
    dx=geo.backward(dy,p,s,codec._unpack,codec.factors)
    torch.testing.assert_close(grouped.forward(x,p,s,256,batched).float(),y.float(),rtol=.02,atol=.05)
    torch.testing.assert_close(grouped.backward(dy,p,s,256,batched).float(),dx.float(),rtol=.02,atol=.15)
