from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research/krea2_littlebit"))
import binary_rank_core as core
import native_codec as codec


def test_rotation_and_payload():
    torch.manual_seed(42)
    a=torch.randn(17,32); b=torch.randn(41,17)
    aa,bb=core.rotate(a,b,chunk=8)
    torch.testing.assert_close(bb@aa,b@a,rtol=1e-4,atol=1e-4)
    da,db,nbytes=core.binarize(aa,bb,codec._pack,codec._unpack)
    assert nbytes==core.binary_bytes(41,32,17)
    assert da.dtype==db.dtype==torch.bfloat16
    assert core.rank_grid(41,32,2048)==[32]


def test_scale_fit_reduces_error_for_separable_distortion():
    torch.manual_seed(42)
    a=torch.randn(8,32).bfloat16(); b=torch.randn(41,8).bfloat16()
    w=((b.float()@a.float())*torch.linspace(.3,2,41)[:,None]*torch.linspace(.5,1.5,32)[None,:]).bfloat16()
    row,col=core.scale_fit(w,a,b,tile=13)
    assert core.weight_error(w,a,b,row,col)<core.weight_error(w,a,b)
    assert row.nbytes+col.nbytes==2*(41+32)
    y,dx=core.actions(torch.randn(7,32),torch.randn(7,41),a,b,row,col)
    assert y.shape==(7,41) and dx.shape==(7,32)
    assert torch.isfinite(y).all() and torch.isfinite(dx).all()
