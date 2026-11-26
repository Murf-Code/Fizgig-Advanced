import torch
import pytest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research/krea2_littlebit"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import geometry_codec as c
import native_codec as native


def unused(*args,**kwargs):
    raise AssertionError('unexpected legacy call')


def test_hadamard():
    x=torch.randn(4,128)
    assert torch.allclose(c.hadamard(c.hadamard(x)),x,atol=1e-6)
    assert torch.allclose(c.hadamard(x).norm(),x.norm(),atol=1e-5)
    w=torch.randn(7,128)
    assert torch.allclose(c.hadamard(x)@c.hadamard(w).T,x@w.T,atol=1e-5)


@pytest.mark.parametrize('bits',[4,6])
def test_scale_fit(bits):
    w=torch.randn(64,64)*torch.linspace(.1,3,64)
    levels=(1<<(bits-1))-1
    q0,s0=c.fitted_codes(w,bits,False)
    q1,s1=c.fitted_codes(w,bits,True)
    err=lambda q,s:(w-(q.float()-levels)*s.float()[:,None]).square().sum(-1)
    assert torch.all(err(q1,s1)<=err(q0,s0)+1e-5)


def verify(name,w,rank):
    p,s=native.encode(w,name,rank=rank)
    native.validate(p,s)
    x=torch.randn(2,3,w.shape[1],requires_grad=True)
    dy=torch.randn(2,3,w.shape[0])
    y=native.linear(x,p,s)
    dx,=torch.autograd.grad(y,x,dy)
    dense=native.linear(torch.eye(w.shape[1]),p,s).T
    assert torch.allclose(y,x@dense.T,atol=3e-5,rtol=3e-5)
    assert torch.allclose(dx,dy@dense,atol=4e-5,rtol=4e-5)
    with pytest.raises(ValueError):
        native.validate(p[:-1],s)
    return p,s


@pytest.mark.parametrize('name',sorted(c.UNIFORM))
def test_forward_backward(name):
    torch.manual_seed(13)
    w=torch.randn(131,128).bfloat16()
    p,s=verify(name,w,8)
    assert p.numel()==131*(128*int(name[1])//8+4)
    base,_=native.encode(w,name[:2],rank=8)
    assert base.numel()==p.numel()


@pytest.mark.parametrize('name',['littlebit-had','littlebit2-had'])
def test_latent(name):
    torch.manual_seed(5)
    w=torch.randn(80,128).bfloat16()
    p,s=verify(name,w,64)
    q,_=native.encode(w,name.removesuffix('-had'),rank=64)
    assert p.numel()==q.numel()
    assert s['latent_hadamard']


def fake_hqq():
    # Layout-only stand-in; not an HQQ accuracy or solver test.
    def quant(w,group_size,compute_dtype,device):
        rows,n=w.shape; g=group_size
        q=(w.float().reshape(-1,g).round()+8).clamp(0,15).byte()
        half=len(q)//2
        return (q[:half]<<4)|q[half:],torch.ones(len(q),1,dtype=torch.uint8),torch.full((len(q),1),8,dtype=torch.uint8),torch.zeros(rows,4,dtype=torch.int32)
    def dequant(q,s,z,meta,shape,dtype,group_size):
        assert meta.shape==(shape[0],4)
        return (torch.cat((q>>4,q&15)).float()-z.float()).reshape(shape).to(dtype)
    return quant,dequant


@pytest.mark.parametrize('name',['hqq','hqq-had','lb2-hqq','lb2-hqq-had'])
def test_hqq_layout_and_residual(name,monkeypatch):
    monkeypatch.setattr(c,'_hqq_api',fake_hqq)
    torch.manual_seed(7)
    w=torch.randn(131,128).bfloat16()
    p,s=verify(name,w,8)
    expected=131*(64+32+16)
    if name.startswith('lb2'):
        expected+=s['base_bytes']
    assert p.numel()==expected


@pytest.mark.parametrize('name',['hqq','hqq-had','lb2-hqq','lb2-hqq-had'])
def test_real_hqq_when_installed(name):
    pytest.importorskip('hqq')
    torch.manual_seed(7)
    w=torch.randn(131,128).bfloat16()
    verify(name,w,8)
