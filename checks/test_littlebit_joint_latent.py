from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research/krea2_littlebit"))
import joint_latent as j


def test_joint_and_alternating():
    torch.manual_seed(42)
    a=torch.where(torch.randn(9,31)>0,1.,-1.).bfloat16()
    b=torch.where(torch.randn(37,9)>0,1.,-1.).bfloat16()
    c=torch.linspace(-.7,1.2,9).bfloat16()
    w=(b.float()*c.float())@a.float()
    assert torch.equal(j.fit_latent(w,a,b,tile=11),c)
    row=torch.linspace(.4,1.5,37).bfloat16()
    col=torch.linspace(.6,1.3,31).bfloat16()
    w=(w*row.float()[:,None]*col.float()[None,:]).bfloat16()
    r,s=torch.ones_like(row),torch.ones_like(col)
    def error(c,r,s):
        return float((((b.float()*c.float())@a.float())*r.float()[:,None]*s.float()[None,:]-w.float()).norm()/w.float().norm())
    f=j.fit_latent(w,a,b)
    before=error(f,r,s)
    for _ in range(5):
        r,s=j.fit_outer(w,a,b,f,r,s,tile=11)
        f=j.fit_latent(w,a,b,r,s,tile=11)
    assert error(f,r,s)<before*.1
    ad=torch.cat([a,a[:1]])
    bd=torch.cat([b,b[:,:1]],1)
    assert torch.isfinite(j.fit_latent(w,ad,bd)).all()
    assert j.payload_bytes(37,31,9,True)==(9*31+7)//8+(9*37+7)//8+18+2*(37+31)
