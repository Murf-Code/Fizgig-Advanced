"""CPU/CUDA research validation independent of the diffusion trainer."""
import torch
from fizgig.training.matrix_geometry import delta_norm
from fizgig.training.matrix_geometry_prototypes import scaled_directions, ScaledLoRASGD


def validate_prototypes(device):
    g=torch.Generator(device=device).manual_seed(741)
    def rand(*shape):
        return torch.randn(*shape,device=device,dtype=torch.float64,generator=g)
    a,b,weight=rand(5,13),rand(11,5),rand(11,13)
    q=torch.linalg.qr(rand(5,5))[0]
    ga,gb=b.T@weight,weight@a.T
    da,db=scaled_directions(a,b,ga,gb)
    dar,dbr=scaled_directions(q@a,b@q.T,q@ga,gb@q.T)
    torch.testing.assert_close(dar,q@da)
    torch.testing.assert_close(dbr,db@q.T)
    torch.testing.assert_close(da,torch.linalg.inv(b.T@b+1e-3*torch.eye(5,device=device,dtype=a.dtype))@ga)
    z=torch.zeros_like(b)
    dz,du=scaled_directions(a,z,z.T@weight,weight@a.T)
    assert torch.count_nonzero(dz)==0 and torch.isfinite(du).all() and du.norm()>0

    # The tangent-only cap is not an exact finite update cap: a scalar
    # counterexample embedded in rank-one factors. This does not assert
    # that a particular TSD run reaches this configuration.
    a0=b0=torch.ones(1,1,device=device,dtype=torch.float64)
    a1=b1=a0+0.05
    linear=(b0@(a1-a0)+(b1-b0)@a0).norm()
    actual=delta_norm(a0,b0,a1,b1)
    assert actual > linear
    torch.testing.assert_close(actual,torch.tensor(0.1025,device=device,dtype=a0.dtype))

    # Rank-deficient QR returns an arbitrary completion beyond the column span.
    deficient=torch.zeros(7,3,device=device,dtype=torch.float64)
    deficient[0,0]=1
    basis=torch.linalg.qr(deficient,mode="reduced")[0]
    assert torch.linalg.matrix_rank(basis)==3
    assert torch.linalg.matrix_rank(deficient)==1

    # Resume reproduces step count and spectral-balancing schedule.
    def params(): return torch.nn.Parameter(a.clone()),torch.nn.Parameter(b.clone())
    def update(opt,aa,bb):
        opt.zero_grad(set_to_none=True)
        (bb@aa-weight).square().mean().backward()
        opt.step()
    aa,bb=params()
    opt=ScaledLoRASGD([(aa,bb)],lr=0.01,balance_every=2)
    update(opt,aa,bb)
    import copy
    state=copy.deepcopy(opt.state_dict())
    ar,br=torch.nn.Parameter(aa.detach().clone()),torch.nn.Parameter(bb.detach().clone())
    resumed=ScaledLoRASGD([(ar,br)],lr=0.01,balance_every=2)
    resumed.load_state_dict(state)
    update(opt,aa,bb)
    update(resumed,ar,br)
    torch.testing.assert_close(bb@aa,br@ar)
    torch.testing.assert_close(aa,ar)
    torch.testing.assert_close(bb,br)
    return {"orthogonal_equivariance":"passed","zero_up":"passed",
            "tangent_cap_counterexample":{"linear":float(linear),"actual":float(actual)},
            "rank_deficient_qr_counterexample":"passed",
            "prototype_resume":"passed"}


def test_prototypes_cpu():
    validate_prototypes(torch.device("cpu"))
