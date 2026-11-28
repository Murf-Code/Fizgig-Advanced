"""Conditional least squares for fixed binary rank-one atoms (research only)."""
import torch

WEIGHTS=(.2,.4,.4)
RTOL=1e-12


@torch.no_grad()
def atoms(cs):
    # Keep the decoded signs, row and column scales fixed. Latent gains are
    # the actual stored coefficients, not multipliers relative to Path bases.
    A=torch.cat([a.double()*col.double()[None,:] for a,b,row,col,c in cs],0)
    B=torch.cat([b.double()*row.double()[:,None] for a,b,row,col,c in cs],1)
    c=torch.cat([t[-1].double() for t in cs])
    return A,B,c


@torch.no_grad()
def system(A,B,w,x,g,weights=WEIGHTS):
    # All calculations in FP64, in original weight coordinates.
    A,B,w,x,g=(t.double() for t in (A,B,w,x,g))
    y=x@w.T
    dx=g@w
    energies=[w.square().sum(),y.square().sum(),dx.square().sum()]
    if not all(torch.isfinite(v) and v>0 for v in energies):
        raise ValueError("positive finite teacher energies required")
    if len(weights)!=3 or min(weights)<0 or abs(sum(weights)-1)>1e-12:
        raise ValueError("invalid weights")
    wa,wf,wb=[v/z for v,z in zip(weights,energies)]
    P=x@A.T
    Q=g@B
    aa=A@A.T
    bb=B.T@B
    H=wa*(aa*bb)+wf*((P.T@P)*bb)+wb*((Q.T@Q)*aa)
    rhs=(wa*((B.T@w)*A).sum(1)
         +wf*(P*(y@B)).sum(0)
         +wb*(Q*(dx@A.T)).sum(0))
    H=(H+H.T)*.5
    if not torch.isfinite(H).all() or not torch.isfinite(rhs).all():
        raise ValueError("nonfinite normal equations")
    return H,rhs


@torch.no_grad()
def solve(H,rhs,c0,rtol=RTOL):
    # Diagonal normalization removes arbitrary atom magnitude disparity.
    # Null/near-null normalized directions retain the starting coefficient.
    # Solve for displacement, not a minimum-norm replacement.
    diag=H.diag()
    if (diag<0).any() or not (diag>0).any():
        raise ValueError("invalid atom Gram diagonal")
    active=diag>0
    d=diag[active].sqrt()
    M=H[active][:,active]/d[:,None]/d[None,:]
    residual=(rhs-H@c0)[active]/d
    vals,U=torch.linalg.eigh((M+M.T)*.5)
    limit=rtol*vals[-1]
    if vals[-1]<=0 or vals[0]<-10*limit:
        raise ValueError("normal equations not numerically PSD")
    keep=vals>limit
    z=U[:,keep]@((U[:,keep].T@residual)/vals[keep])
    c=c0.clone()
    c[active]+=z/d
    if not torch.isfinite(c).all():
        raise ValueError("nonfinite solved gains")
    grad=(H@c-rhs)[active]/d
    ref=rhs[active]/d
    info={"active_atoms":int(active.sum()),"retained_rank":int(keep.sum()),
          "eigenvalue_min":float(vals[0]),"eigenvalue_max":float(vals[-1]),
          "relative_cutoff":rtol,"retained_condition":float(vals[-1]/vals[keep][0]),
          "relative_normal_residual":float(grad.norm()/ref.norm().clamp_min(1e-30))}
    return c,info


@torch.no_grad()
def metrics(A,B,c,w,x,g,weights=WEIGHTS):
    # Direct evaluation independent of the Gram identity; avoids cancellation
    # from subtracting nearly equal quadratic values.
    E=(B*c)@A-w
    y=x@w.T
    dx=g@w
    terms=[E.square().sum()/w.square().sum(),
           (x@E.T).square().sum()/y.square().sum(),
           (g@E).square().sum()/dx.square().sum()]
    if not all(torch.isfinite(t) for t in terms):
        raise ValueError("nonfinite direct objective")
    return {"weight_relative_squared":float(terms[0]),
            "forward_relative_squared":float(terms[1]),
            "backward_relative_squared":float(terms[2]),
            "objective":sum(v*float(t) for v,t in zip(weights,terms))}


def rounded_components(cs,c):
    # Only the latent BF16 coefficient field changes. Sign planes and both
    # outer scale vectors remain bit-identical.
    result=[]
    start=0
    for a,b,row,col,old in cs:
        end=start+old.numel()
        gain=c[start:end].to(device=old.device,dtype=torch.bfloat16)
        if gain.shape!=old.shape or not torch.isfinite(gain).all():
            raise ValueError("invalid rounded gain")
        result.append((a,b,row,col,gain))
        start=end
    if start!=c.numel():
        raise ValueError("latent length mismatch")
    return result
