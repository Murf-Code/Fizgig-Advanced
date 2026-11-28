"""Joint fixed-sign outer-scale least squares, CPU FP64 research only."""
import torch

@torch.no_grad()
def outer_system(cs,w,x,g,axis,gram_limit=2*1024**3):
    if len(cs)!=2 or axis not in (2,3):
        raise ValueError('two paths and row/column axis required')
    w,x,g=(t.cpu().double() for t in (w,x,g))
    y=x@w.T; z=g@w
    energies=[w.square().sum(),y.square().sum(),z.square().sum()]
    if not all(torch.isfinite(t) and t>0 for t in energies):
        raise ValueError('invalid teacher energies')
    wa,wf,wb=[a/b for a,b in zip((.2,.4,.4),energies)]
    if axis==2:
        T=[(b.double()*c.double())@(a.double()*s.double()) for a,b,r,s,c in cs]
        h,k,wl,wg=x,g,wf,wb
    else:
        T=[(((b.double()*r.double()[:,None])*c.double())@a.double()).T.contiguous() for a,b,r,s,c in cs]
        w=w.T; h,k,wl,wg=g,x,wb,wf
    n=w.shape[0]
    # Four dense n-by-n FP64 matrices: shared covariance and three blocks.
    estimate=4*n*n*8
    if estimate>gram_limit: raise MemoryError('outer Gram storage limit exceeded')
    start=torch.cat([p[axis].double() for p in cs])
    E=w-sum(t*v[:,None] for t,v in zip(T,start.split(n)))
    HE=h@E.T; KE=k.T@(k@E)
    rhs=torch.cat([wa*(t*E).sum(1)+wl*(p*HE).sum(0)+wg*(t*KE).sum(1) for t,p in zip(T,[h@t.T for t in T])])
    P=[h@t.T for t in T]; K=k.T@k
    blocks=[]
    for i,j in ((0,0),(0,1),(1,1)):
        block=T[i]@T[j].T
        local=wa*(T[i]*T[j]).sum(1)+wl*(P[i]*P[j]).sum(0)
        block.mul_(K).mul_(wg); block.diagonal().add_(local)
        blocks.append(block)
    a,b,c=blocks
    def op(v):
        u,z=v.split(n)
        return torch.cat((a@u+b@z,b.T@u+c@z))
    return start,rhs,torch.cat((a.diag(),c.diag())),op,estimate

@torch.no_grad()
def pcg(rhs,diag,op,maxiter=40,rtol=1e-8):
    if not torch.isfinite(rhs).all() or not torch.isfinite(diag).all() or (diag<0).any():
        raise ValueError('invalid normal system')
    active=diag>0
    if (rhs[~active]!=0).any(): raise ValueError('nonzero rhs for null atom')
    inv=torch.zeros_like(diag); inv[active]=diag[active].rsqrt()
    r=rhs*inv; ref=r.norm(); d=torch.zeros_like(r); p=r.clone(); rr=r@r
    count=0
    if ref>0:
        for _ in range(maxiter):
            if rr.sqrt()<=rtol*ref: break
            Hp=inv*op(inv*p); den=p@Hp
            if not torch.isfinite(den) or den<=0: break
            step=rr/den; d+=step*p; r-=step*Hp; new=r@r; count+=1
            if new.sqrt()<=rtol*ref: break
            p=r+(new/rr)*p; rr=new
    delta=inv*d
    rel=float(((rhs-op(delta))*inv).norm()/ref) if ref>0 else 0.
    if not torch.isfinite(delta).all() or not torch.isfinite(torch.tensor(rel)): raise ValueError('nonfinite PCG result')
    return delta,dict(iterations=count,relative_normal_residual=rel,converged=rel<=rtol,active=int(active.sum()))

@torch.no_grad()
def rounded_outer(cs,values,axis):
    out=[]; pos=0
    for path in cs:
        p=list(path); n=p[axis].numel(); p[axis]=values[pos:pos+n].to(p[axis]); pos+=n
        if not torch.isfinite(p[axis]).all(): raise ValueError('nonfinite rounded scales')
        out.append(tuple(p))
    if pos!=values.numel(): raise ValueError('scale length mismatch')
    return out
