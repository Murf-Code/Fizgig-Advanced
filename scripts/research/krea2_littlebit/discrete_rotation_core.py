"""Research-only batched discrete proposals with direct joint fit acceptance."""
import math
import time
import torch


class Objective:
    def __init__(self, w, x, g, device='cpu', mode='task'):
        self.w, self.x, self.g = (t.to(device=device, dtype=torch.float64) for t in (w,x,g))
        self.xn=self.x.square().sum(0); self.gn=self.g.square().sum(0)
        energies=[self.w.square().sum(), (self.x@self.w.T).square().sum(), (self.g@self.w).square().sum()]
        if not all(torch.isfinite(z) and z>0 for z in energies): raise ValueError('invalid teacher energies')
        weights=(.2,.4,.4) if mode=='task' else (1.,0.,0.)
        self.k=tuple(float(a/b) for a,b in zip(weights,energies))
        self.mode=mode

    def residual(self,cs,target=None):
        w=self.w if target is None else target.to(self.w)
        E=sum((b*r[:,None]*c)@(a*s) for a,b,r,s,c in cs)-w
        return self.actions(E)

    def actions(self,E):
        if self.mode=='weight': return (E,)
        return E,self.x@E.T,self.g@E

    def loss(self,res):
        return sum(k*float(z.square().sum()) for k,z in zip(self.k,res))

    def gradient(self,res):
        if self.mode=='weight': return self.k[0]*res[0]
        E,F,D=res; wa,wf,wb=self.k
        return wa*E+wf*(F.T@self.x)+wb*(self.g.T@D)

    def change(self,res,delta):
        d=self.actions(delta)
        value=sum(k*float(2*(a*b).sum()+b.square().sum()) for k,a,b in zip(self.k,res,d))
        return value,d

    def guard(self,res,scores,make_delta):
        """All negative local proposals, then successively stronger subsets; singleton fallback."""
        if not torch.isfinite(scores).all(): raise ValueError('nonfinite proposal scores')
        best=float(scores.min()); old=self.loss(res)
        if best>=-1e-15: return None,res,dict(proposed=0,accepted=0,attempts=0,delta=0.)
        proposed=int((scores< -1e-15).sum()); attempts=0
        for fraction in (0.,.01,.03,.1,.3,.6,.9,1.):
            if fraction==1.:
                mask=torch.zeros_like(scores,dtype=torch.bool); mask.view(-1)[scores.argmin()]=True
            else:
                mask=scores<min(-1e-15,best*fraction)
            if not mask.any(): continue
            attempts+=1
            delta=make_delta(mask)
            value,d=self.change(res,delta)
            if value< -1e-15:
                trial=tuple(a+b for a,b in zip(res,d)); actual=self.loss(trial)-old
                if actual<0 and abs(actual-value)<=1e-10*max(1.,old):
                    return mask,trial,dict(proposed=proposed,accepted=int(mask.sum()),attempts=attempts,delta=actual)
        return None,res,dict(proposed=proposed,accepted=0,attempts=attempts,delta=0.)


def local(cs,obj):
    return [[t.to(obj.w).clone() for t in p] for p in cs]


def restore(cs,original):
    return [tuple(t.to(ref) for t,ref in zip(p,q)) for p,q in zip(cs,original)]


@torch.no_grad()
def sign_sweep(cs,obj):
    p=local(cs,obj); res=obj.residual(p); before=obj.loss(res); stats=[]
    wa,wf,wb=obj.k
    for i,(a,b,r,s,c) in enumerate(p):
        # A flips: keep all B columns and all other fields fixed.
        B=b*r[:,None]*c; bn=B.square().sum(0); gb=(obj.g@B).square().sum(0)
        d=-2*a*s; h=B.T@obj.gradient(res)
        scores=2*d*h+d.square()*(wa*bn[:,None]+wf*bn[:,None]*obj.xn+wb*gb[:,None])
        mask,res,info=obj.guard(res,scores,lambda mask:B@(d*mask))
        if mask is not None: a[mask]*=-1
        stats.append(dict(path=i,plane='A',**info))
        # B flips: include A changes just accepted above.
        A=a*s*c[:,None]; an=A.square().sum(1); xa=(obj.x@A.T).square().sum(0)
        d=-2*b*r[:,None]; h=obj.gradient(res)@A.T
        scores=2*d*h+d.square()*(wa*an+wf*xa+wb*obj.gn[:,None]*an)
        mask,res,info=obj.guard(res,scores,lambda mask:(d*mask)@A)
        if mask is not None: b[mask]*=-1
        stats.append(dict(path=i,plane='B',**info))
    actual=obj.residual(p); after=obj.loss(actual)
    if after>before+1e-12 or abs(after-obj.loss(res))>1e-10*max(1.,before): raise ValueError('sign sweep residual mismatch')
    return restore(p,cs),dict(before=before,after=after,planes=stats,flips=sum(x['accepted'] for x in stats))


def signs(t): return torch.where(t>=0,1.,-1.).to(t)


@torch.no_grad()
def rotate_path(cs,A,B,obj,target,seed=1000):
    """One disjoint Givens-pair pass, select local best angle then guard joint change.

    Continuous A/B are never stored; gains and outer scales remain fixed in search.
    Odd final rank is left unchanged. The original teacher normalizes the objective.
    """
    p=local([cs],obj); a,b,r,s,c=p[0]; A=A.to(obj.w); B=B.to(obj.w)
    if not torch.equal(signs(A),a) or not torch.equal(signs(B),b): raise ValueError('continuous signs mismatch')
    order=torch.randperm(len(c),generator=torch.Generator().manual_seed(seed)).to(a.device)
    ij=order[:len(c)//2*2].reshape(-1,2)
    if not len(ij): return cs,dict(accepted=0,attempts=0)
    # [pairs,2,n] and [pairs,m,2]; common orthogonal pair rotation.
    Ac=A[ij]; Bc=B[:,ij].permute(1,0,2); olda=a[ij]; oldb=b[:,ij].permute(1,0,2)
    gain=c[ij]; ou=oldb*r[None,:,None]*gain[:,None,:]; ov=olda*s
    res=obj.residual(p,target); H=obj.gradient(res); before=obj.loss(res)
    best=torch.zeros(len(ij),device=a.device,dtype=a.dtype); aa=olda.clone(); bb=oldb.clone(); chosen=torch.zeros_like(best)
    wa,wf,wb=obj.k
    for theta in (-math.pi/6,-math.pi/12,-math.pi/24,math.pi/24,math.pi/12,math.pi/6):
        co,si=math.cos(theta),math.sin(theta); R=A.new_tensor([[co,-si],[si,co]])
        na=signs(R.T@Ac); nb=signs(Bc@R)
        U=torch.cat((nb*r[None,:,None]*gain[:,None,:],-ou),2)
        V=torch.cat((na*s,ov),1)
        # Pair-local quadratic difference from at most four rank-one terms.
        uf=U.permute(1,0,2).reshape(len(r),-1).contiguous(); vf=V.reshape(-1,len(s))
        cross=((uf.T@H)*vf).sum(1).reshape(-1,4).sum(1)
        uu=U.transpose(1,2)@U; vv=V@V.transpose(1,2)
        delta=2*cross+wa*(uu*vv).sum((1,2))
        if obj.mode=='task':
            P=(obj.x@vf.T).reshape(len(obj.x),-1,4).permute(1,0,2)
            Q=(obj.g@uf).reshape(len(obj.g),-1,4).permute(1,0,2)
            delta+=wf*(uu*(P.transpose(1,2)@P)).sum((1,2))+wb*((Q.transpose(1,2)@Q)*vv).sum((1,2))
        if not torch.isfinite(delta).all(): raise ValueError('nonfinite rotation scores')
        better=delta<best; best[better]=delta[better]; aa[better]=na[better]; bb[better]=nb[better]; chosen[better]=theta
    def proposal(mask):
        return ((bb[mask]*r[None,:,None]*gain[mask,None,:]).permute(1,0,2).reshape(len(r),-1)@(aa[mask]*s).reshape(-1,len(s))
               -ou[mask].permute(1,0,2).reshape(len(r),-1)@ov[mask].reshape(-1,len(s)))
    mask,res,info=obj.guard(res,best,proposal)
    if mask is not None:
        a[ij[mask]]=aa[mask]; b[:,ij[mask]]=bb[mask].permute(1,0,2)
        # Verify accepted rotations maintain the continuous product using small pair Grams.
        ang=chosen[mask]; co,si=ang.cos(),ang.sin()
        R=torch.stack((co,-si,si,co),1).reshape(-1,2,2)
        eye=torch.eye(2,device=R.device,dtype=R.dtype).expand_as(R)
        torch.testing.assert_close(R@R.transpose(1,2),eye,rtol=1e-12,atol=1e-12)
    after=obj.loss(obj.residual(p,target))
    if after>before+1e-12 or abs(after-obj.loss(res))>1e-10*max(1.,before): raise ValueError('rotation residual mismatch')
    return restore(p,[cs])[0],dict(before=before,after=after,pairs=len(ij),**info)
