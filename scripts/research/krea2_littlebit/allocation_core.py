"""Fixed-budget row precision allocation, using bounded joint swap proposals."""
import math
import torch

CAPS=(19,20,22,24) # eighths of one bit


def budget(m,n,eighths):
    if m<1 or n%64 or eighths not in CAPS: raise ValueError('invalid budget')
    cap=m*n*eighths//64
    base=m*(n//4+2*(n//64))+(m+7)//8
    return max(0,min(m,(cap-base)//(n//8))),cap


class Allocation:
    """Compute exact quadratic in row choices, without an m x m Gram."""
    @torch.no_grad()
    def __init__(self,z,x,g,d2,d3,device='cpu',dtype=torch.float64):
        self.x=x.to(device=device,dtype=dtype); self.g=g.to(self.x)
        z=z.to(self.x); e=d2.to(self.x)-z; self.v=d3.to(self.x)-d2.to(self.x)
        energies=(z.square().sum(),(self.x@z.T).square().sum(),(self.g@z).square().sum())
        self.k=tuple(float(a/b) for a,b in zip((.2,.4,.4),energies))
        if not all(math.isfinite(k) and k>0 for k in self.k): raise ValueError('invalid energies')
        a,b,c=self.k; f=self.x@e.T; vf=self.x@self.v.T
        self.wdelta=(2*e*self.v+self.v.square()).sum(1)
        self.local=a*self.wdelta+b*(2*f*vf+vf.square()).sum(0)
        self.d0=self.g@e
        self.diag=c*self.g.square().sum(0)*self.v.square().sum(1)
        self.base=float(a*e.square().sum(dtype=torch.float64)+b*f.square().sum(dtype=torch.float64))
        self.initial=self.base+c*float(self.d0.square().sum(dtype=torch.float64))

    def backward(self,mask):
        return self.d0+self.g[:,mask]@self.v[mask]

    def score(self,mask,D):
        return self.base+float(self.local[mask].sum(dtype=torch.float64))+self.k[2]*float(D.square().sum(dtype=torch.float64))

    def linear(self,D):
        return self.local+2*self.k[2]*((self.g.T@D)*self.v).sum(1)

    def priority(self): return self.linear(self.d0)+self.diag

    @torch.no_grad()
    def refine(self,mask,rounds=8,shortlist=32):
        mask=mask.to(self.v.device).clone(); D=self.backward(mask); before=self.score(mask,D); history=[]
        count=int(mask.sum()); c=self.k[2]
        for _ in range(rounds):
            lin=self.linear(D)
            add=(lin+self.diag).masked_fill(mask,float('inf'))
            remove=(-lin+self.diag).masked_fill(~mask,float('inf'))
            if count==0 or count==len(mask): break
            ii=add.argsort(stable=True)[:min(shortlist,len(mask)-count)]
            jj=remove.argsort(stable=True)[:min(shortlist,count)]
            cross=c*(self.g[:,ii].T@self.g[:,jj])*(self.v[ii]@self.v[jj].T)
            pair=add[ii,None]+remove[jj][None,:]-2*cross
            # Greedy disjoint candidate pairing; exact joint guard for prefixes.
            pairs=[]
            for k in range(min(len(ii),len(jj))):
                at=int(pair.argmin()); i,j=divmod(at,len(jj))
                if not torch.isfinite(pair[i,j]) or float(pair[i,j])>=-1e-14: break
                pairs.append((int(ii[i]),int(jj[j]))); pair[i,:]=float('inf'); pair[:,j]=float('inf')
            accepted=0; old=self.score(mask,D); k=len(pairs)
            while k:
                up,down=zip(*pairs[:k]); up=list(up); down=list(down)
                dv=self.g[:,up]@self.v[up]-self.g[:,down]@self.v[down]
                trial=mask.clone(); trial[up]=True; trial[down]=False
                new=self.score(trial,D+dv)
                if new<old-1e-12:
                    # Direct backward reconstruction prevents drift or false acceptance.
                    actual=self.backward(trial); direct=self.score(trial,actual)
                    if direct<old-1e-12:
                        mask,D=trial,actual; accepted=k; break
                k//=2
            history.append(dict(proposed=len(pairs),accepted=accepted,before=old,after=self.score(mask,D)))
            if not accepted: break
        after=self.score(mask,self.backward(mask))
        if int(mask.sum())!=count or after>before+1e-10: raise ValueError('joint refinement mismatch')
        return mask.cpu(),dict(before=before,after=after,swaps=sum(t['accepted'] for t in history),rounds=history)
