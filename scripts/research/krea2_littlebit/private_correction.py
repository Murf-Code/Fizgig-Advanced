"""Offline-only private/shared frozen corrections; no loader or trainer changes."""
import hashlib
import math
import torch
import torch.nn.functional as F
import native_codec as codec
import geometry_codec as geo

DESCRIPTOR_BYTES = 4096


def ranks(m,n,count=2):
    """Pair budget: all Q6 weights versus Q5 residuals plus BF16 factors."""
    if min(m,n,count)<=0 or n%64:
        raise ValueError("invalid pair dimensions")
    q6=count*m*(n*6//8+2*(n//64))
    q5=count*m*(n*5//8+2*(n//64))
    spare=q6-q5
    return {"budget_bytes":q6+DESCRIPTOR_BYTES,"q5_bytes":q5+DESCRIPTOR_BYTES,
            "private_rank":min(m,n,spare//(2*count*(m+n))),
            "shared_rank":min(count*m,n,spare//(2*(n+count*m)))}


def split_captures(trace):
    """Disjoint item-key groups; never split rows from the same capture/image."""
    steps={s["step"]:s for s in trace["steps"]}
    groups={}
    for r in trace["calibration"]:
        items=steps[r["step"]]["items"]
        if len(items)!=1 or not items[0]:
            raise ValueError("expected exactly one identified image per capture")
        groups.setdefault(items[0],[]).append(r)
    if len(groups)<3:
        raise ValueError("need >=3 distinct captured image IDs; no independent three-way split")
    keys=sorted(groups,key=lambda k:hashlib.sha256(("correction-v1:"+k).encode()).digest())
    return {"fit":[r for k in keys[:-2] for r in groups[k]],
            "development":groups[keys[-2]],"confirmation":groups[keys[-1]]}


def decoded_tiles(p,s):
    m,n=s["shape"]
    for i in range(0,m,geo.TILE):
        j=min(i+geo.TILE,m)
        w=geo._decode_tile(p,s,i,j,torch.float32,codec._unpack)
        yield i,j,geo.hadamard(w) if s["codec"].endswith("-had") else w


def svd_factors(weights,rank,shared=False):
    if rank<1:
        raise ValueError("no rank fits budget")
    if shared:
        # Shared orthonormal input basis; coefficient matrix unique to each layer.
        # One joint CPU sketch; temporary concatenation is not deployment storage.
        z=torch.cat(weights,dim=0).float()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            u,s,v=torch.svd_lowrank(z,q=min(rank+8,*z.shape),niter=3)
        a=v[:,:rank].T.contiguous().bfloat16()
        del z,u,s,v
        # Fit against the actually stored basis, not the unrounded SVD.
        af=a.float()
        gram=af@af.T
        bs=[torch.linalg.solve(gram,(w.float()@af.T).T).T.contiguous().bfloat16() for w in weights]
        return [a],bs
    aa=[]; bb=[]
    for w in weights:
        p,s=codec.encode(w,"svd",rank=rank)
        a,b=codec.factors(p,s,torch.bfloat16)
        aa.append(a.contiguous()); bb.append(b.contiguous())
    return aa,bb


def residual(w,a,b):
    spec={"codec":"q5-had","shape":list(w.shape),"group":64}
    af,bf=a.float(),b.float()
    p=geo._encode_tiles(lambda i,j:w[i:j].float()-bf[i:j]@af,spec,codec._pack)
    codec.validate(p,spec)
    return p,spec


def actions(w,x,g):
    return F.linear(x.float(),w.float()),g.float()@w.float()


def fit_target(w,p,s,capture):
    """Compute targets for the frozen low-rank path, with decoded Q5 fixed."""
    x,g=capture["x"].float(),capture["dy"].float()
    y=torch.empty((len(x),w.shape[0]),dtype=torch.float32)
    dx=torch.zeros((len(g),w.shape[1]),dtype=torch.float32)
    ry=torch.empty_like(y); rg=torch.zeros_like(dx)
    for i,j,r in decoded_tiles(p,s):
        wt=w[i:j].float()
        y[:,i:j]=x@(wt-r).T
        dx+=g[:,i:j]@(wt-r)
        ry[:,i:j]=x@wt.T
        rg+=g[:,i:j]@wt
    return x,g,y,dx,float(ry.square().sum()),float(rg.square().sum())


def objective(a,b,target):
    x,g,y,dx,yn,gn=target
    terms=[]
    if yn>0: terms.append(((x@a.T)@b.T-y).square().sum()/yn)
    if gn>0: terms.append(((g@b)@a-dx).square().sum()/gn)
    return torch.stack(terms).mean() if terms else None


def fit_factors(aa,bb,targets,steps=40):
    """Fit BF16-deployable factors against real X/G, Q5 residual held fixed.

    Float32 optimization with straight-through BF16 factor rounding. All
    targets are cached low-row actions, never a dense trainable base.
    Equal weight per usable projection/capture; forward/back axes normalized
    separately. Fit only, not development or confirmation data.
    """
    a=[t.float().clone().requires_grad_(True) for t in aa]
    b=[t.float().clone().requires_grad_(True) for t in bb]
    params=a+b
    opt=torch.optim.Adam([{"params":[t],"lr":max(1e-6,.01*float(t.detach().square().mean().sqrt()))} for t in params])
    def rounded(t):
        return t+(t.bfloat16().float()-t).detach()
    def loss():
        terms=[]
        for i,ts in enumerate(targets):
            ai=rounded(a[0 if len(a)==1 else i]); bi=rounded(b[i])
            for target in ts:
                v=objective(ai,bi,target)
                if v is not None: terms.append(v)
        if not terms: raise ValueError("no usable fit signal")
        return torch.stack(terms).mean()
    with torch.no_grad():
        initial=float(loss())
    best=initial
    best_a=[t.clone() for t in aa]; best_b=[t.clone() for t in bb]
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        l=loss()
        if not torch.isfinite(l): raise ValueError("nonfinite correction objective")
        l.backward()
        opt.step()
        with torch.no_grad():
            value=float(loss())
            if math.isfinite(value) and value<best:
                best=value
                best_a=[t.detach().bfloat16().clone() for t in a]
                best_b=[t.detach().bfloat16().clone() for t in b]
    return best_a,best_b,{"initial_fit_objective":initial,"best_fit_objective":best,"steps":steps}


def weight_error(w,p,s,a=None,b=None):
    num=den=0.
    af=a.float() if a is not None else None
    for i,j,r in decoded_tiles(p,s):
        wt=w[i:j].float()
        if af is not None: r=r+b[i:j].float()@af
        num+=float((wt-r).square().sum(dtype=torch.float64))
        den+=float(wt.square().sum(dtype=torch.float64))
    return math.sqrt(num/den) if den else None


def operator(p,s,x,g,a=None,b=None):
    """Actual packed runtime forward/backward, plus frozen BF16 factors."""
    x=x.detach().bfloat16().requires_grad_(True)
    y=codec.linear(x,p,s)
    if a is not None:
        y=y+F.linear(F.linear(x,a),b)
    dx,=torch.autograd.grad(y,x,g.bfloat16())
    return y.detach(),dx.detach()


def relative(actual,reference):
    a,b=actual.float(),reference.float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("nonfinite operator result")
    denom=float(b.norm())
    return float((a-b).norm())/denom if denom else None


def score(w,p,s,capture,int8,a=None,b=None):
    x,g=capture["x"].bfloat16(),capture["dy"].bfloat16()
    y,dx=operator(p,s,x,g,a,b)
    raw_y=F.linear(x,w); raw_dx=g@w
    x1=x.detach().requires_grad_(True)
    ref=int8(x1); rd,=torch.autograd.grad(ref,x1,g)
    out={}
    ni=capture["sampled_image"]
    for name,sl in (("all",slice(None)),("image",slice(0,ni)),("text",slice(ni,None))):
        if len(x[sl])==0: continue
        out[name]={"raw_bf16_forward":relative(y[sl],raw_y[sl]),
                   "raw_bf16_input_gradient":relative(dx[sl],raw_dx[sl]),
                   "native_int8_forward":relative(y[sl],ref.detach()[sl]),
                   "native_int8_input_gradient":relative(dx[sl],rd[sl])}
    return out


def aggregate(rows):
    """Mean squared relative errors, equal usable capture/projection/axis weight."""
    values=[]
    for row in rows:
        z=row["errors"]["all"]
        valid=[z[k]**2 for k in ("raw_bf16_forward","raw_bf16_input_gradient") if z[k] is not None]
        if valid: values.append(sum(valid)/len(valid))
    return sum(values)/len(values) if values else None
