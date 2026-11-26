"""Independent CPU tests for compressed Muon-style LoRA updates. No trainer integration."""
import argparse, json, math, time, statistics, traceback
from pathlib import Path
import torch
from memory_core import pack8, unpack8, unique_bytes, scratch_plan, phase_plan, Phase

METHODS=('int8_256','int8_64','split64','split256','hybrid64_5pct')

def encode(x, method):
    if method=='fp32': return (x.clone(),)
    if method in ('bf16','fp16'): return (x.to(torch.bfloat16 if method=='bf16' else torch.float16),)
    block=256 if method.endswith('256') else 64
    flat=x.reshape(-1); n=flat.numel(); pad=(-n)%block
    f=torch.nn.functional.pad(flat,(0,pad)).view(-1,block) if pad else flat.view(-1,block)
    if method.startswith('split'):
        # One exact maximum per block; use uint8 index even for size 256 (0..255).
        idx=f.abs().argmax(1,keepdim=True); val=f.gather(1,idx).flatten(); rest=f.clone().scatter_(1,idx,0.)
        q,s=pack8(rest,block)
        return q,s,idx.flatten().to(torch.uint8),val
    q,s=pack8(f,block)
    if method.startswith('hybrid'):
        # At most 5% blocks in FP32. Choose by worst relative error on weak nonzero values.
        # Prioritize average normalized coordinate error, capped to avoid division by tiny values.
        recon=unpack8(q,s,f.shape,block); floor=f.abs().amax(1,keepdim=True)*1e-6+1e-20
        score=((recon-f).abs()/torch.maximum(f.abs(),floor)).clamp(max=1).mean(1)
        k=int(.05*f.shape[0]); idx=torch.argsort(score,descending=True,stable=True)[:k]; vals=f[idx].clone()
        # Reference retains quantized bytes for fallback blocks for simple indexed restore.
        return q,s,idx.to(torch.int32),vals
    return q,s

def decode(state,shape,method):
    if method in ('fp32','bf16','fp16'): return state[0].float().clone()
    b=256 if method.endswith('256') else 64
    q,s=state[:2]; x=unpack8(q,s,(q.numel()//b,b),b)
    if method.startswith('split'): x.scatter_(1,state[2].long()[:,None],state[3][:,None])
    if method.startswith('hybrid'): x[state[2].long()]=state[3]
    return x.flatten()[:math.prod(shape)].reshape(shape)

def ns(x):
    # Same coefficients, BF16 compute, five iterations as pinned research MuonME.
    a,b,c=3.4445,-4.775,2.0315
    x=x.bfloat16(); tall=x.shape[-2]>x.shape[-1]
    if tall:x=x.T
    x=x/x.norm().clamp(min=1e-7)
    for _ in range(5):
        gram=x@x.T; upd=torch.addmm(gram,gram,gram,beta=b,alpha=c); x=torch.addmm(x,upd,x,beta=a)
    return (x.T if tall else x).float()

def step_direction(state,g,method,beta=.95):
    m=decode(state,g.shape,method) if state is not None else torch.zeros_like(g)
    m.lerp_(g,1-beta); u=ns(torch.lerp(g,m,beta)); return encode(m,method),u

def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-20))
def cos(a,b):return float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0))
def effective_delta(a,b,da,db):return b@da+db@a+db@da

def drift():
    out=[]
    for scenario in ('iid','correlated','single_outlier','two_outliers','subbin'):
        for method in METHODS:
            rng=torch.Generator().manual_seed(45); shape=(64,128); m=torch.zeros(shape); st=None; h=torch.randn(shape,generator=rng); err=[]
            for i in range(1000):
                g=torch.randn(shape,generator=rng)
                if scenario=='correlated':g=.95*h+.05*g
                if scenario in ('single_outlier','two_outliers','subbin'):
                    if scenario=='subbin':g.fill_(.1)
                    g.flatten()[::256]=100
                    if scenario=='two_outliers':g.flatten()[1::256]=-80
                m.lerp_(g,.05); v=decode(st,shape,method) if st is not None else torch.zeros_like(g);v.lerp_(g,.05);st=encode(v,method);v=decode(st,shape,method)
                if i>=900:err.append(rel(v,m))
            weak=(m.abs()>1e-9)&(m.abs()<1)
            out.append(dict(scenario=scenario,method=method,tail_error=statistics.mean(err),weak_error=rel(v[weak],m[weak]),weak_zero=float((v[weak]==0).float().mean()),bytes=unique_bytes(st)))
        print('Momentum recurrence:',scenario,'complete',flush=True)
    return out

def geometry():
    rows=[]
    for rank in (16,64):
      for scenario in ('iid','correlated','outlier'):
        rng=torch.Generator().manual_seed(337+rank); a=torch.randn(rank,128,generator=rng)*.05;b=torch.randn(96,rank,generator=rng)*.05
        # Fixed A/B; externally supplied dense weight gradients. Outlier gradients are
        # structured but synthetic, not drawn from any Krea2 factor trace.
        h=torch.randn(96,128,generator=rng); states={k:[None,None] for k in ('fp32',)+METHODS}; logs={k:[] for k in METHODS}
        for i in range(160):
            d=torch.randn(96,128,generator=rng)
            if scenario=='correlated':d=.95*h+.05*d
            if scenario=='outlier':d[0,0]=200;d[1,1]=-150
            ga=b.T@d;gb=d@a.T; updates={}
            for k in states:
                sa,ua=step_direction(states[k][0],ga,k);sb,ub=step_direction(states[k][1],gb,k);states[k]=[sa,sb]
                da=-.002*ua;db=-.002*ub;updates[k]=(effective_delta(a,b,da,db),ua,ub)
            if i>=120:
                ref,ra,rb=updates['fp32']; u,s,vh=torch.linalg.svd(ref,full_matrices=False)
                # Only non-negligible singular directions; last quartile of those.
                good=int((s>s[0]*1e-5).sum()); start=max(0,good-max(1,good//4)); U=u[:,start:good]; V=vh[start:good].T
                weak=U.T@ref@V
                for k in METHODS:
                    y,ya,yb=updates[k]; logs[k].append((rel(y,ref),cos(y,ref),rel(U.T@y@V,weak),max(rel(ya,ra),rel(yb,rb))))
        for k in METHODS:
            l=logs[k];rows.append(dict(rank=rank,scenario=scenario,method=k,ba_rel=statistics.mean(x[0] for x in l),ba_cos=statistics.mean(x[1] for x in l),weak_projected_rel=statistics.mean(x[2] for x in l),factor_rel=statistics.mean(x[3] for x in l),worst_ba_rel=max(x[0] for x in l)))
        print('Geometry:',rank,scenario,'complete',flush=True)
    return rows

def convergence(steps,methods=('fp32',)+METHODS):
    rows=[]
    for rank in (16,64):
      for seed in (19,37,61):
        rng=torch.Generator().manual_seed(seed); a0=torch.randn(rank,96,generator=rng)*.05;b0=torch.zeros(96,rank); target=torch.randn(96,rank,generator=rng)@torch.randn(rank,96,generator=rng)*(.03/math.sqrt(rank))
        for method in methods:
            a=a0.clone();b=b0.clone();sa=sb=None; curve=[]
            for i in range(steps):
                error=b@a-target; ga=b.T@error;gb=error@a.T
                sa,ua=step_direction(sa,ga,method);sb,ub=step_direction(sb,gb,method)
                # fixed decaying LR, identical across arms; no clipping or WD.
                lr=.003*(.1+.9*(1-i/steps)); a.add_(ua,alpha=-lr);b.add_(ub,alpha=-lr)
                if i in (0,steps//4,steps//2,3*steps//4,steps-1):curve.append([i+1,float((b@a-target).square().mean())])
            loss=float((b@a-target).square().mean());assert math.isfinite(loss)
            rows.append(dict(rank=rank,seed=seed,method=method,loss=loss,curve=curve))
        print('Closed-loop:',rank,seed,'complete',flush=True)
    return rows

class PressurePolicy:
    """Driver-free-only allowance; inputs must describe additional commitments.
    No allocator reserve credit, no live changes to optimizer precision or rank."""
    def __init__(self,grow_after=3):self.current=None;self.n=0;self.candidate=None;self.grow_after=grow_after
    def choose(self,free,pending,commitments,margin,ceiling,unit):
        cap=min(ceiling,max(0,free-pending-commitments-margin))
        if cap<unit:
            self.current=cap;self.n=0;self.candidate=None;return 0
        if self.current is None or cap<=self.current:
            self.current=cap;self.n=0;self.candidate=None
        else:
            self.n+=1;self.candidate=cap if self.candidate is None else min(cap,self.candidate)
            if self.n>=self.grow_after:self.current=self.candidate;self.n=0;self.candidate=None
        # Refuse when hysteresis remains below smallest unit; never bypass safety.
        return self.current//unit

class PeakLedger:
    """Mockable exclusive reset coordinator, not integrated CUDA telemetry."""
    def __init__(self,backend):self.b=backend;self.global_peak=(0,0);self.phase=None;self.records=[]
    def capture(self):
        p=self.b.peaks();self.global_peak=tuple(max(a,b) for a,b in zip(p,self.global_peak));return p
    def begin(self,name):
        if self.phase is not None:raise RuntimeError('nested phase')
        self.capture();self.b.reset();self.phase=name
    def end(self):
        if self.phase is None:raise RuntimeError('no active phase')
        p=self.capture();self.records.append((self.phase,p));self.phase=None;return p
    def external_reset(self):
        if self.phase is not None:raise RuntimeError('reset inside measured phase')
        self.capture();self.b.reset()

class MockAllocator:
    def __init__(self):self.current=(0,0);self.high=(0,0)
    def set(self,a,r):assert r>=a;self.current=(a,r);self.high=tuple(max(x,y) for x,y in zip(self.high,self.current))
    def peaks(self):return self.high
    def reset(self):self.high=self.current

def budget_test():
    p=PressurePolicy();free=[400,480,480,480,200,160,500,500,500,80,500,500,500]; observed=[p.choose(f,40,20,40,512,50) for f in free]
    assert observed==[6,6,6,7,2,1,1,1,8,0,0,0,8],observed
    rng=torch.Generator().manual_seed(9);p=PressurePolicy(); refusals=0; required_refusals=0
    # Include stable plateaus and shocks; independent uniform large obligations
    # otherwise produce almost exclusively refusals and weak coverage.
    for i in range(10000):
        if i%10==0 or i%10==9:
            free=int(torch.randint(1,1500,(),generator=rng));pending,commitments,margin=map(int,torch.randint(0,200,(3,),generator=rng));ceiling=int(torch.randint(64,1025,(),generator=rng));unit=int(torch.randint(1,129,(),generator=rng))
        n=p.choose(free,pending,commitments,margin,ceiling,unit); cap=min(ceiling,max(0,free-pending-commitments-margin))
        assert n>=0 and n*unit<=cap;refusals+=n==0;required_refusals+=cap<unit
        if cap<unit:assert n==0
        if i%10==8 and cap>=unit:assert n>0
    assert 0<refusals<10000
    b=MockAllocator();ledger=PeakLedger(b);b.set(8,9)
    for name,peak in [('forward',(10,11)),('backward',(11,12)),('optimizer',(9,10)),('preview',(12,13))]:
        ledger.begin(name);b.set(*peak);b.set(7,8);assert ledger.end()==peak
    ledger.external_reset();assert ledger.global_peak==(12,13)
    ledger.begin('test')
    try:ledger.external_reset()
    except RuntimeError:pass
    else:raise AssertionError('nested reset allowed')
    ledger.end()
    return dict(dynamic_chunks=observed,random_cases=10000,refusals=refusals,accepted=10000-refusals,required_refusals=required_refusals,phase_peaks=ledger.records,global_peak=ledger.global_peak)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=600);parser.add_argument('--output',default='cpu_results_v2.json');args=parser.parse_args();assert args.steps>=20
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);t=time.monotonic();print('CPU research: PyTorch',torch.__version__,flush=True)
    # codec storage and edge cases
    for method in METHODS:
        for shape in ((3,17),(16,128),(64,96)):
            x=torch.zeros(shape);x.flatten()[0]=100;x.flatten()[-1]=.1; st=encode(x,method);y=decode(st,shape,method);assert y.shape==x.shape and torch.isfinite(y).all() and y.flatten()[0]==100
    report=dict(torch=str(torch.__version__),device='cpu',steps=args.steps,drift=drift(),geometry=geometry(),convergence=convergence(args.steps),budget=budget_test(),seconds=time.monotonic()-t)
    Path(args.output).write_text(json.dumps(report,indent=2));print('PASS: 25 recurrence arms, 30 geometry comparisons, 36 convergence runs, 10,000 budget stress cases; report:',args.output,flush=True)
if __name__=='__main__': main()
