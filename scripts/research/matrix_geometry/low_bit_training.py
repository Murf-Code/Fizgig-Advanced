"""CPU direct packed-factor training proxy. Not a trainer or GPU kernel."""
import json, math, sys
import torch

torch.set_num_threads(1)

@torch.no_grad()
def pack(x, bits, stochastic, rng, block=64):
    flat=x.float().reshape(-1)
    n=flat.numel(); padded=torch.nn.functional.pad(flat,(0,(-n)%block)).reshape(-1,block)
    bound=(1<<(bits-1))-1
    scale=padded.abs().amax(1)/bound
    scale=torch.where(scale==0,torch.ones_like(scale),scale.clamp_min(torch.finfo(torch.float32).tiny))
    z=(padded/scale[:,None]).clamp(-bound,bound)
    q=(z.floor()+(torch.rand(z.shape,generator=rng)<(z-z.floor()))) if stochastic else z.round()
    q=q.clamp(-bound,bound).to(torch.int8).flatten()
    if bits==4:
        u=(q.to(torch.int16)+8).to(torch.uint8); q=u[::2] | (u[1::2]<<4)
    return q,scale,tuple(x.shape),bits,block

def unpack(p):
    q,s,shape,bits,block=p
    if bits==4: q=torch.stack(((q&15).to(torch.int16)-8,(q>>4).to(torch.int16)-8),1).flatten()
    return (q.reshape(-1,block).float()*s[:,None]).flatten()[:math.prod(shape)].reshape(shape)

def size(p):
    return sum(x.numel()*x.element_size() for x in p[:2])

CASES=(('fp32',0,False,False),('int8-rne',8,False,False),('int8-sr',8,True,False),('int4-rne',4,False,False),('int4-sr',4,True,False),('int8-qat',8,False,True),('int4-qat',4,False,True))

@torch.no_grad()
def run(seed, regime, case, steps=240):
    name,bits,sr,master=case
    data=torch.Generator().manual_seed(seed); rounding=torch.Generator().manual_seed(seed+100000)
    r,di,do=8,96,64
    rand=lambda *s:torch.randn(*s,generator=data)
    ta=rand(r,di)/math.sqrt(di); tb=rand(do,r)/math.sqrt(r); target=tb@ta
    if regime=='growth': a=rand(r,di)/math.sqrt(di); b=torch.zeros(do,r); lr=.02
    else: a=ta+.02*rand(r,di); b=tb+.02*rand(do,r); lr=.002
    # B is stored transposed so quantization groups follow rank vectors.
    initial=[a,b.T.contiguous()]; encode=lambda x:pack(x,bits,sr,rounding)
    stored=[encode(x) for x in initial] if bits and not master else initial
    m=[torch.zeros_like(x) for x in initial]; v=[torch.zeros_like(x) for x in initial]
    del a,b,initial
    losses=[]; zero_codes=[]
    for k in range(1,steps+1):
        if not bits: actual=stored
        elif master: actual=[unpack(encode(x)) for x in stored]
        else: actual=[unpack(p) for p in stored]
        a,bt=actual; b=bt.T
        # Identical minibatches per seed across cases; independent rounding RNG.
        x=rand(32,di); y=x@target.T+.02*rand(32,do)
        h=x@a.T; e=(h@bt-y)*(2/(32*do))
        grads=[(e@b).T@x,h.T@e]
        rate=lr*(.1+.9*(1+math.cos(math.pi*(k-1)/(steps-1)))/2)
        for j,g in enumerate(grads):
            m[j].lerp_(g,.1); v[j].mul_(.999).addcmul_(g,g,value=.001)
            update=(m[j]/(1-.9**k))/(v[j].div(1-.999**k).sqrt()+1e-8)
            source=stored[j] if master else actual[j]
            proposed=source-rate*update
            if bits and not master:
                p=encode(proposed); zero_codes.append((p[0]==stored[j][0]).float().mean().item()); stored[j]=p
            else: stored[j]=proposed
        if bits: ev=[unpack(encode(x)) for x in stored] if master else [unpack(p) for p in stored]
        else: ev=stored
        losses.append(((ev[1].T@ev[0]-target).square().sum()/target.square().sum()).item())
        del actual,a,b,bt,ev,source,proposed,update,grads,g
    bytes_=sum(size(p) for p in stored) if bits and not master else sum(p.numel()*p.element_size() for p in stored)
    state_bytes=sum(p.numel()*p.element_size() for p in m+v)
    return dict(case=name,seed=seed,regime=regime,loss10=losses[9],loss60=losses[59],last20=sum(losses[-20:])/20,final=losses[-1],factor_bytes=bytes_,adam_bytes=state_bytes,unchanged_code_bytes=sum(zero_codes)/len(zero_codes) if zero_codes else None)

def checks():
    rng=torch.Generator().manual_seed(123); n=0
    for bits in (4,8):
        for shape in ((3,17),(8,96),(2,1)):
            for sr in (False,True):
                x=torch.randn(shape,generator=rng); p=pack(x,bits,sr,rng); y=unpack(p)
                assert y.shape==x.shape and torch.isfinite(y).all(); n+=1
                assert size(p)==math.ceil(x.numel()/64)*(64*bits//8+4); n+=1
                assert torch.count_nonzero(unpack(pack(torch.zeros_like(x),bits,sr,rng)))==0; n+=1
                state=rng.get_state(); p1=pack(x,bits,sr,rng); rng.set_state(state); p2=pack(x,bits,sr,rng)
                assert torch.equal(p1[0],p2[0]) and torch.equal(p1[1],p2[1]); n+=1
                assert (y-x).abs().max()<=p[1].max()*1.001; n+=1
        # Fixed scale by a unit anchor: stochastic rounding expectation for a tiny update.
        bound=(1<<(bits-1))-1; x=torch.ones(20000,64); x[:,1:]=.13/bound
        y=unpack(pack(x,bits,True,rng)); mean=y[:,1:].mean().item()
        assert abs(mean-.13/bound)<.003/bound; n+=1
        assert torch.count_nonzero(unpack(pack(x,bits,False,rng))[:,1:])==0; n+=1
    return n

if __name__=='__main__':
    n=checks(); rows=[run(s,reg,c) for reg in ('growth','refinement') for c in CASES for s in range(6)]
    out={'checks':n,'steps':240,'rows':rows}; json.dump(out,open(sys.argv[1],'w'),indent=2)
    print('checks',n)
    for reg in ('growth','refinement'):
        for c in CASES:
            a=[x for x in rows if x['regime']==reg and x['case']==c[0]]
            print(reg,c[0],*[round(sum(x[k] for x in a)/len(a),6) for k in ('loss10','loss60','last20','final')],a[0]['factor_bytes'],a[0]['adam_bytes'])
