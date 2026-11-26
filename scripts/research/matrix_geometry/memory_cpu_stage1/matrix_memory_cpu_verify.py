"""Independent CPU tests. Pass --repo to import exact research branch source.
Without --repo, tests use a transcription of the reviewed source's pack/unpack
and checkpoint helper, plus an independent momentum recurrence. No GPU claims.
"""
import argparse, copy, json, sys, pathlib, tempfile
import torch
from torch.utils.checkpoint import checkpoint

def pack(x,bits=8,block=256):
    if bits not in (4,8) or block<=0 or block%2: raise ValueError('bits/block')
    f=x.float().reshape(-1); n=f.numel(); pad=(-n)%block
    if n==0: raise ValueError('empty')
    if pad: f=torch.nn.functional.pad(f,(0,pad))
    f=f.reshape(-1,block); qmax=(1<<(bits-1))-1
    s=f.abs().amax(1)/qmax
    s=torch.where(s==0,torch.ones_like(s),s.clamp_min(torch.finfo(torch.float32).tiny))
    q=(f/s[:,None]).round().clamp(-qmax,qmax).to(torch.int8).flatten()
    if bits==4:
        q=(q.to(torch.int16)+8).to(torch.uint8)
        q=q[::2] | (q[1::2]<<4)
    return q,s

def unpack(q,s,shape,bits=8,block=256):
    if bits==4:
        q=torch.stack(((q&15).to(torch.int16)-8,(q>>4).to(torch.int16)-8),dim=1).flatten()
    n=1
    for d in shape:n*=d
    return (q.reshape(-1,block).float()*s[:,None]).flatten()[:n].reshape(shape)

def checkpoint_activation_only(block,activation,*args,offload=False,stats=None):
    if not offload:return checkpoint(block,activation,*args,use_reentrant=False,preserve_rng_state=True)
    target=id(activation)
    def pack_hook(x):
        if id(x)==target:
            if stats is not None:
                stats['count']=stats.get('count',0)+1
                stats['bytes']=stats.get('bytes',0)+x.numel()*x.element_size()
            return (x.device,x.detach().to('cpu',copy=True))
        return (None,x.detach())
    def unpack_hook(p):
        dev,x=p
        return x if dev is None else x.to(dev)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook,unpack_hook):
        return checkpoint(block,activation,*args,use_reentrant=False,preserve_rng_state=True)

def rel(x,y):return float((x-y).norm()/y.norm().clamp_min(1e-30))
def cos(x,y):return float(torch.nn.functional.cosine_similarity(x.flatten(),y.flatten(),dim=0))

def run():
    torch.set_num_threads(1); torch.manual_seed(741)
    report={'torch':torch.__version__,'device':'CPU','mode':'transcribed reference'}
    if args.repo:
        sys.path.insert(0,str(pathlib.Path(args.repo)/'scripts/research/matrix_geometry'))
        import state_compression as sc, checkpoint_offload as co
        globals().update(pack=sc.pack,unpack=sc.unpack,checkpoint_activation_only=co.checkpoint_activation_only)
        report['mode']='repository imports (caller responsible for checkout SHA)'
    # 4-block chain with trainable conditioning, frozen base, dropout and backward hooks
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.base=torch.nn.Linear(32,32,bias=False);self.base.requires_grad_(False)
            self.a=torch.nn.Linear(32,8,bias=False);self.b=torch.nn.Linear(8,32,bias=False)
        def forward(self,x,c):return x+torch.nn.functional.dropout(torch.tanh(self.base(x)+self.b(self.a(x))+c),.2,self.training)
    blocks=torch.nn.ModuleList([Block() for _ in range(4)])
    def chain(offload):
        net=copy.deepcopy(blocks); order=[]; stats={}
        handles=[b.register_full_backward_hook(lambda m,gi,go,i=i:order.append(i)) for i,b in enumerate(net)]
        torch.manual_seed(777); x=torch.randn(2,17,32,requires_grad=True);c=torch.randn(2,1,32,requires_grad=True)
        y=x
        for b in net:y=checkpoint_activation_only(b,y,c,offload=offload,stats=stats)
        y.square().mean().backward(); rng=torch.rand(5)
        for h in handles:h.remove()
        return [y.detach(),x.grad,c.grad,*[p.grad for p in net.parameters() if p.requires_grad],rng],order,stats
    a,ao,_=chain(False);b,bo,st=chain(True)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert ao==bo==[3,2,1,0] and st['count']==4
    report['activation']={'bitwise_equal_output_grad_rng':True,'hook_order':bo,**st}
    # Quantization error, padded tail, zero, outlier and nonfinite behavior
    x=torch.randn(64,257); q,s=pack(x);y=unpack(q,s,x.shape)
    bound=s.repeat_interleave(256)[:x.numel()].reshape_as(x)/2
    assert ((x-y).abs()<=bound+1e-6).all(); assert q.dtype==torch.int8
    z=torch.zeros(257);qz,sz=pack(z);assert torch.equal(z,unpack(qz,sz,z.shape))
    out=torch.ones(256)*.1;out[0]=100.;qo,so=pack(out);yo=unpack(qo,so,out.shape)
    bad=torch.tensor([float('inf'),1.]);qb,sb=pack(bad);yb=unpack(qb,sb,bad.shape)
    report['codec']={'relative_error':rel(y,x),'mean_signed_error':float((y-x).mean()),'outlier_small_entries_zero':int((yo[1:]==0).sum()),'nonfinite_silently_propagates':not bool(torch.isfinite(yb).all()),'tail_and_zero_pass':True}
    # 1000 steps: common exogenous gradients, not closed-loop model training
    for mom in [.9,.95,.99]:
        for scenario in ['iid','correlated','outlier','subbin']:
            gen=torch.Generator().manual_seed(45); m=torch.zeros(64,128);qm,sm=pack(m); h=torch.randn(m.shape,generator=gen); maxerr=0.; mincos=1.
            for i in range(1000):
                g=torch.randn(m.shape,generator=gen)
                if scenario=='correlated':g=.95*h+.05*g
                if scenario=='outlier':g.flatten()[::256]=100.
                if scenario=='subbin':g.fill_(.1);g.flatten()[::256]=100.
                m.lerp_(g,1-mom);u=unpack(qm,sm,m.shape);u.lerp_(g,1-mom);qm,sm=pack(u);v=unpack(qm,sm,m.shape)
                maxerr=max(maxerr,rel(v,m));mincos=min(mincos,cos(v,m))
            report.setdefault('momentum',[]).append({'beta':mom,'scenario':scenario,'final_rel':rel(v,m),'max_rel':maxerr,'min_cos':mincos,'small_coordinate_rel':rel(v.flatten()[1::256],m.flatten()[1::256])})
    # PyTorch default optimizer loading can cast packed state into FP32.
    p=torch.nn.Parameter(torch.zeros(64,128)); opt=torch.optim.SGD([p],lr=.01);q,s=pack(torch.randn_like(p));opt.state[p]={'q':q,'scale':s}; sd=copy.deepcopy(opt.state_dict());opt.load_state_dict(sd)
    report['default_load_q_dtype']=str(opt.state[p]['q'].dtype)
    if args.repo:
        def make():return torch.nn.Parameter(torch.randn(64,128))
        p=make();o=sc.QuantMuon([p]);gen=torch.Generator().manual_seed(88)
        for _ in range(5):p.grad=torch.randn(p.shape,generator=gen);o.step()
        p2=torch.nn.Parameter(p.detach().clone());o2=sc.QuantMuon([p2]);o2.load_state_dict(copy.deepcopy(o.state_dict()))
        for _ in range(5):
            p.grad=torch.randn(p.shape,generator=gen);p2.grad=p.grad.clone();o.step();o2.step()
        torch.testing.assert_close(p,p2,rtol=0,atol=0);assert torch.equal(o.state[p]['q'],o2.state[p2]['q'])
        report['actual_quantmuon_resume']='bitwise equal for 5 continued CPU steps'
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo');args=parser.parse_args();run()
