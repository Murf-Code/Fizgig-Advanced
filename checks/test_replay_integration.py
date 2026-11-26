import random,copy
import numpy as np,pytest,torch
from fizgig.training.replay_forks import Pair,capture,match
from fizgig.training.muon import BatchedMuon,Muon
from fizgig.training.replay_evaluator import Case,compare_candidates
from fizgig.training.replay_diagnostics import digest
from fizgig.training.replay_probe import Probe,evaluate_fork
from fizgig.training.replay_checkpoint import BoundaryCheckpoint

torch.set_num_threads(1)
torch.backends.mkldnn.enabled=True

def setup(seed=11,n=28):
 g=torch.Generator().manual_seed(seed);pairs=[]
 for b in range(n):
  a=torch.nn.Parameter(torch.randn(2,5,generator=g)*.1);z=torch.nn.Parameter(torch.randn(4,2,generator=g)*.1)
  a.grad=torch.randn(a.shape,generator=g)*.01;z.grad=torch.randn(z.shape,generator=g)*.01;pairs.append(Pair(str(b),b,a,z,1.))
 opt=BatchedMuon([p for pair in pairs for p in (pair.A,pair.B)],lr=.00045,weight_decay=.01,adjust_lr_fn='match_rms_adamw');return pairs,opt,g

def fixtures():
 pairs,opt,g=setup();extra={'buffer':torch.zeros(1),'ema':torch.zeros(1),'scheduler':{'step':0},'selector':{'age':0}}
 def predict(x):return sum((p.B@p.A)@x for p in pairs)
 probes=[]
 with torch.no_grad():
  for i,d in enumerate(('target','near','far')):
   x=torch.randn(5,generator=g);p=predict(x).clone();probes.append(Probe(Case(d,d,'a',.5,1,digest(x)),x,p,torch.zeros_like(p),p-.01 if d=='target' else None))
 alloc={'uniform':{i:1. for i in range(28)},'static':{i:.5 if i==26 else 1. for i in range(28)},'adaptive':{i:.5 if i==13 else 1. for i in range(28)}}
 return pairs,opt,g,extra,probes,alloc,predict

def run(f,alloc,probes,predict,extra,order=None):return evaluate_fork(f,alloc,probes,predict,lambda:extra,minimum=1e-12,atol=1e-12,rtol=.001,max_initial_dip=.01,order=order)

def test_end_to_end_order_and_repeated_allocations():
 pairs,opt,g,extra,probes,alloc,predict=fixtures();alloc['repeat']=dict(alloc['static']);before=digest([(p.A,p.B) for p in pairs]);rng=g.get_state().clone()
 with capture(pairs,opt,generators=(g,)) as f:
  a=run(f,alloc,probes,predict,extra);b=run(f,alloc,probes,predict,extra,list(reversed(alloc)))
  assert a['status']==b['status']=='evaluated';assert a['arms']==b['arms'];assert a['arms']['static']==a['arms']['repeat'];assert a['arms']['static']['metrics']!=a['arms']['uniform']['metrics'];assert a['arms']['static']['depth']['total_squared']>0
 assert digest([(p.A,p.B) for p in pairs])==before;assert not opt.state;assert torch.equal(g.get_state(),rng)

@pytest.mark.parametrize('fault',['optimizer','gradient','weight','input','rng','generator','buffer','scheduler','ema','selector','reference'])
def test_faults_rejected(fault):
 pairs,opt,g,extra,probes,alloc,predict=fixtures()
 def bad(x):
  p=pairs[0].A
  if fault=='optimizer':opt.state[p]['momentum_buffer'].add_(1)
  elif fault=='gradient':p.grad.add_(1)
  elif fault=='weight':p.add_(1)
  elif fault=='input':x.add_(1)
  elif fault=='rng':random.random()
  elif fault=='generator':torch.rand(1,generator=g)
  elif fault in ('buffer','ema'):extra[fault].add_(1)
  elif fault in ('scheduler','selector'):extra[fault]['new']=1
  elif fault=='reference':probes[0].parent.add_(1)
  return predict(x)
 with pytest.raises(RuntimeError,match='mutat'):
  with capture(pairs,opt,generators=(g,)) as f:run(f,alloc,probes,bad,extra)
 assert not opt.state

def test_incorrect_endpoint_rejected(monkeypatch):
 pairs,opt,g,extra,probes,alloc,predict=fixtures()
 with capture(pairs,opt,generators=(g,)) as f:
  original=f.achieved;calls=0
  def wrong(m):
   nonlocal calls
   calls+=1
   v,s=original(m)
   if calls>3:return v*2,{k:4*x for k,x in s.items()}
   return v,s
  monkeypatch.setattr(f,'achieved',wrong)
  with pytest.raises(RuntimeError,match='endpoint'):run(f,alloc,probes,predict,extra)

@pytest.mark.parametrize('cls',[BatchedMuon,Muon])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_compact_equal(cls,dtype):
 g=torch.Generator().manual_seed(12);a=torch.nn.Parameter(torch.randn(2,7,generator=g).to(dtype));b=torch.nn.Parameter(torch.randn(5,2,generator=g).to(dtype));p=Pair('p',0,a,b,1.);opt=cls([a,b],lr=.00045,adjust_lr_fn='match_rms_adamw');a.grad=torch.randn_like(a);b.grad=torch.randn_like(b);opt.step()
 states=[]
 for compact in (False,True):
  with capture([p],opt,compact=compact) as f:
   r=f.achieved({0:.57});states.append((r,digest((a,b,opt.state_dict()))))
 assert states[0]==states[1]

class Counter:
 def __init__(self):self.step=0
 def state_dict(self):return {'step':self.step}
 def load_state_dict(self,s):self.step=s['step']

def training(seed):
 torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);g=torch.Generator().manual_seed(seed+1);a=torch.nn.Parameter(torch.randn(4,8)*.1);b=torch.nn.Parameter(torch.randn(6,4)*.1);opt=BatchedMuon([a,b],lr=.00045,adjust_lr_fn='match_rms_adamw');sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,20);counter=Counter();ema=torch.zeros(6,8)
 cp=BoundaryCheckpoint({'a':a,'b':b},opt,buffers={'ema':ema},components={'scheduler':sched,'selector':counter},generators={'noise':g},metadata={'seed':seed,'dataset':'synthetic-v1'})
 def step():
  target=torch.randn(6,8,generator=g)+torch.randn(6,8)*.01+np.random.randn()*.01+random.random()*.01
  (b@a-target).square().mean().backward();opt.step();opt.zero_grad(set_to_none=True);sched.step();counter.step+=1;ema.lerp_((b@a).detach(),.02);return digest(cp.snapshot())
 return cp,step

@pytest.mark.parametrize('seed',range(20,28))
def test_disk_checkpoint_continuation(seed,tmp_path):
 cp,step=training(seed);[step() for _ in range(4)];path=tmp_path/'state.pt';cp.save(path);expected=[step() for _ in range(8)];other,run=training(seed);other.load(path);assert [run() for _ in range(8)]==expected

@pytest.mark.parametrize('fault',['metadata','groups','components','generators','params','optimizer_class'])
def test_checkpoint_rejects_mismatch(fault):
 cp,step=training(11);step();s=cp.snapshot();original=digest(cp.snapshot());s[fault]=None
 with pytest.raises((ValueError,TypeError)):cp.restore(s)
 assert digest(cp.snapshot())==original


def test_factor_endpoint_identity(monkeypatch):
 pairs,opt,g,extra,probes,alloc,predict=fixtures()
 with capture(pairs,opt,generators=(g,)) as f:
  original=f.achieved;calls=0
  def wrong(m):
   nonlocal calls
   calls+=1
   v,s=original(m)
   if calls>3:
    with torch.no_grad():pairs[0].A.add_(1)
   return v,s
  monkeypatch.setattr(f,'achieved',wrong)
  with pytest.raises(RuntimeError,match='factor endpoint'):run(f,alloc,probes,predict,extra)


def test_checkpoint_live_group_order():
 cp,step=training(12);step();s=cp.snapshot();cp.optimizer.param_groups[0]['params'].reverse()
 with pytest.raises(ValueError,match='parameter order'):cp.restore(s)
