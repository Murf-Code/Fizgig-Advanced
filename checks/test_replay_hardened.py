import pytest,torch,copy
from dataclasses import replace
from test_replay_integration import fixtures,training
from fizgig.training.replay_forks import capture
from fizgig.training.replay_diagnostics import digest
from fizgig.training.replay_hardened import audited_evaluate,audit_manifest,audited_compare,named_summary,seal,restore_sealed
from fizgig.training.replay_disk import disk_capture

def eval_it(f,alloc,probes,predict,extra):return audited_evaluate(f,alloc,probes,predict,lambda x:torch.zeros(4),lambda:extra,minimum=1e-12,atol=1e-12,rtol=.001,max_initial_dip=.01)

def test_reference_provenance_and_order():
 pairs,opt,g,extra,probes,alloc,predict=fixtures()
 with capture(pairs,opt,generators=(g,)) as f:
  r=eval_it(f,alloc,probes,predict,extra);assert r['status']=='evaluated'
  with pytest.raises(ValueError,match='provenance'):eval_it(f,alloc,[replace(p,parent=p.parent+1) for p in probes],predict,extra)
  with pytest.raises(ValueError,match='provenance'):eval_it(f,alloc,[replace(p,base=p.base+1) for p in probes],predict,extra)
  a=r['arms']['uniform']['metrics'];b=copy.deepcopy(a);b['domains']['target']['mean']['target_gain']+=1
  with pytest.raises(ValueError,match='aggregate'):audited_compare(b,a)

def test_last_prior_ref_mutation():
 pairs,opt,g,extra,probes,alloc,predict=fixtures();calls=0
 def bad(x):
  nonlocal calls
  calls+=1
  if calls==len(probes)*(len(alloc)+1):probes[0].parent.add_(1)
  return predict(x)
 with capture(pairs,opt,generators=(g,)) as f:
  with pytest.raises(RuntimeError,match='panel mutation'):eval_it(f,alloc,probes,bad,extra)

def test_duplicate_ids():
 pairs,opt,g,extra,probes,alloc,predict=fixtures();c=probes[0].case
 with pytest.raises(ValueError,match='duplicate input'):audit_manifest([p.case for p in probes]+[replace(c,unit='alias')])
 with pytest.raises(ValueError,match='duplicate seed'):named_summary([('1',.1),('1',.2)])

def test_sealed_momentum():
 cp,step=training(62);[step() for _ in range(4)];envelope=seal(cp.snapshot());expected=step();restore_sealed(cp,envelope);assert step()==expected
 broken=copy.deepcopy(envelope);broken['snapshot']['optimizer']['state'].clear()
 with pytest.raises(ValueError,match='checksum'):restore_sealed(cp,broken)

def test_disk_capture_exact(tmp_path):
 pairs,opt,g,extra,probes,alloc,predict=fixtures();opt.step();before=digest(([(p.A,p.B) for p in pairs],opt.state_dict(),g.get_state()))
 results=[]
 for cap in [capture,disk_capture]:
  with cap(pairs,opt,generators=(g,)) as f:results.append(eval_it(f,alloc,[replace(p,parent=predict(p.inputs).detach()) for p in probes],predict,extra))
  assert before==digest(([(p.A,p.B) for p in pairs],opt.state_dict(),g.get_state()))
 assert results[0]==results[1]

def test_disk_capture_exception(tmp_path):
 pairs,opt,g,extra,probes,alloc,predict=fixtures();opt.step();before=digest(([(p.A,p.B) for p in pairs],opt.state_dict(),g.get_state()))
 with pytest.raises(RuntimeError,match='boom'):
  with disk_capture(pairs,opt,generators=(g,),scratch=tmp_path) as f:
   f.apply({i:.5 for i in range(28)});raise RuntimeError('boom')
 assert before==digest(([(p.A,p.B) for p in pairs],opt.state_dict(),g.get_state()));assert not list(tmp_path.iterdir())
