import pytest
import torch
from fizgig.training.memory_telemetry import PhaseTelemetry, active_probe, optimizer_inventory, storage_inventory

class Mock:
    def __init__(self): self.now=(0,0);self.high=(0,0);self.resets=0
    def set(self,a,r): self.now=(a,r);self.high=tuple(max(x,y) for x,y in zip(self.now,self.high))
    def current(self):return self.now
    def peaks(self):return self.high
    def pressure(self):return {'num_alloc_retries':0}
    def reset(self):self.high=self.now;self.resets+=1

def test_training_and_preview_peak_ownership():
    m=Mock();records=[];p=PhaseTelemetry(m,emit=records.append);m.set(5,6);p.epoch_reset()
    with p.phase('forward',1):m.set(10,12);m.set(6,7)
    with p.phase('backward',1):m.set(13,14);m.set(8,9)
    with p.phase('optimizer',1):m.set(9,11);m.set(6,7)
    assert [r['peak_bytes'] for r in records]==[(10,12),(13,14),(9,11)]
    assert p.training_peaks()==(13,14)
    with p.phase('preview',1):
        assert active_probe() is p
        m.set(20,22);m.set(8,9);p.preview_reset();m.set(16,18)
        assert p.preview_peaks()==(20,22)
        m.set(6,7)
    assert p.training_peaks()==(13,14) and p.preview_peak==(20,22)
    p.epoch_reset()
    with p.phase('forward',2):m.set(9,10)
    assert p.epoch_peak==(9,10) and p.run_peak==(13,14)

def test_exception_and_passive():
    m=Mock();records=[];p=PhaseTelemetry(m,mode='passive',emit=records.append);p.epoch_reset();base=m.resets
    with pytest.raises(ValueError):
        with p.phase('backward',1):
            with pytest.raises(RuntimeError):p.epoch_reset()
            raise ValueError('synthetic')
    assert not records[0]['ok'] and records[0]['peak_bytes'] is None and active_probe() is None and p.active is None
    assert m.resets==base

def test_unique_state_storage():
    base=torch.ones(4,8);assert storage_inventory([base,base[:,0]])==128
    p=torch.nn.Parameter(torch.ones(4,8));p.grad=torch.ones_like(p)
    opt=torch.optim.SGD([p],lr=.1,momentum=.9);opt.step()
    d=optimizer_inventory(opt)
    assert d['optimizer_state']==d['gradients']==d['parameters']==128
