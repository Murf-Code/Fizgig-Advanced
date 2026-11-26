"""Opt-in local instrumentation adapter. Not installed in Fizgig.
All reset calls in a training process must be routed through one owner.
Phase measurements are allocator enqueue intervals, not GPU residency.
"""
from contextlib import contextmanager
import time,json

class CudaBackend:
    def __init__(self,device):
        import torch
        self.cuda=torch.cuda; self.device=device
        if not self.cuda.is_available():raise RuntimeError('CUDA required')
    def current(self):return self.cuda.memory_allocated(self.device),self.cuda.memory_reserved(self.device)
    def peaks(self):return self.cuda.max_memory_allocated(self.device),self.cuda.max_memory_reserved(self.device)
    def reset(self):self.cuda.reset_peak_memory_stats(self.device)

class PhaseProbe:
    def __init__(self,backend,diagnostic=False):
        self.b=backend;self.diagnostic=diagnostic;self.active=None;self.records=[];self.global_peak=(0,0)
    def capture(self):
        peak=self.b.peaks();self.global_peak=tuple(max(a,b) for a,b in zip(self.global_peak,peak));return peak
    def reset(self):
        if self.active is not None:raise RuntimeError('reset within measured phase')
        self.capture();self.b.reset()
    @contextmanager
    def phase(self,name):
        if self.active is not None:raise RuntimeError('nested phase')
        if self.diagnostic:self.reset()
        self.active=name;start=self.b.current();t=time.monotonic_ns();ok=False
        try:
            yield;ok=True
        finally:
            try:
                p=self.capture();self.records.append(dict(phase=name,ok=ok,start_bytes=start,end_bytes=self.b.current(),peak_bytes=p if self.diagnostic else None,cumulative_peak_bytes=self.global_peak,host_elapsed_ns=time.monotonic_ns()-t))
            finally:self.active=None
    def report(self):
        self.capture();return dict(mode='diagnostic' if self.diagnostic else 'passive',global_peak_bytes=self.global_peak,records=self.records)

class Mock:
    def __init__(self):self.now=(0,0);self.high=(0,0);self.resets=0
    def set(self,a,r):self.now=(a,r);self.high=tuple(max(x,y) for x,y in zip(self.now,self.high))
    def current(self):return self.now
    def peaks(self):return self.high
    def reset(self):self.high=self.now;self.resets+=1

def test():
    reports={}
    for diagnostic in (False,True):
        m=Mock();p=PhaseProbe(m,diagnostic);m.set(8,9)
        for name,peak in [('forward',(10,11)),('backward',(11,12)),('optimizer',(9,10)),('preview',(12,13))]:
            with p.phase(name):
                m.set(*peak);m.set(7,8)
                try:p.reset()
                except RuntimeError:pass
                else:raise AssertionError('nested reset accepted')
            assert p.records[-1]['peak_bytes']==(peak if diagnostic else None)
        assert m.resets==(4 if diagnostic else 0)
        p.reset();assert p.global_peak==(12,13)
        try:
            with p.phase('exception'):raise ValueError('synthetic')
        except ValueError:pass
        assert p.active is None and not p.records[-1]['ok']
        assert p.report()['global_peak_bytes']==(12,13)
        reports[str(diagnostic)]=p.report()
    return reports
if __name__=='__main__':
    with open('phase_probe_results.json','w') as f:json.dump(test(),f,indent=2)
    print('PASS: passive/diagnostic peak ownership, exception recovery, reset protection')
