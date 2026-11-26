"""Opt-in phase measurements for eager Krea 2 research.

Allocated/reserved peaks describe allocator activity in host enqueue intervals,
not physical residency or synchronized GPU durations. Existing preview reset
sites must be routed through the active probe. No added CUDA synchronization.
"""
import contextvars
import json
import logging
import time
from contextlib import contextmanager
import torch

logger = logging.getLogger(__name__)
_ACTIVE = contextvars.ContextVar('memory_phase_probe', default=None)


def _max(a,b):
    return tuple(max(x,y) for x,y in zip(a,b))


class CudaCounters:
    def __init__(self,device): self.device=device
    def current(self):
        return (torch.cuda.memory_allocated(self.device),torch.cuda.memory_reserved(self.device))
    def peaks(self):
        return (torch.cuda.max_memory_allocated(self.device),torch.cuda.max_memory_reserved(self.device))
    def reset(self): torch.cuda.reset_peak_memory_stats(self.device)
    def pressure(self):
        s=torch.cuda.memory_stats(self.device)
        return {k:int(s.get(k,0)) for k in ('num_alloc_retries','num_ooms','inactive_split_bytes.all.current')}


def active_probe(): return _ACTIVE.get()


class PhaseTelemetry:
    def __init__(self, backend, mode='peaks', emit=None):
        if mode not in ('passive','peaks'): raise ValueError('invalid memory telemetry mode')
        self.backend=backend;self.mode=mode
        self.emit=emit or (lambda rec:logger.info('[geometry-memory] %s',json.dumps(rec,separators=(',',':'))))
        self.epoch_peak=(0,0);self.run_peak=(0,0);self.preview_peak=(0,0)
        self.active=None;self.phase_peak=(0,0);self.epoch_started=False

    def training_peaks(self):
        if self.active=='preview': return self.epoch_peak
        self.epoch_peak=_max(self.epoch_peak,self.backend.peaks())
        self.run_peak=_max(self.run_peak,self.epoch_peak)
        return self.epoch_peak

    def epoch_reset(self):
        if self.active is not None: raise RuntimeError('cannot reset inside active phase')
        if self.epoch_started: self.training_peaks()
        self.backend.reset();self.epoch_peak=self.backend.current();self.epoch_started=True

    def preview_reset(self):
        if self.active!='preview': raise RuntimeError('preview reset outside preview')
        self.phase_peak=_max(self.phase_peak,self.backend.peaks());self.backend.reset()

    def preview_peaks(self):
        return _max(self.phase_peak,self.backend.peaks())

    @contextmanager
    def phase(self,name,step):
        if self.active is not None: raise RuntimeError('overlapping memory telemetry phases')
        if name=='preview' or self.epoch_started:self.training_peaks()
        if self.mode=='peaks':self.backend.reset()
        self.active=name;self.phase_peak=self.backend.current();start=self.backend.current()
        pressure0=self.backend.pressure();t=time.monotonic_ns();ok=False
        token=_ACTIVE.set(self)
        try:
            yield
            ok=True
        finally:
            try:
                end=self.backend.current();high=_max(self.phase_peak,self.backend.peaks())
                if name=='preview':
                    if self.mode=='peaks':self.preview_peak=_max(self.preview_peak,high)
                else:
                    self.epoch_peak=_max(self.epoch_peak,high);self.run_peak=_max(self.run_peak,self.epoch_peak)
                pressure1=self.backend.pressure()
                self.emit(dict(mode=self.mode,phase=name,step=int(step),ok=ok,start_bytes=start,end_bytes=end,
                               peak_bytes=high if self.mode=='peaks' else None,
                               training_epoch_peak_bytes=self.epoch_peak,training_run_peak_bytes=self.run_peak,
                               host_elapsed_ns=time.monotonic_ns()-t,pressure_before=pressure0,pressure_after=pressure1))
            finally:
                _ACTIVE.reset(token);self.active=None
                # Preview counters must not bleed into the next training interval.
                if name=='preview':self.backend.reset()


def storage_inventory(tensors):
    """Unique backing storage. Do not sum this with allocator totals."""
    seen={}
    for t in tensors:
        if not torch.is_tensor(t):continue
        s=t.untyped_storage();key=(str(t.device),s.data_ptr())
        seen[key]=max(seen.get(key,0),s.nbytes())
    return sum(seen.values())


def optimizer_inventory(optimizer):
    ps=[p for g in optimizer.param_groups for p in g['params']]
    state=[v for st in optimizer.state.values() for v in st.values() if torch.is_tensor(v)]
    by_dtype={str(d):storage_inventory(v for v in state if v.dtype==d) for d in {t.dtype for t in state}}
    return dict(parameters=storage_inventory(ps),gradients=storage_inventory(p.grad for p in ps if p.grad is not None),
                optimizer_state=storage_inventory(state),state_by_dtype=by_dtype)
