"""Proposed eager-mode synchronous activation-offload reference, NOT integrated.
Use one HostBudget per training process/device, shared across live graphs.
No pinned allocations, async copies, compile, CUDA graph capture, or retained
backward graph support is claimed. Requires immutable checkpoint inputs.
CPU test mode exercises saved tensor selection/lifetime, not GPU memory.
"""
import threading
import weakref
import torch
from torch.utils.checkpoint import checkpoint

class HostBudget:
    def __init__(self, cap_bytes):
        if cap_bytes < 0: raise ValueError('negative cap')
        self.cap = int(cap_bytes)
        self.live = self.peak = self.accepted = self.skipped = 0
        self.lock = threading.Lock()
    def acquire(self, n):
        with self.lock:
            if self.live + n > self.cap:
                self.skipped += 1
                return False
            self.live += n
            self.peak = max(self.live, self.peak)
            self.accepted += 1
            return True
    def release(self, n):
        with self.lock:
            self.live -= n
            assert self.live >= 0


def checkpoint_bounded(block, activation, *args, budget, enabled=True, cpu_test=False):
    """Wrap only the existing nonreentrant checkpoint call, not the module loop."""
    def direct():
        return checkpoint(block, activation, *args, use_reentrant=False, preserve_rng_state=True)
    if (not enabled or (not activation.is_cuda and not cpu_test)
            or not activation.is_contiguous() or activation._base is not None
            or activation.storage_offset() != 0
            or activation.untyped_storage().nbytes() != activation.numel()*activation.element_size()
            or any(torch.is_tensor(a) and a.device == activation.device and
                   a.untyped_storage().data_ptr() == activation.untyped_storage().data_ptr() for a in args)):
        return direct()
    target = id(activation)
    def pack(x):
        n = x.numel()*x.element_size()
        # No closure strong reference to the original activation in the hooks.
        if id(x) == target and budget.acquire(n):
            try:
                host = x.detach().to('cpu',copy=True,non_blocking=False)
                weakref.finalize(host,budget.release,n)
            except BaseException:
                budget.release(n)
                raise
            return ('cpu',host,x.device,weakref.ref(x),x._version)
        return ('keep',x.detach(),None,None,x._version)
    def unpack(p):
        kind,value,dev,source_ref,version=p
        if kind=='keep':
            if value._version != version: raise RuntimeError('checkpoint input modified in place')
            return value
        source=source_ref()
        if source is not None and source._version != version:
            raise RuntimeError('offloaded checkpoint input modified in place')
        return value.to(dev,non_blocking=False)
    with torch.autograd.graph.saved_tensors_hooks(pack,unpack):
        return direct()


def self_test():
    import gc, copy
    torch.manual_seed(43)
    model=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(16,16),torch.nn.Dropout(.2),torch.nn.Tanh()) for _ in range(4)])
    def run(off):
        m=copy.deepcopy(model); b=HostBudget(2*3*16*4)
        torch.manual_seed(91);x=torch.randn(3,16,requires_grad=True);y=x
        for layer in m:y=checkpoint_bounded(layer,y,budget=b,enabled=off,cpu_test=True)
        y.sum().backward();out=(y.detach(),x.grad,*[p.grad for p in m.parameters()],torch.rand(3));del y;gc.collect()
        return out,b
    ref,_=run(False);actual,b=run(True)
    for x,y in zip(ref,actual):torch.testing.assert_close(x,y,atol=0,rtol=0)
    assert b.accepted==2 and b.skipped==2 and b.live==0 and b.peak==b.cap,(b.live,b.peak,b.accepted,b.skipped)
    print('PASS: bitwise outputs/gradients/RNG; cap enforced; host leases released',vars(b))
if __name__=='__main__':self_test()
