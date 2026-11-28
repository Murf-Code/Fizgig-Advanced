"""Experimental checkpoint-input offload: D2H overlap, demand H2D restore.

Immutable inputs only. One device, eager non-reentrant checkpoint, no retained
backward graphs. Cap covers live and retired pinned payloads, not allocator
cache or OS locked-memory accounting. No weight-stream scheduling changes.
"""
import weakref
import torch
from .activation_offload import HostBudget, checkpoint_bounded


class AsyncHostBudget(HostBudget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._retired = []
        self._stream = None
        self._device = None
        self.d2h_bytes = self.h2d_bytes = self.restores = 0
        self.retired_peak = self.drain_waits = 0

    def retire(self, host, events, n):
        # Finalizers may run on an autograd worker. No CUDA calls here.
        with self.lock:
            self._retired.append((host, events, n))
            self.retired_peak = max(self.retired_peak, sum(x[2] for x in self._retired))

    def reap(self):
        with self.lock:
            remaining = []
            for host, events, n in self._retired:
                if all(e.query() for e in events):
                    self.live -= n
                else:
                    remaining.append((host, events, n))
            self._retired = remaining
            assert self.live >= 0

    def acquire(self, n):
        self.reap()
        return super().acquire(n)

    def drain(self):
        # Only retired payloads; a live graph is never forcibly released.
        with self.lock:
            events = [e for _, es, _ in self._retired for e in es]
        for e in events:
            if not e.query():
                self.drain_waits += 1
                e.synchronize()
        self.reap()

    def snapshot(self):
        return {k: getattr(self, k) for k in (
            'cap', 'live', 'peak', 'accepted', 'skipped', 'layout_skipped',
            'd2h_bytes', 'h2d_bytes', 'restores', 'retired_peak', 'drain_waits')}

    def pack_host(self, x):
        if not x.is_cuda:
            raise RuntimeError('asynchronous activation offload requires CUDA')
        n = x.numel() * x.element_size()
        if not self.acquire(n):
            return None
        stream = None
        try:
            with torch.cuda.device(x.device):
                if self._stream is None:
                    self._stream = torch.cuda.Stream(device=x.device)
                    self._device = x.device
                if self._device != x.device:
                    raise RuntimeError('activation budget cannot cross devices')
                stream = self._stream
                host = torch.empty_like(x, device='cpu', pin_memory=True)
                producer = torch.cuda.current_stream(x.device)
                stream.wait_stream(producer)
                # Keep the CUDA allocation alive on the copy stream even when
                # the checkpoint's last Python reference disappears.
                x.record_stream(stream)
                with torch.cuda.stream(stream):
                    host.copy_(x.detach(), non_blocking=True)
                    ready = torch.cuda.Event()
                    ready.record(stream)
                lease = PinnedLease(self, host, ready, x.device, n)
                self.d2h_bytes += n
                return lease
        except BaseException:
            if stream is not None:
                stream.synchronize()
            self.release(n)
            raise


class PinnedLease:
    def __init__(self, budget, host, ready, device, n):
        self.budget, self.host, self.ready = budget, host, ready
        self.device, self.n = device, n
        self.events = [ready]
        weakref.finalize(self, budget.retire, host, self.events, n)

    def restore(self):
        with torch.cuda.device(self.device):
            consumer = torch.cuda.current_stream(self.device)
            consumer.wait_event(self.ready)
            try:
                # Demand restore on the consuming stream: no extra H2D stream
                # racing the checkpoint's recompute or the frozen weight ring.
                result = torch.empty_like(self.host, device=self.device)
                result.copy_(self.host, non_blocking=True)
                done = torch.cuda.Event()
                done.record(consumer)
                self.events.append(done)
            except BaseException:
                consumer.synchronize()
                raise
        self.budget.h2d_bytes += self.n
        self.budget.restores += 1
        return result


def cuda_preflight(device='cuda'):
    """Small deterministic checkpoint/RNG test; not Krea2 model parity."""
    import gc
    device = torch.device(device)
    idx = device.index if device.index is not None else torch.cuda.current_device()
    device = torch.device('cuda', idx)
    n = 32 * 64 * 4
    def run(mode, cap):
        budget = (AsyncHostBudget if mode == 'async' else HostBudget)(cap)
        with torch.cuda.stream(torch.cuda.Stream(device=device)):
            torch.manual_seed(173)
            x = torch.randn(32, 64, device=device, requires_grad=True)
            p = torch.randn(64, device=device, requires_grad=True)
            y = x
            for _ in range(4):
                y = checkpoint_bounded(lambda a, w: torch.nn.functional.dropout(a * w, .2, True).tanh(),
                                       y, p, budget=budget, enabled=mode != 'off')
            y.sum().backward()
            result = [a.detach().cpu() for a in (y, x.grad, p.grad, torch.rand(8, device=device))]
        del y, x, p
        gc.collect()
        if mode == 'async': budget.drain()
        assert budget.live == 0
        if mode != 'off':
            assert budget.accepted == cap // n and budget.skipped == 4 - cap // n
            assert budget.peak == cap
        if mode == 'async':
            assert budget.d2h_bytes == budget.h2d_bytes == cap
            assert budget.restores == budget.accepted
        return result
    with torch.random.fork_rng(devices=[idx]):
        ref = run('off', 4*n)
        for mode, cap in [('sync', 4*n), ('async', 4*n), ('async', 2*n)]:
            for a, b in zip(ref, run(mode, cap)):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
    return {'passed': True, 'cases': 3, 'device': str(device)}
