import gc
import weakref
import pytest
import torch
from fizgig.training.activation_async import AsyncHostBudget, PinnedLease, cuda_preflight
from fizgig.training.activation_offload import self_test


class Event:
    def __init__(self, ready=False): self.ready = ready
    def query(self): return self.ready
    def synchronize(self): self.ready = True


def test_retired_payload_charged_until_all_events():
    b = AsyncHostBudget(16)
    assert b.acquire(16)
    host = torch.empty(4)
    ref = weakref.ref(host)
    e1, e2 = Event(True), Event()
    lease = PinnedLease(b, host, e1, torch.device('cuda'), 16)
    lease.events.append(e2)
    del lease, host
    gc.collect()
    assert ref() is not None
    assert not b.acquire(1)
    assert b.live == 16
    b.drain()
    assert b.live == 0 and b.drain_waits == 1
    assert ref() is None
    assert b.acquire(16)
    b.release(16)


def test_drain_does_not_release_active_graph():
    b = AsyncHostBudget(16)
    assert b.acquire(16)
    lease = PinnedLease(b, torch.empty(4), Event(), torch.device('cuda'), 16)
    b.drain()
    assert b.live == 16
    del lease
    gc.collect()
    b.drain()
    assert b.live == 0


def test_host_floor_and_sync_regression():
    b = AsyncHostBudget(16, host_floor_bytes=32, available=lambda: 40)
    assert not b.acquire(9)
    self_test()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_preflight():
    assert cuda_preflight()['passed']
