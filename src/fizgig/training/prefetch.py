"""Background loading: the next batches of the training data are read from the cache on a helper thread while the
GPU trains on the current one, so a step doesn't wait on the drive.

It changes nothing about what trains. The DataLoader's own iterator is made, and each pass's first batch read, on the
calling thread, as before (the sampler's shuffle draws from torch's RNG there). The dataset's other draws from
Python's global ``random`` - its buckets' shuffle at an epoch's start (``random.seed(seed + epoch)``) - are made on
their own ``random.Random`` while a helper thread reads (``image_dataset.global_random``), and the global state they
leave is handed over with the batch and put in place when the trainer takes that batch: the same state at the same
point as when the read happened there and then. A dataset whose reads draw on the global ``random`` mid-epoch
(MiniMax's reference slots) is read without the helper (``reads_draw_global_random``).

``Prefetch`` also counts the time the trainer spent waiting for each batch and working on it (``stats``), for the
CPU / GPU / disk readout (training/perf_watch.py), with background loading on or off.

On by default; Preferences > Training memory > "Background loading" (--no-background_loading) or
FIZGIG_BACKGROUND_LOADING=0 turns it off, FIZGIG_BACKGROUND_LOADING=1 on whatever the setting.
"""
from __future__ import annotations

import os
import queue
import random
import threading
import time

DEPTH = 2                      # batches read ahead: the next step's and one more


def wanted(setting=True):
    """On or off for this run: the environment's FIZGIG_BACKGROUND_LOADING (0/1) over the setting."""
    env = str(os.environ.get("FIZGIG_BACKGROUND_LOADING", "")).strip().lower()
    if env in ("0", "off", "false", "no"):
        return False
    if env in ("1", "on", "true", "yes"):
        return True
    return bool(setting)


class Stats:
    """Time the trainer waited for its batches and worked on them (seconds), and how many it took. Read from another
    thread (the readout's sampler): plain floats, updated in place."""
    __slots__ = ("wait_s", "busy_s", "batches")

    def __init__(self):
        self.wait_s = self.busy_s = 0.0
        self.batches = 0


_END = object()


class Prefetch:
    """A DataLoader whose passes are read ahead on a helper thread (``enabled``), or as before with only the timing
    counted. ``len()`` and other attributes are the loader's."""

    def __init__(self, loader, enabled=True, depth=DEPTH):
        self.loader = loader
        self.enabled = bool(enabled)
        self.depth = max(1, int(depth))
        self.stats = Stats()

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        st = self.stats
        it = iter(self.loader)                     # here, as a plain loop would make it (torch's RNG)
        t = time.perf_counter()
        try:
            first = next(it)                       # the pass's first batch here too: its epoch start's shuffle
        except StopIteration:
            return
        st.wait_s += time.perf_counter() - t
        st.batches += 1
        if not self.enabled:
            yield from self._plain(first, it)
            return
        yield from self._ahead(first, it)

    def _plain(self, first, it):
        st = self.stats
        done = time.perf_counter()
        yield first
        while True:
            t = time.perf_counter()
            st.busy_s += t - done
            try:
                b = next(it)
            except StopIteration:
                return
            done = time.perf_counter()
            st.wait_s += done - t
            st.batches += 1
            yield b

    def _ahead(self, first, it):
        from fizgig.dataset import image_dataset
        st = self.stats
        q = queue.Queue(maxsize=self.depth)
        stop = threading.Event()

        def put(x):
            while not stop.is_set():
                try:
                    q.put(x, timeout=0.2)
                    return True
                except queue.Full:
                    continue
            return False

        def read():
            while not stop.is_set():
                sink = image_dataset.global_random_sink([])
                try:
                    b = next(it)
                except StopIteration:
                    put((_END, None, None))
                    return
                except BaseException as e:            # raised on the trainer's thread, where the read would have
                    put((e, None, None))
                    return
                finally:
                    image_dataset.global_random_sink(None)
                if not put((None, b, sink[-1] if sink else None)):
                    return

        th = threading.Thread(target=read, name="fizgig-background-loading", daemon=True)
        th.start()
        try:
            done = time.perf_counter()
            yield first
            while True:
                t = time.perf_counter()
                st.busy_s += t - done
                err, b, rng_state = q.get()
                done = time.perf_counter()
                st.wait_s += done - t
                if err is _END:
                    return
                if err is not None:
                    raise err
                if rng_state is not None:             # the global random state this read would have left
                    random.setstate(rng_state)
                st.batches += 1
                yield b
        finally:                                      # the end of a pass, a break (early stop) or an error
            stop.set()
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
            th.join(timeout=10)


def reads_draw_global_random(group):
    """True when a dataset's reads draw on Python's global random mid-epoch: MiniMax H3 reference distillation picks a
    reference slot per read (``..._teref{N}``). Background loading stays off for such a run - the draws would land
    between the trainer's own."""
    for ds in getattr(group, "datasets", None) or []:
        bm = getattr(ds, "batch_manager", None)
        for items in (getattr(bm, "buckets", None) or {}).values():
            for it in items[:1]:
                te = getattr(it, "text_encoder_output_cache_path", None)
                if te:
                    stem, ext = os.path.splitext(te)
                    if any(os.path.exists(f"{stem}ref{i}{ext}") for i in range(8)):
                        return True
            break
    return False
