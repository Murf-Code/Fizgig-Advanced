"""The CPU / GPU / disk readout: while a run trains, a sampler thread reads every few seconds how busy the CPU and the
GPU are, the GPU's power and memory, how fast the drives read, and how much of the training time went waiting for
data (from background loading's own timing, training/prefetch.py). It writes them to ``<output_dir>/.perf_<name>.json``
for the web UI's run card, and at the end logs a summary with a plain hint and writes it into the same file.

Nothing here touches the training thread past reading a few counters: the sampler works on its own daemon thread, and
uses psutil and nvidia-ml-py (both already in requirements.txt). A part that can't be read (no GPU, no NVML) is left
out. FIZGIG_PERF_READOUT=0 turns it off.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

INTERVAL = 3.0                      # seconds between samples
WAIT_SLOW = 0.25                    # waiting on data this share of the time or more: the drive is likely slow
WAIT_SOME = 0.10
GPU_LOW = 60.0                      # % busy under which the GPU is held back by something else

_ACTIVE = None                      # this process's running watch (finish() ends it, however the run ends)


def file_for(output_dir, output_name):
    return os.path.join(output_dir, f".perf_{output_name}.json")


def _gpu_handle(device_index):
    """NVML's handle for the GPU torch trains on (by its UUID: CUDA_VISIBLE_DEVICES renumbers), or None."""
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception:
        return None, None
    try:
        import torch
        uuid = str(torch.cuda.get_device_properties(device_index).uuid)
        return pynvml, pynvml.nvmlDeviceGetHandleByUUID(uuid if uuid.startswith(("GPU-", "MIG-")) else "GPU-" + uuid)
    except Exception:
        pass
    try:
        vis = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
        idx = int(vis[device_index]) if device_index < len(vis) and vis[device_index].isdigit() else device_index
        return pynvml, pynvml.nvmlDeviceGetHandleByIndex(idx)
    except Exception:
        return None, None


def hint(summary):
    """One plain sentence on what held the run back, from a summary."""
    w = summary.get("wait_share")
    if w is None:
        return ""
    pct = round(100 * w)
    if w >= WAIT_SLOW:
        tip = (" Background loading is off; turning it on (Preferences > Training memory) may help."
               if summary.get("background_loading") is False else
               " Moving the cache to a faster drive (an SSD) would speed training up.")
        return f"GPU waited on data {pct}% of the time - the cache drive may be slow.{tip}"
    if w >= WAIT_SOME:
        return f"GPU waited on data {pct}% of the time - a little; a faster cache drive would help a bit."
    u = summary.get("gpu_util")
    if u is not None and u < GPU_LOW:
        return (f"Data kept up (GPU waited {pct}% of the time), but the GPU was busy only {round(u)}% - block swap, "
                "previews or other CPU work may be holding it back.")
    return f"Data kept up: the GPU waited on data only {pct}% of the time."


class PerfWatch:
    def __init__(self, output_dir, output_name, stats=None, device_index=0, background_loading=None,
                 interval=INTERVAL, gpu=True):
        self.path = file_for(output_dir, output_name)
        self.stats, self.interval, self.bg = stats, float(interval), background_loading
        self.device_index, self.gpu = int(device_index or 0), gpu
        self._stop = threading.Event()
        self._th = None
        self.samples = 0
        self._sum = {"cpu": 0.0, "gpu_util": 0.0, "gpu_w": 0.0, "disk_mb_s": 0.0}
        self._n = {k: 0 for k in self._sum}
        self.vram_peak = None
        self.vram_total = None
        self.t0 = time.time()
        self._base = self._counts()
        self.last = {}

    # --- reads -----------------------------------------------------------------------------------------------------
    def _counts(self):
        st = self.stats
        return (st.wait_s, st.busy_s, st.batches) if st is not None else (0.0, 0.0, 0)

    def _disk_bytes(self):
        try:
            import psutil
            d = psutil.disk_io_counters()
            return d.read_bytes if d is not None else None
        except Exception:
            return None

    def _sample(self, prev):
        """One reading; ``prev`` is the last one's (time, disk bytes, wait/busy counts)."""
        now = time.perf_counter()
        s = {"t": round(time.time() - self.t0, 1)}
        try:
            import psutil
            s["cpu"] = round(psutil.cpu_percent(None), 1)
        except Exception:
            pass
        disk = self._disk_bytes()
        if disk is not None and prev[1] is not None and now > prev[0]:
            s["disk_mb_s"] = round(max(0, disk - prev[1]) / (now - prev[0]) / 1e6, 1)
        nv, h = self._nv
        if h is not None:
            try:
                s["gpu_util"] = float(nv.nvmlDeviceGetUtilizationRates(h).gpu)
            except Exception:
                pass
            try:
                s["gpu_w"] = round(nv.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)
            except Exception:
                pass
            try:
                m = nv.nvmlDeviceGetMemoryInfo(h)
                s["vram_gb"], s["vram_total_gb"] = round(m.used / 2 ** 30, 2), round(m.total / 2 ** 30, 2)
            except Exception:
                pass
        elif self.gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    s["vram_gb"] = round(torch.cuda.memory_reserved(self.device_index) / 2 ** 30, 2)
            except Exception:
                pass
        wait, busy, n = self._counts()
        dw, db, dn = wait - prev[2][0], busy - prev[2][1], n - prev[2][2]
        if dw + db > 0:
            s["wait_share"] = round(dw / (dw + db), 3)
        if dn > 0:
            s["wait_ms"], s["step_ms"] = round(1000 * dw / dn, 1), round(1000 * db / dn, 1)
        w0, b0, _ = self._base
        if (wait - w0) + (busy - b0) > 0:
            s["wait_share_run"] = round((wait - w0) / ((wait - w0) + (busy - b0)), 3)
        for k in self._sum:
            if k in s:
                self._sum[k] += s[k]
                self._n[k] += 1
        if "vram_gb" in s:
            self.vram_peak = max(self.vram_peak or 0.0, s["vram_gb"])
            self.vram_total = s.get("vram_total_gb", self.vram_total)
        self.samples += 1
        return s, (now, disk, (wait, busy, n))

    # --- the thread ----------------------------------------------------------------------------------------------
    def _loop(self):
        self._nv = _gpu_handle(self.device_index) if self.gpu else (None, None)
        try:
            import psutil
            psutil.cpu_percent(None)                  # the first call only primes it
        except Exception:
            pass
        prev = (time.perf_counter(), self._disk_bytes(), self._counts())
        while not self._stop.wait(self.interval):
            try:
                self.last, prev = self._sample(prev)
                self._write({"live": self.last, "background_loading": self.bg})
            except Exception:                         # a readout never stops a run
                logger.debug("[perf] sample failed", exc_info=True)

    def start(self):
        self._nv = (None, None)
        self._write({"live": {}, "background_loading": self.bg})
        self._th = threading.Thread(target=self._loop, name="fizgig-perf-readout", daemon=True)
        self._th.start()
        return self

    def _write(self, data):
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self.path)
        except OSError:
            pass

    def summary(self):
        avg = {k: round(self._sum[k] / self._n[k], 1) for k in self._sum if self._n[k]}
        wait, busy, n = self._counts()
        w0, b0, n0 = self._base
        dw, db, dn = wait - w0, busy - b0, n - n0
        out = {**avg, "steps": dn, "background_loading": self.bg, "seconds": round(time.time() - self.t0, 1)}
        if dw + db > 0:
            out["wait_share"] = round(dw / (dw + db), 3)
        if dn:
            out["wait_ms"], out["step_ms"] = round(1000 * dw / dn, 1), round(1000 * db / dn, 1)
        if self.vram_peak is not None:
            out["vram_peak_gb"] = self.vram_peak
            if self.vram_total:
                out["vram_total_gb"] = self.vram_total
        out["hint"] = hint(out)
        return out

    def finish(self):
        """Stop the sampler, log the summary and write it into the readout file. Returns the summary."""
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=self.interval + 2)
        s = self.summary()
        if s.get("steps"):
            parts = [f"data wait {round(100 * s['wait_share'])}%" if "wait_share" in s else None,
                     f"{s['wait_ms']:.0f} ms waiting / {s['step_ms']:.0f} ms training a step" if "wait_ms" in s else None,
                     f"CPU {s['cpu']:.0f}%" if "cpu" in s else None,
                     f"GPU {s['gpu_util']:.0f}%" if "gpu_util" in s else None,
                     f"{s['gpu_w']:.0f} W" if "gpu_w" in s else None,
                     f"VRAM peak {s['vram_peak_gb']:.1f} GB" if "vram_peak_gb" in s else None,
                     f"disk read {s['disk_mb_s']:.0f} MB/s" if "disk_mb_s" in s else None]
            logger.info("[perf] " + ", ".join(p for p in parts if p))
            if s["hint"]:
                logger.info(f"[perf] {s['hint']}")
        self._write({"live": self.last, "summary": s, "background_loading": self.bg})
        return s


def wanted():
    return str(os.environ.get("FIZGIG_PERF_READOUT", "1")).strip().lower() not in ("0", "off", "false", "no")


def start(output_dir, output_name, stats=None, device_index=0, background_loading=None, **kw):
    """Start this run's readout (ends any earlier one in this process first). None when it's turned off."""
    global _ACTIVE
    finish()
    if not wanted():
        return None
    try:
        _ACTIVE = PerfWatch(output_dir, output_name, stats, device_index, background_loading, **kw).start()
    except Exception:
        logger.debug("[perf] could not start the readout", exc_info=True)
        _ACTIVE = None
    return _ACTIVE


def finish():
    """End the running readout, if any: its summary, or None."""
    global _ACTIVE
    w, _ACTIVE = _ACTIVE, None
    if w is None:
        return None
    try:
        return w.finish()
    except Exception:
        logger.debug("[perf] could not finish the readout", exc_info=True)
        return None
