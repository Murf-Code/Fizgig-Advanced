"""Background export controller for the experimental CPU checkpoint mixer.

Separate from the GPU Studio worker and training lifecycle. At most one mix
exports at a time; the UI polls only while the Studio panel is open. Cancel
signals a chunk boundary, so a partial output never replaces a real model.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
import uuid

from fizgig.families import checkpoint_mix

_LOCK = threading.RLock()
_CANCEL = threading.Event()
_STATE = {
    "state": "idle", "id": "", "family": "", "base": "", "donor": "",
    "output": "", "bytes_done": 0, "bytes_total": 0, "error": "",
    "result": None, "started_at": None, "ended_at": None
}


def status():
    with _LOCK:
        return dict(_STATE)


def inspect(data):
    return checkpoint_mix.inspect_pair(
        str(data.get("base") or "").strip(),
        str(data.get("donor") or "").strip(),
        str(data.get("family") or "").strip())


def start(data):
    family = str(data.get("family") or "").strip()
    base = str(data.get("base") or "").strip()
    donor = str(data.get("donor") or "").strip()
    output = os.path.abspath(str(data.get("output") or "").strip())
    weights = data.get("weights") or {}
    overwrite = data.get("overwrite") is True
    with _LOCK:
        if _STATE["state"] in {"running", "cancelling"}:
            raise ValueError("A checkpoint blend is already running.")
    plan = checkpoint_mix.inspect_pair(base, donor, family)
    if not output.lower().endswith(".safetensors"):
        raise ValueError("Output must be a .safetensors checkpoint.")
    folder = os.path.dirname(output)
    if not os.path.isdir(folder):
        raise ValueError("Choose an existing output folder.")
    if os.path.abspath(base) == output or os.path.abspath(donor) == output:
        raise ValueError("Never export over either input model.")
    if os.path.exists(output) and not overwrite:
        return {"exists": True, "path": output}
    # A checkpoint export writes a full new model; refuse obvious disk exhaustion up front.
    free = shutil.disk_usage(folder).free
    if free < plan["base_bytes"] + (64 << 20):
        raise ValueError("Not enough disk space for a new checkpoint. "
                         f"Need approximately {plan['base_bytes'] / 1e9:.1f} GB free.")
    run_id = uuid.uuid4().hex[:12]
    with _LOCK:
        if _STATE["state"] in {"running", "cancelling"}:
            raise ValueError("A checkpoint blend is already running.")
        _CANCEL.clear()
        _STATE.update(
            state="running", id=run_id, family=family, base=base, donor=donor,
            output=output, bytes_done=0, bytes_total=plan["base_bytes"],
            error="", result=None, started_at=time.time(), ended_at=None)

    def worker():
        last_report = [0.0]
        try:
            def progress(done, total):
                now = time.monotonic()
                if now - last_report[0] > 0.25 or done >= total:
                    with _LOCK:
                        if _STATE["id"] == run_id:
                            _STATE.update(bytes_done=done, bytes_total=total)
                    last_report[0] = now
            result = checkpoint_mix.merge_pair(
                base, donor, output, family, weights, overwrite=overwrite,
                cancel=_CANCEL, progress=progress)
            with _LOCK:
                if _STATE["id"] == run_id:
                    _STATE.update(state="completed", result=result,
                                  bytes_done=_STATE["bytes_total"], ended_at=time.time())
        except checkpoint_mix.MixCancelled:
            with _LOCK:
                if _STATE["id"] == run_id:
                    _STATE.update(state="cancelled", ended_at=time.time())
        except Exception as e:
            with _LOCK:
                if _STATE["id"] == run_id:
                    _STATE.update(state="failed", error=str(e), ended_at=time.time())

    threading.Thread(target=worker, daemon=True, name="fizgig-checkpoint-mix").start()
    return {"ok": True, "id": run_id, "output": output}


def cancel():
    with _LOCK:
        if _STATE["state"] not in {"running", "cancelling"}:
            return {"ok": False, "state": _STATE["state"]}
        _CANCEL.set()
        _STATE["state"] = "cancelling"
        return {"ok": True, "state": "cancelling"}
