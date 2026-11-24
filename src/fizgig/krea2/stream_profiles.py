"""Local, workload-scoped calibration profiles. No implicit training changes.

Each profile is an atomic JSON record keyed by the exact calibration identity.
The caller must supply a fresh identity and current free-memory measurements.
Files are local preferences, not trusted proof from an external machine.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

SCHEMA = 1
GIB = 2**30
HARDWARE_KEYS = ("os", "os_release", "machine", "cpu_model", "cpu_logical",
                 "ram_total_gib", "gpu", "gpu_uuid", "gpu_driver", "vram_total_gib",
                 "compute_capability", "torch", "cuda", "hip")
WORKLOAD_KEYS = ("family", "checkpoint_sha256", "batches_sha256", "swap", "rank",
                 "precision", "optimizer", "lr", "seed", "steps", "warmup",
                 "batch_size", "network_type", "checkpointing", "attention")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _hex(value, length):
    return (isinstance(value, str) and len(value) == length
            and all(c in "0123456789abcdef" for c in value))


def _num(value, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def make_identity(hardware, workload, revision, *, clean):
    """Return a canonical identity or reject missing/unsupported evidence."""
    if not clean or not _hex(revision, 40):
        raise ValueError("A clean, identified source revision is required")
    if any(k not in hardware for k in HARDWARE_KEYS):
        raise ValueError("Incomplete hardware identity")
    hw = {k: hardware[k] for k in HARDWARE_KEYS}
    for k in ("os", "os_release", "machine", "cpu_model", "gpu", "gpu_uuid", "torch", "cuda"):
        if not isinstance(hw[k], str) or not hw[k].strip():
            raise ValueError("Missing hardware identity: " + k)
    if hw["hip"] is not None:
        raise ValueError("Only CUDA profiles are supported by this diagnostic")
    if not isinstance(hw["gpu_driver"], str) or not hw["gpu_driver"].strip():
        raise ValueError("GPU driver version unavailable")
    if any(not _num(hw[k], True) for k in ("ram_total_gib", "vram_total_gib", "cpu_logical")):
        raise ValueError("Invalid hardware capacity")
    if (not isinstance(hw["compute_capability"], list) or len(hw["compute_capability"]) != 2
            or any(type(x) is not int or x < 0 for x in hw["compute_capability"])):
        raise ValueError("Missing GPU compute capability")
    if any(k not in workload for k in WORKLOAD_KEYS):
        raise ValueError("Incomplete workload identity")
    work = {k: workload[k] for k in WORKLOAD_KEYS}
    if (work["family"] != "krea2" or work["network_type"] != "lora"
            or work["checkpointing"] is not True or work["attention"] != "auto"
            or work["batch_size"] != 1 or work["precision"] not in ("bf16", "int8")
            or work["optimizer"] not in ("muon", "adamw")):
        raise ValueError("Workload is outside the validated diagnostic scope")
    if any(not _hex(work[k], 64) for k in ("checkpoint_sha256", "batches_sha256")):
        raise ValueError("Real checkpoint and cached input fingerprints required")
    if (any(type(work[k]) is not int for k in ("swap", "rank", "seed", "steps", "warmup"))
            or not 1 <= work["swap"] <= 26 or work["rank"] < 1
            or work["warmup"] < 1 or work["steps"] - work["warmup"] < 3
            or not _num(work["lr"], True)):
        raise ValueError("Invalid calibration workload")
    return {"schema": SCHEMA, "revision": revision, "hardware": hw, "workload": work}


def _valid_identity(identity):
    if identity.get("schema") != SCHEMA:
        raise ValueError("Unsupported profile schema")
    rebuilt = make_identity(identity["hardware"], identity["workload"],
                            identity["revision"], clean=True)
    if rebuilt != identity:
        raise ValueError("Noncanonical profile identity")
    return rebuilt


def file_fingerprint(path):
    """Stream the checkpoint hash; reject modifications during the read."""
    path = Path(path)
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 2**20), b""):
            h.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("Checkpoint changed during fingerprinting")
    return h.hexdigest()


def profile_path(directory, identity):
    return Path(directory) / (_digest(_valid_identity(identity)) + ".json")


def _evidence(profile):
    selected = profile["selected"]
    if (selected.get("mode") != "fast" or type(selected.get("pinned")) is not bool
            or type(selected.get("ring")) is not int or not 2 <= selected["ring"] <= 4
            or selected["ring"] > 28 - profile["identity"]["workload"]["swap"]):
        raise ValueError("Invalid selected strategy")
    m = profile["measurements"]
    if any(not _num(m.get(k)) for k in ("conservative_gain", "peak_allocated_gib",
                                       "peak_reserved_gib", "pinned_storage_gib",
                                       "min_available_ram_gib")):
        raise ValueError("Invalid calibration measurements")
    hw = profile["identity"]["hardware"]
    if (not 0.05 <= m["conservative_gain"] < 1 or
            not 0 < m["peak_allocated_gib"] <= m["peak_reserved_gib"] <= hw["vram_total_gib"]
            or m["min_available_ram_gib"] < max(6.0, hw["ram_total_gib"] * 0.15)):
        raise ValueError("Calibration evidence does not meet safety thresholds")
    if selected["pinned"] or selected["ring"] != 2:
        if not _num(m.get("gain_over_staged_ring2")) or m["gain_over_staged_ring2"] < 0.05:
            raise ValueError("Aggressive strategy lacks staged comparison")
    return m


def save_profile(directory, identity, report, parity_check):
    """Recompute qualification; never trust the report's cached recommendation."""
    from fizgig.krea2.stream_calibration import recommend_stream_strategy
    _valid_identity(identity)
    cfg = report.get("configuration", {})
    if not cfg.get("model") or not cfg.get("cache") or cfg.get("cpu") is not False:
        raise ValueError("Real checkpoint and real cached inputs are required")
    if report.get("profile_identity") != identity or report.get("git_commit") != identity["revision"]:
        raise ValueError("Report identity does not match requested profile")
    if any(report.get("hardware", {}).get(k) != v for k, v in identity["hardware"].items()):
        raise ValueError("Report hardware differs from profile identity")
    for k in ("swap", "rank", "precision", "optimizer", "lr", "seed", "steps", "warmup"):
        if cfg.get(k) != identity["workload"][k]:
            raise ValueError("Report workload differs: " + k)
    recommendation = recommend_stream_strategy(report, parity_check)
    if recommendation["status"] != "recommended":
        raise ValueError("No qualified strategy: " + recommendation["reason"])
    selected = recommendation["selected"]
    entry = next(c for c in recommendation["candidates"] if c.get("eligible") and
                 all(c.get(k) == v for k, v in selected.items()))
    keys = ("conservative_gain", "gain_over_staged_ring2", "peak_allocated_gib",
            "peak_reserved_gib", "pinned_storage_gib", "min_available_ram_gib", "repeat_medians")
    profile = {"identity": identity, "selected": selected,
               "measurements": {k: entry[k] for k in keys if k in entry},
               "created_utc": datetime.now(timezone.utc).isoformat()}
    _evidence(profile)
    path = profile_path(directory, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=".profile-", suffix=".tmp", delete=False) as out:
            temp = Path(out.name)
            json.dump(profile, out, indent=2, allow_nan=False)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return path


def lookup_profile(directory, identity, *, available_ram_gib, free_vram_gib):
    """Read a matching profile and recheck live headroom before offering reuse."""
    try:
        path = profile_path(directory, identity)
        if not path.is_file():
            return None, "No exact matching calibration profile"
        if path.stat().st_size > 128 * 1024:
            raise ValueError("Profile exceeds size limit")
        with path.open(encoding="utf-8") as source:
            profile = json.load(source)
        if profile["identity"] != identity:
            raise ValueError("Profile identity mismatch")
        m = _evidence(profile)
        reserve = max(6.0, identity["hardware"]["ram_total_gib"] * 0.15)
        if not _num(available_ram_gib) or available_ram_gib < reserve + m["pinned_storage_gib"]:
            return None, "Insufficient current system RAM for the measured pinned storage"
        if not _num(free_vram_gib) or free_vram_gib < m["peak_reserved_gib"]:
            return None, "Insufficient current free VRAM for the measured peak"
        return profile, "Exact profile match; current memory preflight passed"
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return None, "Profile unavailable: " + str(exc)
