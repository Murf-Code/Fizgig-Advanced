"""Profile identity, atomic persistence and safe reuse regression tests."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

# Reuse the exact calibration fixture; no GPU/model loads.
path = Path(__file__).with_name("test_stream_calibration.py")
spec = importlib.util.spec_from_file_location("stream_calibration_fixture", path)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
from fizgig.krea2 import stream_profiles as profiles


def _fixture():
    report = fixture._report()
    hardware = dict(os="Windows", os_release="11", machine="AMD64", cpu_model="test cpu",
                    cpu_logical=16, ram_total_gib=64, gpu="test gpu", gpu_uuid="GPU-abc",
                    gpu_driver="test driver", vram_total_gib=24, compute_capability=[8, 9],
                    torch="test torch", cuda="test cuda", hip=None)
    work = dict(family="krea2", checkpoint_sha256="a"*64, batches_sha256="b"*64,
                swap=22, rank=32, precision="int8", optimizer="muon", lr=0.00025,
                seed=42, steps=4, warmup=1, batch_size=1, network_type="lora",
                checkpointing=True, attention="auto")
    identity = profiles.make_identity(hardware, work, "c"*40, clean=True)
    report["hardware"] = hardware
    report["profile_identity"] = copy.deepcopy(identity)
    report["git_commit"] = identity["revision"]
    report["configuration"].update({k: work[k] for k in
        ("swap", "rank", "precision", "optimizer", "lr", "seed", "steps", "warmup")})
    report["configuration"]["cache"] = "cache"
    return identity, report


def _save(tmp, identity, report):
    return profiles.save_profile(tmp, identity, report, fixture.diag.evaluate_exact_parity)


def _lookup(tmp, identity, ram=40, vram=24):
    return profiles.lookup_profile(tmp, identity, available_ram_gib=ram, free_vram_gib=vram)


def test_roundtrip_and_atomic_replacement(tmp_path):
    identity, report = _fixture()
    path = _save(tmp_path, identity, report)
    profile, reason = _lookup(tmp_path, identity)
    assert profile["selected"] == dict(mode="fast", pinned=False, ring=3)
    assert profile["identity"] == identity
    assert path == _save(tmp_path, identity, report)
    assert list(tmp_path.iterdir()) == [path]
    assert "passed" in reason


@pytest.mark.parametrize("key,value", [
    ("gpu_uuid", "GPU-other"), ("gpu_driver", "new driver"), ("ram_total_gib", 128),
    ("torch", "updated"), ("cuda", "updated"), ("cpu_model", "other cpu"),
])
def test_changed_hardware_invalidates(tmp_path, key, value):
    identity, report = _fixture()
    _save(tmp_path, identity, report)
    changed = copy.deepcopy(identity)
    changed["hardware"][key] = value
    assert _lookup(tmp_path, changed)[0] is None


@pytest.mark.parametrize("key,value", [
    ("checkpoint_sha256", "d"*64), ("batches_sha256", "e"*64), ("swap", 20),
    ("rank", 64), ("precision", "bf16"), ("optimizer", "adamw"), ("seed", 99),
])
def test_changed_workload_invalidates(tmp_path, key, value):
    identity, report = _fixture()
    _save(tmp_path, identity, report)
    changed = copy.deepcopy(identity)
    changed["workload"][key] = value
    assert _lookup(tmp_path, changed)[0] is None


def test_source_revision_and_dirty_worktree(tmp_path):
    identity, report = _fixture()
    _save(tmp_path, identity, report)
    changed = copy.deepcopy(identity)
    changed["revision"] = "f"*40
    assert _lookup(tmp_path, changed)[0] is None
    with pytest.raises(ValueError, match="clean"):
        profiles.make_identity(identity["hardware"], identity["workload"], identity["revision"], clean=False)


@pytest.mark.parametrize("ram,vram", [(5, 24), (11, 24), (40, 16), (float("nan"), 24)])
def test_live_memory_guard(tmp_path, ram, vram):
    identity, report = _fixture()
    _save(tmp_path, identity, report)
    assert _lookup(tmp_path, identity, ram, vram)[0] is None


@pytest.mark.parametrize("defect", ["cpu", "synthetic", "cached_recommendation", "identity",
                                   "workload", "hardware", "worker_error"])
def test_refuse_unqualified_profile(tmp_path, defect):
    identity, report = _fixture()
    if defect == "cpu":
        report["configuration"]["cpu"] = True
    elif defect == "synthetic":
        report["configuration"]["cache"] = None
    elif defect == "cached_recommendation":
        report["calibration"] = {"status": "recommended"}
        report["runs"] = []
    elif defect == "identity":
        report["profile_identity"]["revision"] = "d"*40
    elif defect == "workload":
        report["configuration"]["rank"] = 8
    elif defect == "hardware":
        report["hardware"]["gpu_driver"] = "different"
    elif defect == "worker_error":
        report["errors"].append({"error": "worker failed"})
    with pytest.raises(ValueError):
        _save(tmp_path, identity, report)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("defect", ["invalid_json", "ring", "gain", "schema", "oversize"])
def test_corrupt_profile_is_ignored(tmp_path, defect):
    identity, report = _fixture()
    path = _save(tmp_path, identity, report)
    if defect == "invalid_json":
        path.write_text("{", encoding="utf-8")
    elif defect == "oversize":
        path.write_text("x" * 140000, encoding="utf-8")
    else:
        record = json.loads(path.read_text())
        if defect == "ring":
            record["selected"]["ring"] = 5
        elif defect == "gain":
            record["measurements"]["conservative_gain"] = 0.01
        elif defect == "schema":
            record["identity"]["schema"] = 999
        path.write_text(json.dumps(record), encoding="utf-8")
    assert _lookup(tmp_path, identity)[0] is None


def test_checkpoint_content_fingerprint(tmp_path):
    path = tmp_path / "checkpoint"
    path.write_bytes(b"first")
    initial = profiles.file_fingerprint(path)
    path.write_bytes(b"other")
    assert profiles.file_fingerprint(path) != initial


def test_batch_content_fingerprint():
    torch = fixture.diag.torch
    batch = [("test", torch.zeros(1, 2, dtype=torch.bfloat16), torch.zeros(1, 2),
              torch.ones(1), {"hidden_states": torch.zeros(1, 2),
                              "attention_mask": torch.ones(1, 2, dtype=torch.bool)})]
    first = fixture.diag.batch_fingerprint(batch)
    batch[0][1][0, 0] = 1
    assert fixture.diag.batch_fingerprint(batch) != first
