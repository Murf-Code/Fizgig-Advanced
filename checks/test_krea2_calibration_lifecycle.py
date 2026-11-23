"""Krea 2 calibration lifecycle: diagnostics, resume identity and state persistence."""
import json
import logging
import os
from types import SimpleNamespace

import pytest
import torch
from fizgig.families import train


def call_prepare(tmp_path, *, resume=None, enabled=True, held=None):
    return train._prepare_krea2_direction_edit(
        SimpleNamespace(key="krea2"), {"direction_edit": "1" if enabled else ""},
        "lora", False, False, None, object(), held if held is not None else {"a": "caption"},
        "dit.safetensors", "te.safetensors", str(tmp_path), "demo", 42, "cpu",
        resume_state_dir=str(resume) if resume else None)


def test_resume_restores_exact_calibration_even_when_toggle_is_off(tmp_path):
    state = tmp_path / "demo-000001-state"
    state.mkdir()
    (state / "training_state.json").write_text(json.dumps({"calibration_active": True}))
    (state / "calibration.safetensors").write_bytes(b"fixed calibration")
    result = call_prepare(tmp_path, resume=state, enabled=False)
    assert result["resume"] is True
    assert result["lora"] == str(state / "calibration.safetensors")


def test_missing_calibration_in_resume_is_fatal(tmp_path):
    state = tmp_path / "demo-000001-state"
    state.mkdir()
    (state / "training_state.json").write_text(json.dumps({"calibration_active": True}))
    with pytest.raises(RuntimeError, match="missing"):
        call_prepare(tmp_path, resume=state)


def test_legacy_resume_does_not_recalibrate(tmp_path, caplog):
    state = tmp_path / "demo-000001-state"
    state.mkdir()
    (state / "training_state.json").write_text(json.dumps({"epoch": 1}))
    with caplog.at_level(logging.WARNING):
        assert call_prepare(tmp_path, resume=state) is None
    assert "skipped on resume" in caplog.text


def test_missing_validation_holdout_is_logged(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        result = call_prepare(tmp_path, held={})
    assert result is None
    assert "Image Validation hold-out" in caplog.text


def test_calibration_exception_is_visible(tmp_path, monkeypatch, caplog):
    from fizgig.krea2 import direction_edit_v2
    monkeypatch.setattr(direction_edit_v2, "captions_from_group",
                        lambda _: {str(i): "caption" for i in range(8)})
    monkeypatch.setattr(direction_edit_v2, "captions_from_held", lambda _: {"h": "caption"})
    monkeypatch.setattr(direction_edit_v2, "make_config", lambda *a, **kw: {})
    def fail(*a, **kw):
        raise RuntimeError("calibration sentinel")
    monkeypatch.setattr(direction_edit_v2, "run_direction_edit", fail)
    with caplog.at_level(logging.ERROR):
        assert call_prepare(tmp_path) is None
    assert "calibration sentinel" in caplog.text


def test_fresh_calibration_reads_family_options_and_starts_scan(tmp_path, monkeypatch):
    """A fresh enabled scan must reach the candidate runner, not fail in setup."""
    from fizgig.krea2 import direction_edit_v2

    monkeypatch.setattr(direction_edit_v2, "captions_from_group",
                        lambda _: {str(i): "furry, paws, fur" for i in range(8)})
    monkeypatch.setattr(direction_edit_v2, "captions_from_held",
                        lambda _: {"held": "furry, paws, fur"})
    seen = {}

    def fake_config(opts, **kwargs):
        seen["options"] = dict(opts)
        return {}

    def fake_run(cfg, **kwargs):
        seen["config"] = dict(cfg)
        path = os.path.join(kwargs["output_dir"], "candidate.safetensors")
        with open(path, "wb") as f:
            f.write(b"test candidate")
        return {"lora": path, "verify": {"held_out": True, "beats_random": True}}

    monkeypatch.setattr(direction_edit_v2, "make_config", fake_config)
    monkeypatch.setattr(direction_edit_v2, "run_direction_edit", fake_run)
    result = train._prepare_krea2_direction_edit(
        SimpleNamespace(key="krea2"),
        {"direction_edit": "1", "direction_edit_target_tags": "furry, paws"},
        "lora", False, False, None, object(), {"held": object()},
        "dit.safetensors", "te.safetensors", str(tmp_path), "demo", 42, "cpu")
    assert result is not None
    try:
        assert os.path.isfile(result["lora"])
        assert seen["options"] == {"direction_edit_scan": "1",
                                   "direction_edit_target_tags": "furry, paws"}
        assert seen["config"]["scan"] is True
        assert seen["config"]["caption_mode"] is True
        assert seen["config"]["scan_targets"] == 1  # ceil(4% of eight captions)
        assert seen["config"]["scan_auto_top_fraction"] == 0.25
        assert seen["config"]["verification_records"] == {"held": "furry, paws, fur"}
    finally:
        import shutil
        shutil.rmtree(result["_temporary_dir"], ignore_errors=True)





def test_calibration_auto_target_budget_is_four_percent(tmp_path, monkeypatch):
    """156 training images allocate seven ADDITIONAL automatic target slots."""
    from fizgig.krea2 import direction_edit_v2
    monkeypatch.setattr(direction_edit_v2, "captions_from_group",
                        lambda _: {str(i): "tag01, tag02" for i in range(156)})
    monkeypatch.setattr(direction_edit_v2, "captions_from_held",
                        lambda _: {"holdout": "tag01, tag02"})
    monkeypatch.setattr(direction_edit_v2, "make_config", lambda *a, **kw: {})
    found = {}
    def fake_runner(cfg, **kwargs):
        found.update(cfg)
        return {"status": "skipped", "verify": None}
    monkeypatch.setattr(direction_edit_v2, "run_direction_edit", fake_runner)
    assert call_prepare(tmp_path) is None
    assert found["scan_targets"] == 7
    assert found["scan_auto_top_fraction"] == 0.25

def test_prompt_encoder_unloads_model_when_encoding_fails():
    """A failed calibration encoding must not strand the text encoder on the GPU."""
    from fizgig.krea2 import direction_edit_v2 as direction_edit

    class FakeDriver:
        def __init__(self):
            self.loaded = 0
            self.unloaded = 0

        def load_text_encoder(self, _path, _device):
            self.loaded += 1
            return object()

        def encode_text(self, _te, _prompts):
            raise RuntimeError("simulated encoder error")

        def unload_text_encoder(self, _te):
            self.unloaded += 1

    driver = FakeDriver()
    with pytest.raises(RuntimeError, match="simulated encoder error"):
        direction_edit.encode_prompts(["furry, mane"], "dummy-te", "cpu", driver=driver)
    assert driver.loaded == 1 and driver.unloaded == 1


class DummyNet:
    def save(self, path, **kwargs):
        with open(path, "wb") as f:
            f.write(b"trainable")


def test_state_carries_calibration_and_removes_stale_copy(tmp_path):
    candidate = tmp_path / "candidate.safetensors"
    candidate.write_bytes(b"same bytes on resume")
    opt = torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=0.1)
    params = dict(output_dir=str(tmp_path), output_name="demo", net=DummyNet(), optimizer=opt,
                  epoch=1, global_step=10, arch_id="krea2")
    train._save_state(**params, calibration_path=str(candidate))
    state = tmp_path / "demo-000001-state"
    assert (state / "calibration.safetensors").read_bytes() == candidate.read_bytes()
    assert json.loads((state / "training_state.json").read_text())["calibration_active"] is True
    train._save_state(**params)
    assert not (state / "calibration.safetensors").exists()
    assert json.loads((state / "training_state.json").read_text())["calibration_active"] is False
