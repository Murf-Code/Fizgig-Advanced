"""H3 Frankenstein Studio bridge regression: no GPU or model download required."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from fizgig.families.workbench import WorkbenchEngine
from fizgig.scripts import studio_worker
from fizgig.web import server


def test_frank_plan_requires_both_h3_files(monkeypatch, tmp_path):
    desc = SimpleNamespace(frankenstein_with=(
        ("minimax_dit", "MiniMax H3 video", "minimax_turbo_lora", 0.75),))
    video = tmp_path / "h3.safetensors"
    turbo = tmp_path / "turbo.safetensors"
    monkeypatch.setattr(server, "pref_path", lambda p, key: p.get(key, ""))
    prefs = {"minimax_dit": str(video), "minimax_turbo_lora": str(turbo)}
    kw, problems = server._frank_plan(desc, prefs)
    assert len(problems) == 2
    video.write_bytes(b"")
    kw, problems = server._frank_plan(desc, prefs)
    assert len(problems) == 1 and "Turbo" in problems[0]
    turbo.write_bytes(b"")
    kw, problems = server._frank_plan(desc, prefs)
    assert not problems
    assert kw["path"] == str(video)
    assert kw["lora"] == str(turbo)
    assert kw["strength"] == 0.75


class _Frank:
    def __init__(self):
        self.set_args = None
    def set(self, values):
        self.set_args = values


def _engine():
    eng = WorkbenchEngine.__new__(WorkbenchEngine)
    eng.primary_network = object()
    eng.frank = _Frank()
    eng.desc = SimpleNamespace(activation_cache=True)
    eng._turbo_enabled = True
    return eng


def _state(values):
    return SimpleNamespace(blocks={
        key: SimpleNamespace(primary_strength=value, primary_enabled=enabled)
        for key, (value, enabled) in values.items()})


def test_live_blend_weights_clamped_and_disabled_blocks_zero():
    eng = _engine()
    state = _state({"h3blk_1": (0.6, True), "h3blk_2": (2.1, True),
                    "h3blk_3": (-1.0, True), "h3blk_4": (0.9, False)})
    eng.apply_state(state)
    assert eng.frank.set_args == {
        "h3blk_1": 0.6, "h3blk_2": 1.0, "h3blk_3": 0.0, "h3blk_4": 0.0}
    assert not eng.same_as_baseline(state)
    assert not eng.turbo_preview  # replay of unchanged LoRA blocks is not valid for a blended base
    assert eng.same_as_baseline(_state({"h3blk_1": (0.0, True)}))


def test_small_card_auto_swap_requires_int8_streaming(monkeypatch):
    from fizgig.families import quant
    monkeypatch.setenv("FIZGIG_SIM_VRAM_GB", "12")
    monkeypatch.setattr(quant, "free_vram_gb", lambda: 12.0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda index: SimpleNamespace(total_memory=int(12 * 1024**3)))
    driver = SimpleNamespace(dit_gb=lambda path: 21.0, max_blocks_to_swap=lambda: 40)
    desc = SimpleNamespace(load_driver=lambda: driver, swap_ring_slots=2,
                           train_memory={"int8": (((0.25, 22.7), (1.0, 22.7)), 0.36)})
    assert studio_worker._frank_swap(desc, "h3.safetensors") == 34
    driver.max_blocks_to_swap = lambda: 20
    with pytest.raises(ValueError, match="Not enough VRAM"):
        studio_worker._frank_swap(desc, "h3.safetensors")


def test_only_h3_image_opted_in_for_live_blending():
    from fizgig.families.registry import get
    h3 = get("minimax_image")
    assert len(h3.frankenstein_with) == 1
    assert [x[0] for x in h3.frankenstein_presets] == ["✨Motion", "✨Motion, gentle"]
    assert not get("anima").frankenstein_with
    assert not get("krea2").frankenstein_with
    assert not get("moss_image").frankenstein_with
    assert not get("sdxl").frankenstein_with
