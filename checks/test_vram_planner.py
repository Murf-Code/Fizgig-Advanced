"""VRAM planning that knows the rank and optimizer, learns from measured peaks, and leaves a configurable margin
(families/vram_plan.py, quant.plan's extra_gb, Preferences > Training memory > Safety margin)."""
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from fizgig.families import launch, quant, vram_plan
from test_save_stop import _run, data, tiny  # noqa: F401 - the fixtures


def _model(tmp_path):
    p = str(tmp_path / "dit.safetensors")
    save_file({f"blocks.{b}.{n}.weight": torch.zeros(o, i) for b in range(4)
               for n, o, i in (("qkv", 3 * 512, 512), ("proj", 512, 512), ("up", 2048, 512), ("down", 512, 2048))}
              | {"blocks.0.norm.weight": torch.zeros(512)}, p)
    return p


def test_rank_and_optimizer_size_the_lora(tmp_path):
    shapes = vram_plan.linear_shapes(_model(tmp_path))
    assert len(shapes) == 16 and (1536, 512) in shapes                # 2D weights only
    a = vram_plan.adapter_gb(shapes, 16, "adamw8bit")
    assert vram_plan.adapter_gb(shapes, 64, "adamw8bit") == a * 4    # linear in rank
    assert vram_plan.adapter_gb(shapes, 16, "adamw") > a > vram_plan.adapter_gb(shapes, 16, "lion8bit")
    assert vram_plan.adapter_gb(shapes, 16, "soap") > vram_plan.adapter_gb(shapes, 16, "adamw") + 0.002  # its matrices
    for opt in ("muon", "lora_tsd", "polora", "prodigy", "cmuon"):
        assert opt in vram_plan.STATE_BYTES
    desc = SimpleNamespace(train_memory_recipe=None)
    gb, why = vram_plan.extra_gb(desc, _model(tmp_path), "lora", 128, "adamw")
    assert gb > 0 and "rank 128 with adamw" in why
    assert vram_plan.extra_gb(desc, _model(tmp_path), "lora", 8, "adamw8bit")[0] == 0    # never less than the figures
    assert vram_plan.extra_gb(desc, _model(tmp_path), "lokr", 128, "adamw")[0] == 0


def test_the_plan_swaps_more_for_extra_and_a_bigger_margin():
    desc = SimpleNamespace(train_memory={"bf16": (((0.25, 10.0), (1.0, 10.0)), 0.5)}, precisions=("bf16",),
                           auto_precisions=(), batch_gb=(0.0, 0.0), display_name="t", train_memory_small={})
    drv = SimpleNamespace(max_blocks_to_swap=lambda: 30)
    base = quant.plan(desc, drv, "bf16", -1, free_gb=12.0)[1]
    assert base == 0
    assert quant.plan(desc, drv, "bf16", -1, free_gb=12.0, extra_gb=2.0)[1] == 3       # 12 - 1.5 = 10.5 vs 12
    assert quant.plan(desc, drv, "bf16", -1, free_gb=12.0, extra_gb={"bf16": 2.0})[1] == 3
    assert quant.plan(desc, drv, "bf16", -1, free_gb=12.0, margin_gb=3.0)[1] == 2
    assert quant.predicted_gb(desc, "bf16", 4, 0.5) == 8.0 and quant.predicted_gb(desc, "int8", 0) is None


def test_measured_peaks_correct_the_estimate(tmp_path):
    h = str(tmp_path / "vram_history.json")
    k = vram_plan.key("krea2", "int8", "lora", "muon")
    assert vram_plan.correction(k, path=h) == (0.0, "")
    vram_plan.record(k, "run1", 10.0, 10.8, path=h)
    vram_plan.record(k, "run2", 10.0, 10.3, path=h)
    c, why = vram_plan.correction(k, path=h)
    assert abs(c - 0.8) < 1e-9 and "2 of past runs like this one" in why
    other = vram_plan.key("krea2", "int8", "lora", "adamw8bit")          # same model + precision, no runs of its own
    assert abs(vram_plan.correction(other, path=h)[0] - 0.8) < 1e-9
    vram_plan.record(k, "run1", 12.0, 8.0, path=h)                      # a re-run replaces its own entry
    vram_plan.record(k, "run2", 12.0, 8.0, path=h)
    assert vram_plan.correction(k, path=h)[0] == -vram_plan.CORRECTION_DOWN_GB    # lowered, but only so far
    for i in range(8):
        vram_plan.record(k, f"r{i}", 1.0, 1.0, path=h)
    assert len(vram_plan._read(h)[k]) == vram_plan.KEEP_RUNS


def test_the_margin_is_a_preference_passed_to_the_trainer(tiny, tmp_path):
    assert launch.RECLAIM_PREF_DEFAULTS["vram_margin_gb"] == "1.5"
    assert launch.vram_margin_flags({}) == [] and launch.vram_margin_flags({"vram_margin_gb": "1.5"}) == []
    assert launch.vram_margin_flags({"vram_margin_gb": "3"}) == ["--vram_margin_gb", "3"]
    assert launch.clean_reclaim_prefs({"vram_margin_gb": ""})["vram_margin_gb"] == "1.5"
    assert launch.clean_reclaim_prefs({"vram_margin_gb": "0"})["vram_margin_gb"] == "0"
    for bad in ("lots", "-1", "20"):
        try:
            launch.clean_reclaim_prefs({"vram_margin_gb": bad})
            raise AssertionError(bad)
        except ValueError:
            pass
    T, cfg = tiny                                                        # the trainer takes it (a real tiny run)
    assert _run(T, cfg, tmp_path / "m", vram_margin_gb=2.5, max_train_epochs=1, save_state=False) == "done"
