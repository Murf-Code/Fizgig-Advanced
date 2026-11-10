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
    assert gb > 0 and "rank 128 adamw vs" in why
    small = vram_plan.extra_gb(desc, _model(tmp_path), "lora", 8, "adamw8bit")[0]
    assert vram_plan.ADAPTER_FLOOR_GB <= small < 0                    # a smaller LoRA plans less, down to the floor
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


def test_chris_4070ti_krea2_config_swaps_at_least_20(tmp_path):
    """Chris's 12 GB 4070 Ti, Nov 2026: Krea 2 at 0.25 MP, int8, rank 8 PoLoRA, EMA, a context LoRA. Auto swapped 17
    and spilled (22 s a step uncompiled, 66 s compiled); 20 ran clean at 3.9 s. The plan now counts the EMA, the
    optimizer's step and compile, the context LoRA and a small card's reserve."""
    import json
    import struct
    from fizgig.families.krea2 import KREA2

    def header(path, shapes):               # a header-only .safetensors: the planner reads shapes, never data
        head, off = {}, 0
        for k, (o, i) in enumerate(shapes):
            head[f"m.{k}.weight"] = {"dtype": "BF16", "shape": [o, i], "data_offsets": [off, off + o * i * 2]}
            off += o * i * 2
        b = json.dumps(head).encode()
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(b)) + b)
        return path
    blk = [(6144, 6144), (1536, 6144), (1536, 6144), (6144, 6144), (6144, 6144), (16384, 6144), (16384, 6144),
           (6144, 16384)]
    fus = [(2560, 2560)] * 5 + [(6912, 2560), (6912, 2560), (2560, 6912)]
    io = [(6144, 64), (6144, 256), (6144, 6144), (6144, 2560), (6144, 6144), (36864, 6144), (64, 6144), (6144, 2560)]
    shapes = blk * 28 + fus * 4 + io                                   # Krea 2's 264 LoRA Linears (29.4 M at rank 8)
    dit = header(str(tmp_path / "krea2.safetensors"), shapes)
    ctx = header(str(tmp_path / "ctx.safetensors"), [s for o, i in shapes for s in ((16, i), (o, 16))])
    drv = SimpleNamespace(max_blocks_to_swap=lambda: 26)
    picks = {}
    for args in ("magnitude_scale=3", "magnitude_scale=3 compile=False"):
        gb, why = vram_plan.extra_gb(KREA2, dit, "lora", 8, "polora", optimizer_args=args, ema=True,
                                     frozen_files=(ctx, None), card_gb=12.0)
        assert "EMA" in why and "ctx.safetensors" in why and "12 GB card reserve" in why
        assert ("polora compile" in why) == ("compile=False" not in args)
        picks[args] = quant.plan(KREA2, drv, "auto", -1, free_gb=10.8, megapixels=0.25, extra_gb=gb)[:2]
    assert picks["magnitude_scale=3"] in (("int8", 20), ("int8", 21))
    assert picks["magnitude_scale=3 compile=False"] == ("int8", 20)
    assert quant.plan(KREA2, drv, "auto", -1, free_gb=10.8, megapixels=0.25)[1] == 17      # before: too few
    # any optimizer, by its own state and step: AdamW's fp32 moments and Prodigy plan more than AdamW8bit
    g = {o: vram_plan.extra_gb(KREA2, dit, "lora", 16, o)[0] for o in ("adamw8bit", "adamw", "prodigy", "muon")}
    assert g["adamw8bit"] == 0 and g["prodigy"] > g["adamw"] > g["muon"] > 0
    # the user's own Blocks Swap is what the reason says
    why = quant.plan(KREA2, drv, "auto", 20, free_gb=10.8, megapixels=0.25)[2]
    assert why == "int8 with your Blocks Swap 20; Auto would swap 17 to fit 10.8 GB"


def test_slow_steps_warn_once_and_the_warmup_line_tells_the_truth():
    from fizgig.training.slow_step import SlowStepWatch, warmup_line
    w = SlowStepWatch(17, 26)
    w.set_pass(64.1, 60)                                               # his baseline: 15 images x 4 levels
    assert all(w.observe(3.9) is None for _ in range(20))              # swap 20: fine
    w = SlowStepWatch(17, 26)
    w.set_pass(64.1, 60)
    msgs = [w.observe(66.0) for _ in range(20)]
    said = [m for m in msgs if m]
    assert len(said) == 1 and "Shared GPU memory" in said[0] and "try 20" in said[0]
    assert warmup_line(66.02, 66.0) == "[warm-up] Not faster yet: 66.02 s/step now, 66.0 s at the start."
    assert warmup_line(4.0, 9.0).startswith("[warm-up] Still speeding up: 4.00 s/step now")
