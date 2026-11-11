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
    """Issue #2: the gap stayed ~constant over 4 learning runs. The offset is now measured less today's figures for
    that run's own swap and size, applied in full (plus a little headroom), carried across swap and rank."""
    h = str(tmp_path / "vram_history.json")
    desc = SimpleNamespace(train_memory={"int8": (((0.25, 16.0), (1.0, 19.0)), 0.5), "nf4": (((0.25, 11.0), (1.0, 14.0)), 0.0)},
                           train_memory_small={}, batch_gb=(0.0, 0.0))
    k = vram_plan.key("krea2", "int8", "lora", "muon")

    def run(name, swap, measured, preview=None, key=k, extra=1.0, prec="int8"):
        base = quant.predicted_gb(desc, prec, swap, 1.0, 1)
        vram_plan.record(key, name, base + extra, measured, path=h, info=dict(
            base=base, extra=extra, swap=swap, mp=1.0, batch=1, **({"preview": preview} if preview else {})))

    assert vram_plan.correction(k, path=h, desc=desc) == (0.0, "")
    run("run1", 4, 14.3)                                   # predicted 19 - 2 + 1 = 18: measured 3.7 under it
    c, why = vram_plan.correction(k, path=h, desc=desc)
    assert abs(c - (-3.7 + vram_plan.LEARN_HEADROOM_GB)) < 1e-9 and "1 of past runs like this one" in why
    # the next plan, even at another swap, predicts measured + headroom for the same run, not 3.7 over
    pred = quant.predicted_gb(desc, "int8", 8, 1.0, 1) + 1.0 + c
    assert abs(pred - (14.3 - 8 * 0.5 + 4 * 0.5 + vram_plan.LEARN_HEADROOM_GB)) < 1e-9
    run("run2", 8, 13.0)                                   # a second run held more than the first: the larger wins
    assert abs(vram_plan.correction(k, path=h, desc=desc)[0] - (-3.0 + vram_plan.LEARN_HEADROOM_GB)) < 1e-9
    run("run3", 8, 16.5)                                   # a run over its prediction raises the plan in full
    assert abs(vram_plan.correction(k, path=h, desc=desc)[0] - 0.5) < 1e-9
    # retuned figures: the stored base is not reused, today's figures for that run are
    desc2 = SimpleNamespace(**{**vars(desc), "train_memory": {"int8": (((0.25, 15.0), (1.0, 18.0)), 0.5)}})
    assert abs(vram_plan.correction(k, path=h, desc=desc2)[0] - 1.5) < 1e-9
    h2 = str(tmp_path / "h2.json")
    h, h_keep = h2, h
    run("a", 0, 12.0)                                      # 7 GB under: bounded by CORRECTION_DOWN_GB
    assert vram_plan.correction(k, path=h, desc=desc)[0] == -vram_plan.CORRECTION_DOWN_GB
    # previews on: a run without a measured preview can't lower the plan; with one, the larger peak counts
    assert vram_plan.correction(k, path=h, desc=desc, previews=True) == (0.0, "")
    run("a", 0, 12.0, preview=17.0)
    c, why = vram_plan.correction(k, path=h, desc=desc, previews=True)
    assert abs(c - (-3.0 + vram_plan.LEARN_HEADROOM_GB)) < 1e-9 and "previews counted" in why
    # same model + precision, other optimizer: in full; other precision: half a lowering
    other = vram_plan.key("krea2", "int8", "lora", "adamw8bit")
    assert vram_plan.correction(other, path=h, desc=desc)[0] == -vram_plan.CORRECTION_DOWN_GB
    nf4 = vram_plan.key("krea2", "nf4", "lora", "muon")
    c, why = vram_plan.correction(nf4, path=h, desc=desc)
    assert abs(c - (-8.0 + vram_plan.LEARN_HEADROOM_GB) * vram_plan.OTHER_PRECISION_SHARE) < 1e-9 and "another precision" in why
    # entries from before (no base / extra) are not used: they were weighed against an uncorrected prediction
    h = str(tmp_path / "old.json")
    vram_plan.record(k, "old", 12.0, 8.0, path=h)
    d = vram_plan._read(h)
    d[k]["old"].pop("v")
    import json
    with open(h, "w") as f:
        json.dump(d, f)
    assert vram_plan.correction(k, path=h, desc=desc) == (0.0, "")
    for i in range(8):
        vram_plan.record(k, f"r{i}", 1.0, 1.0, path=h)
    assert len(vram_plan._read(h)[k]) == vram_plan.KEEP_RUNS
    assert h_keep


def test_nothing_fits_never_picks_an_nf4_that_cannot_fit():
    """Issue #2: 12 GB, rank 64 - Krea 2 and Fibo 2 planned NF4 with no swap and ran out of memory; Qwen and Z-Image at
    16 GB rank 64 dropped to NF4 where INT8 fits with a block or two swapped."""
    desc = SimpleNamespace(train_memory={"int8": (((0.25, 16.2), (1.0, 19.1)), 0.42),
                                         "nf4": (((0.25, 11.4), (1.0, 15.0)), 0.0)},
                           precisions=("int8", "nf4"), auto_precisions=("int8", "nf4"), batch_gb=(0.0, 0.0),
                           display_name="t", train_memory_small={})
    drv = SimpleNamespace(max_blocks_to_swap=lambda: 26)
    # 10.7 GB free, 4 GB over the figures: NF4 needs 19, INT8 with all 26 swapped 12.18 - neither fits, INT8 is nearer
    p, n, why = quant.plan(desc, drv, "auto", -1, free_gb=10.7, extra_gb=4.0)
    assert (p, n) == ("int8", 26) and "most blocks swapped" in why
    # an NF4 that needs less than the most-swapped INT8 still goes first
    tiny = SimpleNamespace(max_blocks_to_swap=lambda: 2)
    assert quant.plan(desc, tiny, "auto", -1, free_gb=10.7, extra_gb=4.0)[:2] == ("nf4", 0)
    # INT8 with a few blocks swapped beats NF4 (19.1 + 0.3 vs 18.0 budget: 4 blocks of 26)
    p, n, why = quant.plan(desc, drv, "auto", -1, free_gb=19.5, extra_gb=0.3)
    assert (p, n) == ("int8", 4) and "not nf4" in why
    # many blocks: NF4 that fits comfortably still wins (#216: heavy swap is far slower)
    assert quant.plan(desc, drv, "auto", -1, free_gb=17.0)[:2] == ("nf4", 0)


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
    optimizer's step and compile, the context LoRA (its real rank and layers, from the file) and a small card's
    reserve."""
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
    def lora_header(path, modules):         # proper lora_down / lora_up names, rank and layers from the header
        head, off = {}, 0
        for stem, o, i, r in modules:
            for key, shape in ((f"{stem}.lora_down.weight", (r, i)), (f"{stem}.lora_up.weight", (o, r)),
                               (f"{stem}.alpha", (1,))):
                n = 1
                for d in shape:
                    n *= d
                head[key] = {"dtype": "F32", "shape": list(shape), "data_offsets": [off, off + n * 4]}
                off += n * 4
        b = json.dumps(head).encode()
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(b)) + b)
        return path
    mods, n = [], 0
    for bi, group in enumerate([blk] * 28 + [fus] * 4 + [io]):
        for j, (o, i) in enumerate(group):
            tag = "blocks" if bi < 28 else ("txt" if bi < 32 else "io")
            idx = bi if bi < 28 else (bi - 28 if bi < 32 else j)
            mods.append((f"lora_unet_{tag}_{idx}_m{j}", o, i, 16))
            n += 1
    assert n == 264
    ctx = lora_header(str(tmp_path / "Furry_Parts.safetensors"), mods)
    drv = SimpleNamespace(max_blocks_to_swap=lambda: 26)
    picks = {}
    for args in ("magnitude_scale=3", "magnitude_scale=3 compile=False"):
        gb, why = vram_plan.extra_gb(KREA2, dit, "lora", 8, "polora", optimizer_args=args, ema=True,
                                     frozen_files=(ctx, None), card_gb=12.0, megapixels=0.25, pictures=1)
        assert "EMA" in why and "Furry_Parts.safetensors (rank 16, 264 layers)" in why
        assert "12 GB card reserve" in why
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


def test_frozen_lora_cost_comes_from_the_file(tmp_path, caplog):
    """The context LoRA term is the file's real tensors in bf16, its rank and layer count, plus a little for
    applying it; a missing or unreadable file falls back (or is skipped) and is logged."""
    import json
    import logging
    import struct

    def write_lora(path, modules, keyfn=None):
        keyfn = keyfn or (lambda stem, ab: f"{stem}.{ab}.weight")
        head, off = {}, 0
        for stem, i, o, r in modules:
            for ab, shape in (("lora_down", (r, i)), ("lora_up", (o, r))):
                n = 1
                for d in shape:
                    n *= d
                # stored as fp32 in the header: the plan still counts bf16 (how Fizgig loads a frozen file)
                head[keyfn(stem, ab)] = {"dtype": "F32", "shape": list(shape), "data_offsets": [off, off + n * 4]}
                off += n * 4
            head[f"{stem}.alpha"] = {"dtype": "F32", "shape": [1], "data_offsets": [off, off + 4]}
            off += 4
        b = json.dumps(head).encode()
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(b)) + b)
        return path

    p = write_lora(str(tmp_path / "parts.safetensors"),
                   [(f"lora_unet_blocks_{b}_attn_wq", 64, 128, 8) for b in range(3)]
                   + [(f"lora_unet_blocks_{b}_attn_wo", 128, 64, 8) for b in range(3)])
    info = vram_plan.inspect_frozen_lora(p)
    assert info["rank"] == 8 and info["layers"] == 6 and info["ok"] and not info["fallback"]
    assert abs(info["weights_gb"] - 6 * 8 * (64 + 128) * 2 / 1024 ** 3) < 1e-15
    label, gb = vram_plan.frozen_file_term(p, megapixels=0.25, pictures=1, bucket_step=16)
    assert label == "parts.safetensors (rank 8, 6 layers)"
    assert gb >= info["weights_gb"]
    act = gb - info["weights_gb"]
    assert 0 <= act < 0.01                               # a little (the busiest block's rank-r activations)
    _, gb2 = vram_plan.frozen_file_term(p, megapixels=1.0, pictures=4, bucket_step=16)
    assert gb2 > gb and (gb2 - info["weights_gb"]) < 0.05

    p2 = write_lora(str(tmp_path / "peft.safetensors"),
                    [("diffusion_model.layers.0.q", 32, 32, 4), ("diffusion_model.layers.1.q", 32, 32, 16)],
                    keyfn=lambda stem, ab: f"{stem}.{'lora_A' if ab == 'lora_down' else 'lora_B'}.weight")
    info2 = vram_plan.inspect_frozen_lora(p2)
    assert info2["rank"] == (4, 16) and info2["layers"] == 2
    assert "rank 4-16, 2 layers" in vram_plan.frozen_file_term(p2)[0]

    desc = SimpleNamespace(train_memory_recipe=None, train_memory_small={})
    _, why = vram_plan.extra_gb(desc, None, "lora", 8, "adamw8bit", frozen_files=(p, p2), megapixels=0.25)
    assert "parts.safetensors (rank 8, 6 layers)" in why
    assert "peft.safetensors (rank 4-16, 2 layers)" in why

    with caplog.at_level(logging.WARNING, logger="fizgig.families.vram_plan"):
        assert vram_plan.frozen_file_term(str(tmp_path / "nope.safetensors")) == ("", 0.0)
    assert "missing" in caplog.text

    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"\x00" * (2 * 1024 * 1024))
    with caplog.at_level(logging.WARNING, logger="fizgig.families.vram_plan"):
        label, w = vram_plan.frozen_file_term(str(junk))
    assert label == "junk.safetensors (from file size)"
    assert abs(w - 2 * 1024 * 1024 / 1024 ** 3) < 1e-12
    assert "couldn't read" in caplog.text


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
