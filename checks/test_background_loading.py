"""Background loading (training/prefetch.py) trains exactly what reading each step's cache as it starts does: a tiny
CPU family (test_save_stop's) with two dataset folders (the second's epoch shuffle lands mid-epoch, on the helper
thread) and a slow drive: the same batches in the same order, the same losses and LoRA, and the trainer's own draws
from the global random the same. Plus the setting's flag, the environment override and an early end of a pass."""
import os
import random
import sys
import threading
import time

import numpy as np
import pytest
import torch

import test_save_stop as tss

from fizgig.training import prefetch


@pytest.fixture(scope="module")
def two_folders(tmp_path_factory):
    from PIL import Image
    d = tmp_path_factory.mktemp("bgload")
    rng = np.random.default_rng(1)
    parts = []
    for f in ("a", "b"):
        img = d / f
        img.mkdir()
        for i in range(4):
            h = 64 if i % 2 == 0 else 96
            Image.fromarray(rng.integers(0, 255, (h, 64, 3), dtype=np.uint8)).save(img / f"{f}{i}.png")
            (img / f"{f}{i}.txt").write_text(f"a photo {f} {i}")
        parts.append(f'[[datasets]]\nimage_directory = "{img.as_posix()}"\n'
                     f'cache_directory = "{(d / ("cache_" + f)).as_posix()}"\nnum_repeats = 1\n')
    cfg = d / "ds.toml"
    cfg.write_text('[general]\nresolution = [64, 64]\ncaption_extension = ".txt"\nbatch_size = 2\n'
                   'enable_bucket = true\n\n' + "\n".join(parts))
    return d, str(cfg)


@pytest.fixture
def tiny2(monkeypatch, two_folders):
    import dataclasses
    import fizgig.families.cache as C
    import fizgig.families.train as T
    from fizgig.families import registry
    desc = dataclasses.replace(registry.FAMILIES["zimage"], driver="test_save_stop:TinyDriver", batch_training=True)
    monkeypatch.setitem(registry.FAMILIES, "zimage", desc)
    monkeypatch.setattr(T, "torch", tss._CpuTorch("torch"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a, **k: 0)
    d, cfg = two_folders
    if not (d / "cache_a").exists():
        for stage in ("latents", "text"):
            monkeypatch.setattr(sys, "argv", ["cache", "--family", "zimage", "--stage", stage, "--dataset_config",
                                              cfg, "--model", "x", "--device", "cpu"])
            C.main()
    return T, cfg


_REAL = {"load": None, "loss": None}       # the unpatched ones (a second run must not wrap the first's patch)


def _train(T, cfg, out, monkeypatch, on):
    """3 epochs on a slow drive (each cache read 15 ms); what each step trained on, the trainer's own global random
    draws, and which threads read the cache."""
    from fizgig.dataset import image_dataset
    seen, draws, readers = [], [], set()
    real_load, real_loss = _REAL["load"] or image_dataset.load_file, _REAL["loss"] or tss.TinyDriver.training_loss
    _REAL.update(load=real_load, loss=real_loss)

    def slow_load(path, *a, **k):
        readers.add(threading.current_thread().name)
        time.sleep(0.015)
        return real_load(path, *a, **k)

    def loss(self, dit, latents, cond, generator, **kw):
        if torch.is_grad_enabled():
            seen.append(float(latents.sum()))
            draws.append(random.random())          # e.g. MiniMax's caption dropout draws from the global random
        return real_loss(self, dit, latents, cond, generator, **kw)
    random.seed(123)                           # the same start for both: the dataset's own seed comes from it
    torch.manual_seed(123)
    monkeypatch.setattr(image_dataset, "load_file", slow_load)
    monkeypatch.setattr(tss.TinyDriver, "training_loss", loss)
    T.train_family("zimage", "x", cfg, str(out), "t", network_dim=4, network_alpha=4, learning_rate=1e-3,
                   lr_scheduler="cosine", max_train_epochs=3, save_state=False, compile_blocks="off", seed=7,
                   background_loading=on)
    from safetensors.torch import load_file
    return seen, draws, readers, [r["l"] for r in tss._losses(out)], load_file(str(out / "t.safetensors"))


def test_same_training_with_and_without(tiny2, tmp_path, monkeypatch):
    T, cfg = tiny2
    monkeypatch.delenv("FIZGIG_BACKGROUND_LOADING", raising=False)
    off = _train(T, cfg, tmp_path / "off", monkeypatch, False)
    on = _train(T, cfg, tmp_path / "on", monkeypatch, True)
    assert len(off[3]) == 12 and on[0] == off[0]                 # the same batches in the same order
    assert on[1] == off[1]                                        # the trainer's global random draws
    assert on[3] == off[3]                                        # the same losses
    assert set(on[4]) == set(off[4]) and all(torch.equal(on[4][k], off[4][k]) for k in off[4])
    assert "fizgig-background-loading" in on[2] and "fizgig-background-loading" not in off[2]


def test_setting_flag_and_env(monkeypatch):
    from fizgig.families import launch
    from fizgig.families.train import setup_parser
    assert launch.RECLAIM_PREF_DEFAULTS["background_loading"] == "1"
    assert launch.background_loading_flags({}) == []
    assert launch.background_loading_flags({"background_loading": "0"}) == ["--no-background_loading"]
    assert launch.clean_reclaim_prefs({"background_loading": False}) == {"background_loading": "0"}
    req = ["--family", "zimage", "--dit", "x", "--dataset_config", "c", "--output_dir", "o", "--output_name", "n"]
    assert setup_parser().parse_known_args(req)[0].background_loading is True
    assert setup_parser().parse_known_args(req + ["--no-background_loading"])[0].background_loading is False
    for env, setting, want in (("", True, True), ("", False, False), ("0", True, False), ("1", False, True)):
        monkeypatch.setenv("FIZGIG_BACKGROUND_LOADING", env)
        assert prefetch.wanted(setting) is want
    js = open(os.path.join(tss.ROOT, "src", "fizgig", "web", "static", "app.js"), encoding="utf-8").read()
    assert "prefBackgroundLoading" in js and "Background loading" in js


def test_a_pass_ended_early_stops_its_reader_and_errors_come_through():
    class Slow:
        def __init__(self, n, bad=None):
            self.n, self.bad = n, bad

        def __len__(self):
            return self.n

        def __iter__(self):
            for i in range(self.n):
                if i == self.bad:
                    raise OSError("the drive went away")
                time.sleep(0.01)
                yield i
    p = prefetch.Prefetch(Slow(50))
    for i in p:
        if i == 3:
            break
    time.sleep(0.1)
    assert not any(t.name == "fizgig-background-loading" for t in threading.enumerate())
    assert list(prefetch.Prefetch(Slow(5))) == [0, 1, 2, 3, 4] and len(prefetch.Prefetch(Slow(5))) == 5
    got = []
    with pytest.raises(OSError, match="went away"):
        for i in prefetch.Prefetch(Slow(6, bad=4)):
            got.append(i)
    assert got == [0, 1, 2, 3]                                    # every batch before the failed read, then it
