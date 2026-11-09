"""Validation hold-out (training/holdout.py): on by default, a few pictures are set aside before the cache is made -
never cached as training pictures, never trained - and validated on for the whole run; the run's end (Save & stop
too) gives them back, and a resume sets aside the same ones. A tiny CPU family caches and trains for real
(test_save_stop's), plus Start's pick (auto, your picked images, too few pictures)."""
import json
import os
import sys

import numpy as np
import pytest

from test_save_stop import _losses, _run, data, tiny  # noqa: F401 - the fixtures
from fizgig.training import holdout


def _pics(d, n, sizes=(64, 96)):
    from PIL import Image
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (sizes[i % 2], 64, 3), dtype=np.uint8)).save(d / f"p{i}.png")
        (d / f"p{i}.txt").write_text(f"a photo number {i}")


def _toml(d, extra=""):
    return (f'[general]\nresolution = [64, 64]\ncaption_extension = ".txt"\nbatch_size = 2\nenable_bucket = true\n'
            f'{extra}\n[[datasets]]\nimage_directory = "{(d / "img").as_posix()}"\n'
            f'cache_directory = "{(d / "cache").as_posix()}"\nnum_repeats = 1\n')


def _cache(cfg):
    import fizgig.families.cache as C
    for stage in ("latents", "text"):
        sys.argv = ["cache", "--family", "zimage", "--stage", stage, "--dataset_config", cfg, "--model", "x",
                    "--device", "cpu", "--holdout_cache"]
        C.main()


def _watch(monkeypatch):
    """The item keys every training batch carries (the validator sees each one), not its own prefill's."""
    from fizgig.training import val_loss
    trained, real_offer, real_prefill = set(), val_loss.FixedValidation.offer, val_loss.FixedValidation.prefill

    def offer(self, batch):
        if not getattr(self, "_pre", False):
            trained.update(os.path.splitext(os.path.basename(str(k)))[0] for k in batch.get("item_keys") or [])
        return real_offer(self, batch)

    def prefill(self, *a, **k):
        self._pre = True
        try:
            return real_prefill(self, *a, **k)
        finally:
            self._pre = False
    monkeypatch.setattr(val_loss.FixedValidation, "offer", offer)
    monkeypatch.setattr(val_loss.FixedValidation, "prefill", prefill)
    return trained


def test_held_out_before_the_cache_given_back_on_stop_and_the_same_on_resume(tiny, tmp_path, caplog, monkeypatch):
    T, _cfg = tiny
    from fizgig.families import launch
    from fizgig.families.registry import FAMILIES
    caplog.set_level("INFO")
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    _pics(tmp_path / "img", 10)                          # two buckets of 5: 3 + 3 = 6 steps an epoch
    cfg = tmp_path / "ds.toml"
    hfile = holdout.path_for(cfg)
    base = _toml(tmp_path)
    paths, names, how, _why = launch.holdout_pick(FAMILIES["zimage"], {}, base)        # Start's automatic pick
    assert how == "auto" and len(paths) == 1                                            # a tenth of 10
    stem = os.path.splitext(names[0])[0]
    holdout.write(hfile, holdout.content(paths, names, how))
    cfg.write_text(_toml(tmp_path, f'holdout_file = "{hfile.replace(os.sep, "/")}"\n'))
    _cache(str(cfg))
    cached = os.listdir(tmp_path / "cache")
    assert not any(f.startswith(stem + "_") for f in cached)                             # never a training cache
    assert {f.split("_")[0] for f in os.listdir(tmp_path / "cache" / "_fizgig_holdout")} == {stem}   # validation's
    trained = _watch(monkeypatch)
    out = tmp_path / "run"
    assert _run(T, str(cfg), out, press_at=5 + 2, max_train_epochs=2) == "stopped"       # Save & stop in epoch 2
    assert f"hold-out on: 1 picture(s) set aside for the whole run, never trained - " in caplog.text
    assert len(trained) == 9 and stem not in trained
    assert any(ln.get("k") == "val" for ln in map(json.loads, open(out / ".fizgig_loss_t.jsonl")))
    assert holdout.read(hfile)["paths"] == [] and holdout.read(hfile)["returned"] == names   # back in the dataset
    assert "set-aside picture(s) are back in the dataset" in caplog.text
    sd = out / "t-000002-state"
    assert holdout.from_state(str(sd)) == {"paths": paths, "names": names}
    # Resume: exactly the same picture set aside again (Start reads it from the restore point)
    again = launch.holdout_pick(FAMILIES["zimage"], {"RESUME_TRAINING": str(sd), "VALIDATION_PICKED": ["p0.png"]},
                                base)
    assert again[:3] == (paths, names, "resume")
    holdout.write(hfile, holdout.content(*again[:3]))
    trained.clear()
    assert _run(T, str(cfg), out, resume=str(sd), max_train_epochs=2) == "done"
    assert trained and stem not in trained and len(_losses(out)) == 2 * 5
    assert holdout.read(hfile)["paths"] == []


def test_start_picks_your_images_or_about_a_tenth_and_says_so(tmp_path):
    from fizgig.families import form, launch
    from fizgig.families.registry import get as get_family
    krea = get_family("krea2")
    _pics(tmp_path / "imgs", 8)
    models = {}
    for mf in krea.model_files:
        if mf.required or mf.pref_key == krea.training_adapter:
            f = tmp_path / f"{mf.pref_key}.safetensors"
            f.write_bytes(b"0")
            models[mf.pref_key] = str(f)

    def plan(**kw):
        inp = form.inputs(krea, {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"),
                                 "DATASET_CONFIG": str(tmp_path / "dataset.toml"), **kw},
                          models=models, python=sys.executable, repo_dir=str(tmp_path),
                          image_folder=str(tmp_path / "imgs"), cache_root=str(tmp_path / "cache"),
                          samples_dir=str(tmp_path / "out" / "sample"))
        p = launch.plan(krea, inp)
        assert p.problems == []
        files = dict(p.files)
        return p, json.loads(files[str(tmp_path / "dataset_holdout.json")])
    assert form.DEFAULTS["VALIDATION_HOLDOUT"] is True and form.DEFAULTS["VALIDATION_LOSS"] is False
    p, h = plan()                                                   # on by default, validation loss off
    assert h["how"] == "auto" and len(h["paths"]) == 1
    assert "holdout_file = " in p.dataset_toml
    cache = [s.cmd for s in p.stages if s.name == "Cache Preparation"][0]
    train = p.stages[-1].cmd
    assert "--holdout_cache" in cache and "--validation_loss" in train and "--validation_early_stop" not in train
    assert any(ln.startswith("[validation] Hold-out: 1 picture(s) set aside") for ln in p.console)
    p, h = plan(VALIDATION_PICKED=["p3.png", "p6.png"])             # your picked images are the hold-out
    assert h["how"] == "picked" and h["names"] == ["p3.png", "p6.png"]
    p, h = plan(VALIDATION_HOLDOUT=False)                           # off: an empty list, nothing in the TOML
    assert h["paths"] == [] and "holdout_file" not in p.dataset_toml and "--validation_loss" not in p.stages[-1].cmd
    for f in ("p5", "p6", "p7", "p4"):                             # 4 pictures: too few to set any aside
        os.remove(tmp_path / "imgs" / f"{f}.png")
    p, h = plan()
    assert h["paths"] == [] and any("Hold-out off for this run: only 4 picture(s)" in ln for ln in p.console)


def test_never_every_picture_and_the_form():
    from fizgig.families import form
    from fizgig.families.registry import FAMILIES
    cands = [("", [f"/d/p{i}.png" for i in range(6)])]
    assert holdout.choose(cands, 4, [f"p{i}.png" for i in range(6)])[0] == []
    assert len(holdout.choose([("", [f"/d/p{i}.png" for i in range(400)])], 4)[0]) == 4    # capped
    desc = FAMILIES["zimage"]
    assert "VALIDATION_HOLDOUT" not in form.state(desc, {**form.DEFAULTS})["disabled"]     # no longer needs val loss


@pytest.mark.parametrize("n", [1])
def test_a_list_left_by_a_killed_run_holds_nothing_after_give_back(tmp_path, n):
    f = tmp_path / "x_holdout.json"
    holdout.write(str(f), holdout.content([str(tmp_path / "p.png")], ["p.png"], "auto"))
    assert holdout.held_stems(str(f), str(tmp_path)) == {"p"}
    assert holdout.give_back(str(f)) == ["p.png"] and holdout.held_stems(str(f), str(tmp_path)) == set()
