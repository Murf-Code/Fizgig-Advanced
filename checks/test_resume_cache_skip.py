"""A resume with its cache all there caches nothing, and a cache pass over a cache that is all there encodes nothing:

- the resume's uncached check reads Start's own hold-out list (the file on disk is still the one the pause emptied,
  so read from there the held-out pictures counted as "no cache" and every resume cached again);
- the text stage keeps a caption's cache (--skip_existing) only when it was encoded from that caption, so an edited
  caption is encoded again;
- the latent stage logs how many pictures were already cached and how many were encoded, and neither stage loads its
  model when nothing needs encoding (the hold-out's own pass too)."""
import dataclasses
import json
import os
import sys

import pytest

from test_save_stop import TinyDriver
from test_validation_holdout import _pics, _toml
from fizgig.families import registry
from fizgig.training import holdout

LOADS = {"vae": 0, "te": 0, "img": 0, "txt": 0}


class CountingDriver(TinyDriver):
    def load_vae(self, path, device):
        LOADS["vae"] += 1
        return super().load_vae(path, device)

    def encode_images(self, vae, images):
        LOADS["img"] += len(images)
        return super().encode_images(vae, images)

    def load_text_encoder(self, path, device):
        LOADS["te"] += 1
        return super().load_text_encoder(path, device)

    def encode_text(self, te, captions):
        LOADS["txt"] += len(captions)
        return super().encode_text(te, captions)


@pytest.fixture
def counting(monkeypatch):
    desc = dataclasses.replace(registry.FAMILIES["zimage"], driver=f"{__name__}:CountingDriver")
    monkeypatch.setitem(registry.FAMILIES, "zimage", desc)
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    return desc


def _pass(cfg, model):
    import fizgig.families.cache as C
    for k in LOADS:
        LOADS[k] = 0
    for stage in ("latents", "text"):
        sys.argv = ["cache", "--family", "zimage", "--stage", stage, "--dataset_config", cfg, "--model", model,
                    "--device", "cpu", "--holdout_cache", "--skip_existing"]
        C.main()
    return dict(LOADS)


def test_a_cache_that_is_all_there_encodes_nothing_and_loads_no_model(counting, tmp_path, caplog):
    caplog.set_level("INFO")
    _pics(tmp_path / "img", 10)
    model = tmp_path / "te.safetensors"
    model.write_bytes(b"0" * 10)
    cfg = tmp_path / "ds.toml"
    hfile = holdout.path_for(cfg)
    held = [str(tmp_path / "img" / "p3.png")]
    holdout.write(hfile, holdout.content(held, ["p3.png"], "auto"))
    cfg.write_text(_toml(tmp_path, f'holdout_file = "{hfile.replace(os.sep, "/")}"\n'))
    first = _pass(str(cfg), str(model))
    assert first == {"vae": 2, "te": 2, "img": 10, "txt": 10}           # 9 training + 1 held out, each pass its own
    assert "[cache] dataset [0]: 0 already cached, 9 encoded" in caplog.text
    caplog.clear()
    again = _pass(str(cfg), str(model))                                  # what a resume's check pass does
    assert again == {"vae": 0, "te": 0, "img": 0, "txt": 0}
    assert "[cache] dataset [0]: 9 already cached, 0 encoded" in caplog.text
    assert "[cache] dataset [0]: 1 already cached, 0 encoded" in caplog.text      # the hold-out's own pass
    # an edited caption is encoded again (and only it); its latent stays
    (tmp_path / "img" / "p5.txt").write_text("a photo number 5, edited")
    assert _pass(str(cfg), str(model)) == {"vae": 0, "te": 1, "img": 0, "txt": 1}
    from safetensors import safe_open
    te = [f for f in os.listdir(tmp_path / "cache") if f.startswith("p5_") and f.endswith("_te.safetensors")][0]
    with safe_open(str(tmp_path / "cache" / te), framework="pt") as f:
        assert f.metadata()["caption1"] == "a photo number 5, edited"
    # another text encoder file: every caption is encoded again (a cache records the encoder it came from)
    model.write_bytes(b"0" * 11)
    assert _pass(str(cfg), str(model))["txt"] == 10


def test_legacy_text_caches_without_an_encoder_stamp_are_reused(tmp_path):
    import torch
    from safetensors.torch import save_file
    from fizgig.families.cache import _cached_matches

    path = str(tmp_path / "legacy.safetensors")
    save_file({"cond__test": torch.ones(1)}, path, metadata={"caption1": "same caption"})
    assert _cached_matches(path, "same caption", "", "encoder.safetensors:123")
    assert not _cached_matches(path, "edited caption", "", "encoder.safetensors:123")

    save_file({"cond__test": torch.ones(1)}, path,
              metadata={"caption1": "same caption", "text_encoder": "older-encoder:123"})
    assert not _cached_matches(path, "same caption", "", "new-encoder:123")


def test_the_text_stage_skips_by_caption(tmp_path):
    import argparse
    from fizgig.scripts.cache_text import process_batches

    class It:
        def __init__(self, n, cap):
            self.text_encoder_output_cache_path, self.caption = str(tmp_path / f"{n}_te.safetensors"), cap

    class DS:
        batch_size = 1

        def __init__(self, items):
            self.items = items

        def retrieve_text_encoder_output_cache_batches(self, n):
            for it in self.items:
                yield [it]

        def __len__(self):
            return len(self.items)
    items = [It(i, f"cap {i}") for i in range(4)]
    files = [{os.path.normpath(it.text_encoder_output_cache_path) for it in items[:3]}]   # 3 cached, 1 not

    def run(**kw):
        sent = []
        process_batches(argparse.Namespace(num_workers=1, batch_size=8, **kw), [DS(items)], files, [set()],
                        sent.extend)
        return sorted(it.caption for it in sent)
    assert run(skip_existing=False) == ["cap 0", "cap 1", "cap 2", "cap 3"]       # no skip: all
    assert run(skip_existing=True) == ["cap 3"]                                    # existence only (Klein's script)
    assert run(skip_existing=True, text_matches=lambda p, c: c != "cap 1") == ["cap 1", "cap 3"]


class OwnLayoutDriver(CountingDriver):
    SEEN = []

    def cache_stage(self, stage, datasets, args, device, aux):
        OwnLayoutDriver.SEEN.append((stage, args.skip_existing))
        return True


def test_a_family_text_layout_keeps_the_skip_it_had(monkeypatch, tmp_path):
    """--skip_existing on the text stage is the shared, caption-checked one: a family that caches text in its own
    layout (H3: existence only, reference slots rewritten) is handed the flag off, as before."""
    desc = dataclasses.replace(registry.FAMILIES["zimage"], driver=f"{__name__}:OwnLayoutDriver")
    monkeypatch.setitem(registry.FAMILIES, "zimage", desc)
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    _pics(tmp_path / "img", 6)
    cfg = tmp_path / "ds.toml"
    cfg.write_text(_toml(tmp_path))
    OwnLayoutDriver.SEEN.clear()
    import fizgig.families.cache as C
    for stage in ("latents", "text"):
        sys.argv = ["cache", "--family", "zimage", "--stage", stage, "--dataset_config", str(cfg), "--model", "x",
                    "--device", "cpu", "--skip_existing"]
        C.main()
    assert OwnLayoutDriver.SEEN == [("latents", True), ("text", False)]


def test_cache_commands_skip_existing_for_both_stages():
    from fizgig.families import launch
    from fizgig.families.registry import get
    krea = get("krea2")
    inp = {"python": "py", "repo_dir": "r", "DATASET_CONFIG": "d.toml", "models": {}}
    for stage in ("latents", "text"):
        assert "--skip_existing" in launch.cache_command(krea, inp, stage)


def _krea_resume(tmp_path, monkeypatch):
    """A Krea 2 resume whose cache is all there: training pictures in the dataset's cache folder, the hold-out's in
    its _fizgig_holdout/, the restore point naming the hold-out, and the hold-out file on disk emptied by the pause."""
    from fizgig.families import form, launch
    from fizgig.families.registry import get
    krea = get("krea2")
    img = tmp_path / "imgs"
    _pics(img, 10)
    models = {}
    for mf in krea.model_files:
        if mf.required or mf.pref_key == krea.training_adapter:
            f = tmp_path / f"{mf.pref_key}.safetensors"
            f.write_bytes(b"0")
            models[mf.pref_key] = str(f)
    cdir = launch.cache_dir_for(str(tmp_path / "cache"), str(img))
    os.makedirs(os.path.join(cdir, holdout.HOLDOUT_CACHE))
    held = ["p2.png", "p7.png"]
    for i in range(10):
        d = os.path.join(cdir, holdout.HOLDOUT_CACHE) if f"p{i}.png" in held else cdir
        for name in (f"p{i}_0064x0064_{krea.arch_id}.safetensors", f"p{i}_{krea.arch_id}_te.safetensors"):
            open(os.path.join(d, name), "wb").close()
    state = tmp_path / "out" / "t-000001-state"
    state.mkdir(parents=True)
    paths = [str(img / n) for n in held]
    (state / "training_state.json").write_text(json.dumps({"validation_holdout": {"paths": paths, "names": held}}))
    hfile = holdout.path_for(tmp_path / "dataset.toml")
    holdout.write(hfile, holdout.content(paths, held, "auto"))
    holdout.give_back(hfile)                                            # Save & stop emptied it
    assert holdout.read(hfile)["paths"] == []

    def plan():
        inp = form.inputs(krea, {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"),
                                 "DATASET_CONFIG": str(tmp_path / "dataset.toml"), "RESUME_TRAINING": str(state)},
                          models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder=str(img),
                          cache_root=str(tmp_path / "cache"), samples_dir=str(tmp_path / "out" / "sample"))
        p = launch.plan(krea, inp)
        assert p.problems == []
        return p
    return plan, cdir


def test_a_resume_with_its_cache_all_there_caches_nothing(tmp_path, monkeypatch):
    plan, cdir = _krea_resume(tmp_path, monkeypatch)
    p = plan()
    assert any("[validation] Hold-out: 2 picture(s)" in ln for ln in p.console)
    assert not any("have no cache" in ln for ln in p.console)
    assert not any(s.name.startswith(("Cache Preparation", "Text Encoder Caching")) for s in p.stages)
    # a training picture's cache deleted by hand: that one is cached again before the run resumes
    os.remove(os.path.join(cdir, [f for f in os.listdir(cdir) if f.startswith("p4_") and "_te" not in f][0]))
    p = plan()
    assert any("[cache] 1 picture(s) have no cache" in ln for ln in p.console)
    assert any(s.name == "Cache Preparation" for s in p.stages)
