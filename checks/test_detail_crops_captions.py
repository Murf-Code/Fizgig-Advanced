"""Detail crops' captions: Image Prep makes crops only at the picked size; "Copy the picture's caption to its crops"
works for them (and at Start for other sizes); each size is captioned from its own crop, never copied from another
size; and Start captions them with the Captions tab's model."""
import json
import os
import sys
import types

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from fizgig.dataset import detail_crops as dc  # noqa: E402

Q = lambda *a: None                                                  # noqa: E731


def _folder(tmp_path, captions=True):
    d = tmp_path / "pics"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        Image.fromarray(rng.integers(0, 255, (900, 900, 3), dtype=np.uint8)).save(d / f"p{i}.png")
        if captions:
            (d / f"p{i}.txt").write_text(f"zwx, cat, flower crown, windmill {i}")
    return str(d)


def _prep(folder, mp, copy=True):
    from fizgig.dataset.crop_captions import CropCaptions
    from fizgig.dataset.prep import Prep
    Prep(Q, captions=CropCaptions(copy=copy, keep=["zwx"], closeup=True, drop=[])).run(
        "Detail Crops (Style)", folder, mp)


def test_image_prep_makes_only_the_picked_size_and_copies_captions(tmp_path):
    f = _folder(tmp_path)
    _prep(f, 0.5)
    assert [os.path.basename(d) for _l, d in dc.sizes(f)] == ["0.5mp"]          # not every size up to it
    cap = open(os.path.join(dc.size_dir(f, 0.5), "p0.txt"), encoding="utf-8").read()
    assert cap.startswith("zwx, cat") and "close-up" in cap                     # Keep captions works with crops
    assert dc.copy_settings(f)["copy"] is True and dc.is_on(f)
    # Start at another size: the saved setting copies captions there too (no tagger at Start)
    dc.make(f, 0.25, log=Q)
    assert dc.copy_captions(f, 0.25, log=Q) == 2
    assert dc.to_caption(f, mps=[0.25]) == []


def test_no_copy_leaves_crops_for_the_captioner_at_each_size(tmp_path):
    f = _folder(tmp_path)
    _prep(f, 0.5, copy=False)
    assert dc.copy_settings(f) is None
    assert not [n for n in os.listdir(dc.size_dir(f, 0.5)) if n.endswith(".txt")]
    dc.make(f, 0.25, log=Q)
    with open(os.path.join(dc.size_dir(f, 0.5), "p0.txt"), "w") as fh:          # captioned at 0.5 MP only
        fh.write("the whole cat")
    todo = [os.path.relpath(p, os.path.join(f, dc.DIR)).replace(os.sep, "/") for p in dc.to_caption(f)]
    assert todo == ["0.5mp/p1.png", "0.25mp/p0.png", "0.25mp/p1.png"]           # never copied from another size
    assert [os.path.basename(p) for p in dc.to_caption(f, mps=[0.5])] == ["p1.png"]


def test_captions_tab_captions_every_size_of_a_crop(tmp_path):
    from fizgig.web.captions import _crop_images
    f = _folder(tmp_path)
    _prep(f, 0.5, copy=False)
    dc.make(f, 0.25, log=Q)
    big = os.path.join(dc.size_dir(f, 0.5), "p0.png")
    out, roots = _crop_images([big], False, ".txt")
    assert out == [big, os.path.join(dc.size_dir(f, 0.25), "p0.png")] and roots == []
    open(os.path.join(dc.size_dir(f, 0.25), "p0.txt"), "w").write("fur and grass")
    assert _crop_images([big], True, ".txt")[0] == [big]                        # a re-caption: that size only


def test_start_uses_the_captions_tab_model(tmp_path, monkeypatch):
    from fizgig.scripts import detail_crops as script
    tagger = {"backend": "tagger", "tagger": "jtp3", "model": "JTP-3 Hydra", "tags": {"min_score": 0.3}}
    assert script.captioner_config(tagger, "") == tagger
    assert script.captioner_config(None, "") is None
    enc = tmp_path / "qwen.safetensors"
    enc.write_bytes(b"x")
    assert script.captioner_config(None, str(enc)) == {"backend": "qwen", "text_encoder": str(enc)}
    f = _folder(tmp_path, captions=False)
    dc.switch(f, True)
    seen = {}
    fake = types.ModuleType("fizgig.scripts.batch_caption")
    fake._load_state_from_config = lambda cfg, dev: seen.setdefault("cfg", cfg) and types.SimpleNamespace(
        backend="florence", model=None)
    fake._caption_fn = lambda st, job, trig, n: (lambda p: f"tags for {os.path.basename(os.path.dirname(p))}")
    fake._write_caption = lambda p, cap, trig, ext: open(os.path.splitext(p)[0] + ext, "w").write(cap)
    monkeypatch.setitem(sys.modules, "fizgig.scripts.batch_caption", fake)
    import fizgig.scripts as pkg
    monkeypatch.setattr(pkg, "batch_caption", fake, raising=False)
    monkeypatch.setattr(sys, "argv", ["x", "--folder", f, "--mp", "0.5", "--mp", "0.25",
                                      "--recaption_with", json.dumps(tagger)])
    assert script.main() == 0
    assert seen["cfg"]["backend"] == "tagger"                                   # JTP-3, not Qwen3-VL
    assert open(os.path.join(dc.size_dir(f, 0.25), "p1.txt")).read() == "tags for 0.25mp"   # its own size
    assert open(os.path.join(dc.size_dir(f, 0.5), "p1.txt")).read() == "tags for 0.5mp"


def test_launch_passes_the_model_and_checks_captions_per_size(tmp_path):
    from fizgig.families import launch
    tagger = {"backend": "tagger", "tagger": "jtp3"}
    assert launch.detail_crops_captioner({"recaption_with": tagger}) is tagger
    assert launch.detail_crops_captioner({"recaption_with": None, "captioner": ""}) is None
    f = _folder(tmp_path)
    _prep(f, 0.5, copy=False)
    desc = types.SimpleNamespace(detail_crops=True)
    monkey = {"image_folder": f, "megapixels": "0.5", "caption_ext": ".txt"}
    for k in ("edit_on", "slider_on", "ft_on", "multi_person_on"):
        assert hasattr(launch, k)
    orig = {k: getattr(launch, k) for k in ("edit_on", "slider_on", "ft_on", "multi_person_on", "stage_mp")}
    try:
        for k in ("edit_on", "slider_on", "ft_on", "multi_person_on"):
            setattr(launch, k, lambda *a: False)
        launch.stage_mp = lambda *a: 0
        assert "Copy the picture's caption" in launch.detail_crops_problem(desc, monkey)
        assert launch.detail_crops_problem(desc, {**monkey, "recaption_with": tagger}) == ""
        dc.switch(f, True, captions={"copy": True, "closeup": True, "trigger": "", "drop": None})
        assert launch.detail_crops_problem(desc, monkey) == ""              # the pictures' captions are copied
    finally:
        for k, v in orig.items():
            setattr(launch, k, v)
