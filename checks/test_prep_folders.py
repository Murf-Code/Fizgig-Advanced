"""Image Prep never changes the training folder (Chris's puss folder, 30 Oct 2026: resize + crops moved the
full-size pictures to originals/ without their captions). Every mode writes beside it instead: resized copies (with
captions and masks) in resized/, close-ups in crops/, pictures the cropping skipped copied to skipped/ with the
reason in skipped.txt. A re-run replaces those folders. Training uses resized/ in place of the full-size pictures."""
import hashlib
import os

import numpy as np
import pytest
from PIL import Image

from fizgig import face_utils as F
from fizgig.dataset import prep
from fizgig.dataset.image_dataset import glob_images
from fizgig.dataset.masks import MASK_SUFFIX
from fizgig.families import launch
from fizgig.families.registry import get as get_family

from test_crop_captions import _Det
from test_crop_other import FakeSeg


def _pic(path, w=400, h=320, seed=0):
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)).save(path)


def _folder(d):
    _pic(d / "a.jpg", seed=1)
    _pic(d / "b.png", seed=2)
    (d / "a.txt").write_text("ruaidri, paw, tail")
    (d / "b.txt").write_text("ruaidri, wolf")
    Image.new("L", (400, 320), 255).save(d / f"a{MASK_SUFFIX}.png")


def _top(d):
    return {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in os.listdir(d) if (d / f).is_file()}


FACE = [F._face_info(100, 80, 180, 160, "female", 0.9)]
RUNS = [
    ("Resize Only", None, None),
    ("Auto Prep (Face Crops)", "face", None),
    ("Face Crop Only", "face", None),
    ("Auto Prep (Face Crops)", "other", None),
    ("Face Crop Only", "other", None),
]


def _run(d, mode, kind, log=None, fit_all=False, hits=None):
    log = log if log is not None else []
    if kind == "other":
        finder = prep.ThingFinder(FakeSeg(hits or {"paw": [((100, 80, 180, 160), 0.9)]}),
                                  list((hits or {"paw": 0}).keys()), fit_all=fit_all)
        prep.Prep(log.append, None, finder).run(mode, str(d), 0.05, prep.OTHER_TARGET, 20.0, False)
    else:
        prep.Prep(log.append, _Det(FACE) if kind else None).run(mode, str(d), 0.05, "Largest Face", 20.0, False)
    return "".join(log)


@pytest.mark.parametrize("mode,kind,_x", RUNS)
def test_no_mode_changes_the_training_folder(tmp_path, mode, kind, _x):
    _folder(tmp_path)
    before = _top(tmp_path)
    text = _run(tmp_path, mode, kind)
    assert _top(tmp_path) == before and not (tmp_path / "originals").exists()
    assert "not changed" in text
    if mode != "Face Crop Only":
        r = tmp_path / "resized"
        assert sorted(os.listdir(r)) == ["a-masklabel.png", "a.png", "a.txt", "b.png", "b.txt"]
        assert (r / "a.txt").read_text() == "ruaidri, paw, tail"
        assert Image.open(r / "a.png").size == Image.open(r / f"a{MASK_SUFFIX}.png").size
    else:
        assert not (tmp_path / "resized").exists()
    if mode != "Resize Only":
        crops = sorted(os.listdir(tmp_path / "crops"))
        assert len([c for c in crops if c.endswith(".png") and MASK_SUFFIX not in c]) == 2
        assert any(c.endswith(MASK_SUFFIX + ".png") for c in crops)          # a's mask follows its crop


def test_crops_come_from_the_full_size_picture(tmp_path):
    _pic(tmp_path / "a.png", 2000, 1600, seed=5)
    _run(tmp_path, "Auto Prep (Face Crops)", "face")
    full = Image.open(tmp_path / "a.png").convert("RGB")
    want, _ = prep.resize_image(F.crop_to_face(full, FACE[0], 20.0), prep.target_area(0.05))
    got = Image.open(tmp_path / "crops" / "FaceCrop_001.png")
    assert np.array_equal(np.asarray(got), np.asarray(want))
    small = Image.open(tmp_path / "resized" / "a.png")
    assert small.size[0] < 2000                       # the resized copy is smaller: the crop wasn't cut from it


def test_skipped_pictures_are_copied_with_the_reason_and_rerun_replaces(tmp_path):
    _folder(tmp_path)
    text = _run(tmp_path, "Face Crop Only", "other", hits={"hat": []})
    s = tmp_path / "skipped"
    assert sorted(os.listdir(s)) == ["a-masklabel.png", "a.jpg", "a.txt", "b.png", "b.txt", "skipped.txt"]
    assert (s / "a.jpg").read_bytes() == (tmp_path / "a.jpg").read_bytes()     # the true file, not re-encoded
    assert "a.jpg: nothing found to crop" in (s / "skipped.txt").read_text()
    assert "Nothing found in a.jpg" in text
    _run(tmp_path, "Face Crop Only", "other")                 # found this time: skipped/ starts afresh
    assert not (tmp_path / "skipped").exists() and len(os.listdir(tmp_path / "crops")) >= 4


def test_an_already_right_png_is_copied_as_is(tmp_path):
    """Why 12f67f8c... was 'skipped' on Chris's run: already a PNG at the bucket size, so nothing to change."""
    _pic(tmp_path / "x.png", 64, 64)
    text = _run(tmp_path, "Resize Only", None)
    assert (tmp_path / "resized" / "x.png").read_bytes() == (tmp_path / "x.png").read_bytes()
    assert "Copied as is" in text


def test_training_uses_the_resized_copy_and_crops_without_duplicates(tmp_path):
    _folder(tmp_path)
    _run(tmp_path, "Auto Prep (Face Crops)", "face")
    assert glob_images(str(tmp_path)) == []                   # both have a copy in resized/
    st = {"image_folder": str(tmp_path), "LORA_OUTPUT_DIR": str(tmp_path / "out"), "megapixels": 0.25,
          "cache_root": str(tmp_path.parent / "cache"), "subfolders": False}
    assert [r for r, _ in launch.run_subfolders(get_family("krea2"), st)] == ["resized", "crops"]
    assert launch.is_own_subfolder("skipped") and launch._media_count(str(tmp_path)) == 0
    toml = launch.dataset_toml(get_family("krea2"), st)
    assert 'item_prefix = "resized/"' in toml and 'item_prefix = "crops/"' in toml and "skipped" not in toml


def test_old_originals_folders_still_left_out(tmp_path):
    (tmp_path / "originals").mkdir()
    _pic(tmp_path / "originals" / "o.jpg")
    _pic(tmp_path / "p.png")
    assert launch.subfolders(str(tmp_path)) == [] and launch._media_count(str(tmp_path)) == 1


# ---- Fit all words in one crop ----
def test_fit_all_crops_around_every_word(tmp_path):
    _pic(tmp_path / "a.png", 1000, 800, seed=3)
    hits = {"pussy": [((300, 300, 360, 360), 0.9)], "anus": [((380, 380, 420, 430), 0.9)]}
    _run(tmp_path, "Face Crop Only", "other", fit_all=True, hits=hits)
    pngs = [c for c in os.listdir(tmp_path / "crops") if c.endswith(".png")]
    assert pngs == ["Crop_pussy_anus_001.png"]
    full = Image.open(tmp_path / "a.png").convert("RGB")
    want, _ = prep.resize_image(F.crop_to_face(full, F._face_info(300, 300, 420, 430), 20.0), prep.target_area(0.05))
    assert np.array_equal(np.asarray(Image.open(tmp_path / "crops" / pngs[0])), np.asarray(want))


def test_fit_all_replaces_per_word_crops_in_auto_prep_and_off_keeps_them(tmp_path):
    hits = {"pussy": [((300, 300, 360, 360), 0.9)], "anus": [((380, 380, 420, 430), 0.9)]}
    for sub, on in (("on", True), ("off", False)):
        d = tmp_path / sub
        d.mkdir()
        _pic(d / "a.png", 1000, 800, seed=3)
        _run(d, "Auto Prep (Face Crops)", "other", fit_all=on, hits=hits)
        n = len([c for c in os.listdir(d / "crops") if c.endswith(".png")])
        assert n == (1 if on else 2)


def test_fit_all_too_big_is_skipped_and_one_word_crops_alone(tmp_path):
    _pic(tmp_path / "a.png", 400, 320, seed=4)
    hits = {"pussy": [((5, 5, 60, 60), 0.9)], "anus": [((330, 260, 395, 315), 0.9)]}
    text = _run(tmp_path, "Face Crop Only", "other", fit_all=True, hits=hits)
    assert "nearly the whole picture" in (tmp_path / "skipped" / "skipped.txt").read_text()
    assert "Not cropped: a.png" in text
    hits = {"pussy": [((100, 100, 160, 160), 0.9)], "anus": []}
    text = _run(tmp_path, "Face Crop Only", "other", fit_all=True, hits=hits)
    assert "Only pussy found (not anus)" in text
    assert [c for c in os.listdir(tmp_path / "crops") if c.endswith(".png")] == ["Crop_pussy_001.png"]


def test_fit_all_is_on_unless_turned_off_in_the_page():
    js = open(os.path.join(os.path.dirname(prep.__file__), "..", "web", "static", "app.js"), encoding="utf-8").read()
    assert '$("prepFitAll").checked = saved.fitAll !== false' in js


# ---- a too-big box is ignored; YOLO beats SAM for the same word (Chris's 'pussy, anus, ass' run, 30 Oct 2026:
# SAM's near-whole-picture 'ass' boxes made the fit-all crop too big, so 43 of 58 pictures were skipped) ----
class _Fixed(prep.ThingFinder):
    def __init__(self, hits, words, fit_all=True):
        super().__init__(None, words, fit_all=fit_all)
        self.hits = hits

    def find(self, img):
        return list(self.hits)


def _hit(box, word, source):
    return F._face_info(*box, gender=word, score=0.9, source=source)


def _prep_fixed(d, hits, words, mode="Face Crop Only", fit_all=True):
    log = []
    prep.Prep(log.append, None, _Fixed(hits, words, fit_all)).run(mode, str(d), 0.05, prep.OTHER_TARGET, 20.0, False)
    return "".join(log)


def test_sam_giant_box_ignored_crop_fits_yolo_boxes(tmp_path):
    _pic(tmp_path / "a.png", 1000, 800, seed=6)
    hits = [_hit((300, 300, 360, 360), "pussy", "yolo"), _hit((380, 380, 420, 430), "anus", "yolo"),
            _hit((0, 0, 1000, 790), "ass", "sam3")]
    text = _prep_fixed(tmp_path, hits, ["pussy", "anus", "ass"])
    assert "ass: SAM 3.1's box is too big (99%), ignored" in text
    assert [c for c in os.listdir(tmp_path / "crops") if c.endswith(".png")] == ["Crop_pussy_anus_001.png"]
    full = Image.open(tmp_path / "a.png").convert("RGB")
    want, _ = prep.resize_image(F.crop_to_face(full, F._face_info(300, 300, 420, 430), 20.0), prep.target_area(0.05))
    assert np.array_equal(np.asarray(Image.open(tmp_path / "crops" / "Crop_pussy_anus_001.png")), np.asarray(want))


def test_yolo_preferred_over_sam_for_the_same_word(tmp_path):
    _pic(tmp_path / "a.png", 1000, 800, seed=7)
    hits = [_hit((100, 100, 500, 500), "pussy", "sam3"), _hit((300, 300, 360, 360), "pussy", "yolo")]
    _prep_fixed(tmp_path, hits, ["pussy"], fit_all=False)
    full = Image.open(tmp_path / "a.png").convert("RGB")
    want, _ = prep.resize_image(F.crop_to_face(full, F._face_info(300, 300, 360, 360), 20.0), prep.target_area(0.05))
    assert np.array_equal(np.asarray(Image.open(tmp_path / "crops" / "Crop_pussy_001.png")), np.asarray(want))


def test_only_a_giant_box_is_skipped_with_the_reason(tmp_path):
    _pic(tmp_path / "a.png", 1000, 800, seed=8)
    _prep_fixed(tmp_path, [_hit((0, 0, 1000, 800), "ass", "sam3")], ["pussy", "anus", "ass"])
    assert "only boxes covering most of the picture" in (tmp_path / "skipped" / "skipped.txt").read_text()
    assert not (tmp_path / "crops").exists() or not os.listdir(tmp_path / "crops")


def test_chris_run_replayed_every_picture_cropped(tmp_path):
    """58 pictures; YOLO finds pussy and/or anus in each, SAM an 'ass' box over nearly all of it: all get a crop."""
    words = ["pussy", "anus", "ass"]

    class PerPic(prep.ThingFinder):
        def __init__(self):
            super().__init__(None, words, fit_all=True)
            self.n = 0

        def find(self, img):
            self.n += 1
            out = [_hit((5, 5, 395, 315), "ass", "sam3")]
            if self.n % 3:
                out.append(_hit((150, 150, 200, 200), "pussy", "yolo"))
            if self.n % 2:
                out.append(_hit((190, 200, 230, 240), "anus", "yolo"))
            if not self.n % 6:                           # neither: SAM's own small find is used
                out.append(_hit((100, 100, 160, 160), "ass", "sam3"))
            return out
    for i in range(58):
        _pic(tmp_path / f"p{i:02d}.png", seed=i)
    log = []
    prep.Prep(log.append, None, PerPic()).run("Auto Prep (Face Crops)", str(tmp_path), 0.05, prep.OTHER_TARGET,
                                             20.0, False)
    crops = [c for c in os.listdir(tmp_path / "crops") if c.endswith(".png")]
    assert len(crops) == 58 and not (tmp_path / "skipped").exists()
