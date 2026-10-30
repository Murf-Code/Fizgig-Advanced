"""Where Image Prep puts things, pinned: every mode that makes crops names them the same way - Crop_<word>_NNN for
what you type, FaceCrop_NNN for faces, numbered on after any file of that name already there (Chris: "no matter
what crop setting the file name should remain consistent"). Resize + close-up crops saves them beside the resized
pictures; Close-up crops only (Face Crop Only) adds the crop beside its picture, which stays in the dataset with its caption
untouched (Chris: "when it crops an image, I still want that original image to remain in the dataset") - no originals
folder for crops. Resized originals go into ONE originals folder - originals, or originals_2 when an earlier run's
originals are there.
Chris's Zaush run (Face Crop Only, Other… sheath, captions on) put each original in its own folder, originals_2 to
originals_46: the caption step forgot the run's originals folder after every picture, so the next one picked a new one."""
import os

import numpy as np
import pytest
from PIL import Image

from fizgig import face_utils as F
from fizgig.dataset import crop_captions as C
from fizgig.dataset import prep

from test_crop_captions import FakeTagger as _FakeTagger, _Det


class FakeTagger(_FakeTagger):
    """test_crop_captions' stand-in, which also knows the targets (a tag it doesn't know always stays, so a target
    would never be trimmed and the checks below would prove nothing)."""

    def __init__(self, scores):
        super().__init__(scores)
        self.categories.update({"sheath": 0, "face": 0, "paw pads": 0, "pawpads": 0})

CAP = "ruaidri, sheath, wolf, full body"
SEEN = {"sheath": 0.9, "wolf": 0.8, "full body": 0.02}
LOW = {"sheath": 0.0, "wolf": 0.8, "full body": 0.02}


class _Seg:
    def masks(self, image, prompts, threshold=0.5, refine=2, max_each=None):
        return [{"prompt": p, "score": 0.9, "box": [60.0, 60.0, 140.0, 140.0], "mask": None} for p in prompts]


def _folder(tmp_path, n=4, earlier=True):
    if earlier:                                        # an earlier run's originals already there (Chris's folder)
        os.makedirs(tmp_path / "originals")
        Image.new("RGB", (8, 8)).save(tmp_path / "originals" / "old.jpg")
    for i in range(n):
        Image.fromarray(np.random.default_rng(i).integers(0, 255, (300, 300, 3), dtype=np.uint8)).save(
            tmp_path / f"p{i}.jpg")
        (tmp_path / f"p{i}.txt").write_text(CAP)


def _tree(d):
    return sorted(os.path.relpath(os.path.join(r, f), d).replace(os.sep, "/") for r, _, fs in os.walk(d) for f in fs)


def _caps(trim):
    return C.CropCaptions(keep=["ruaidri"], tagger=FakeTagger(SEEN) if trim else None)


def _things(tmp_path, mode, caps):
    prep.Prep(lambda _t: None, None, prep.ThingFinder(_Seg(), ["sheath"]), captions=caps).run(
        mode, str(tmp_path), 0.05, prep.OTHER_TARGET, 20.0, False)







def test_crop_only_without_caption_settings_still_keeps_the_caption(tmp_path):
    """captions=None (the desktop's Image Prep): the crop gets its picture's caption as written, as it did when the
    crop took the picture's own name."""
    _folder(tmp_path, n=1, earlier=False)
    prep.Prep(lambda _t: None, None, prep.ThingFinder(_Seg(), ["sheath"])).run(
        "Face Crop Only", str(tmp_path), 0.05, prep.OTHER_TARGET, 20.0, False)
    assert (tmp_path / "crops" / "Crop_sheath_001.txt").read_text() == CAP
    assert (tmp_path / "p0.txt").read_text() == CAP and not (tmp_path / "originals").exists()







@pytest.mark.parametrize("cap", ["ruaidri, Sheath, wolf, full body", "ruaidri, wolf, sheath, full body"])
def test_trim_keeps_the_find_this_word(tmp_path, cap):
    _folder(tmp_path, n=1, earlier=False)
    (tmp_path / "p0.txt").write_text(cap)
    caps = C.CropCaptions(keep=["ruaidri"], tagger=FakeTagger(LOW), closeup=False)
    _things(tmp_path, "Face Crop Only", caps)
    tags = [t.strip() for t in (tmp_path / "crops" / "Crop_sheath_001.txt").read_text().split(",")]
    assert "full body" not in tags and "wolf" in tags
    assert any(t.lower() == "sheath" for t in tags), tags


def test_target_match_ignores_case_spaces_underscores():
    caps = C.CropCaptions(tagger=FakeTagger({"wolf": 0.8}), closeup=False)
    caps.targets = ["paw pads"]
    for cap, kept in (("x, Paw_Pads, wolf, tail", "Paw_Pads"), ("x, pawpads, wolf", "pawpads"),
                      ("x, paw-pads, wolf", "paw-pads")):
        text, dropped, added = caps.caption_for(cap, Image.new("RGB", (64, 64)))
        assert kept in text.split(", ") and kept not in dropped and not added, (cap, text)


def test_face_is_the_target_of_face_crops(tmp_path):
    _folder(tmp_path, n=1, earlier=False)
    (tmp_path / "p0.txt").write_text("ruaidri, face, wolf, full body")
    caps = C.CropCaptions(keep=["ruaidri"], tagger=FakeTagger({"wolf": 0.8, "face": 0.0}), closeup=False)
    prep.Prep(lambda _t: None, _Det([F._face_info(60, 60, 140, 140)]), captions=caps).run(
        "Face Crop Only", str(tmp_path), 0.05, "Largest Face", 20.0, False)
    assert (tmp_path / "crops" / "FaceCrop_001.txt").read_text() == "ruaidri, face, wolf"


@pytest.mark.parametrize("add", [False, True])
def test_missing_target_added_only_with_add_missing_tags(tmp_path, add):
    _folder(tmp_path, n=1, earlier=False)
    (tmp_path / "p0.txt").write_text("ruaidri, wolf, full body")
    caps = C.CropCaptions(keep=["ruaidri"], tagger=FakeTagger(LOW), closeup=True, add_score=0.9 if add else None)
    _things(tmp_path, "Auto Prep (Face Crops)", caps)
    crop = (tmp_path / "crops" / "Crop_sheath_001.txt").read_text()
    assert crop == ("ruaidri, wolf, sheath, close-up" if add else "ruaidri, wolf, close-up")
    assert (tmp_path / "p0.txt").read_text() == "ruaidri, wolf, full body"     # the resized picture: as written


def test_added_target_follows_underscore_style():
    caps = C.CropCaptions(tagger=FakeTagger({"wolf": 0.8}), closeup=False, add_score=0.9)
    caps.targets = ["paw pads"]
    text, _d, added = caps.caption_for("x, long_hair, wolf", Image.new("RGB", (64, 64)))
    assert added == ["paw_pads"] and text.endswith(", paw_pads")
    caps.add_score = None
    assert caps.caption_for("x, long_hair, wolf", Image.new("RGB", (64, 64)))[2] == []
