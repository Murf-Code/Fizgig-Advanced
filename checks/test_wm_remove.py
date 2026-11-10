"""Linking Make masks' watermark sidecars to the picture editor's Remove: a sidecar is written when
watermarks are found, Remove can fill that area, the training mask is restored only there (other
edits stay), originals are copied to removed_watermarks/, and a finished sidecar is not applied again."""
import io

import numpy as np
from PIL import Image

from fizgig.dataset import detectors as D
from fizgig.dataset import wm_regions
from fizgig.dataset.masks import MASK_SUFFIX, mask_for
from fizgig.web import image_edit, mask_edit


class Stub:
    def __init__(self, boxes):
        self.boxes = boxes

    def all_classes(self):
        return {0: "watermark"}

    def find_classes(self, arr, ids):
        return self.boxes


def _png(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _pic(path, size=(80, 60), colour=(40, 80, 120)):
    Image.new("RGB", size, colour).save(path)


def test_clamp_pad_and_encode_round_trip():
    assert wm_regions.clamp_pad(3) == 3 and wm_regions.clamp_pad(1) == 2 and wm_regions.clamp_pad(9) == 5
    assert wm_regions.clamp_pad("x") == 3
    cover = np.zeros((10, 12), bool); cover[2:5, 3:8] = True
    prior = np.full((10, 12), 200, np.uint8); prior[:, :4] = 0
    c2, p2 = wm_regions.decode(wm_regions.encode(cover, prior))
    assert np.array_equal(c2, cover) and np.array_equal(p2, prior)


def test_pad_cover_grows_about_three_px():
    cover = np.zeros((40, 40), bool); cover[15:20, 15:20] = True
    padded = wm_regions.pad_cover(cover, 3)
    assert padded[17, 17] and padded[17, 12] and not padded[17, 8]
    assert padded.sum() > cover.sum()


def test_just_hide_writes_a_sidecar_and_remove_restores_then_drops_the_mask(tmp_path):
    _pic(tmp_path / "a.png", (100, 80), (90, 90, 90))
    # a dark rectangle in the corner stands in for a watermark, so the fill has something to cover
    im = np.array(Image.open(tmp_path / "a.png"))
    im[10:25, 60:90] = (20, 20, 20)
    Image.fromarray(im).save(tmp_path / "a.png")
    orig = (tmp_path / "a.png").read_bytes()
    wm = [Stub([(60, 10, 90, 25, 0.9, 0)])]
    c = D.watermarks_only_folder(str(tmp_path), wm, log=lambda m: None)
    assert c["made"] == 1 and wm_regions.has_active(str(tmp_path / "a.png"))
    m = np.asarray(Image.open(mask_for(str(tmp_path / "a.png"))))
    assert m[15, 70] == 0 and m[40, 40] == 255
    listed = mask_edit.listing(str(tmp_path))
    assert listed == [{"name": "a.png", "mask": True, "watermark": True}]

    stroke = image_edit.watermark_stroke_png(str(tmp_path), "a.png", pad=3)
    assert stroke is not None
    with Image.open(io.BytesIO(stroke)) as s:
        s = np.asarray(s.convert("L"))
    assert s[15, 70] > 127 and s[15, 57] > 127          # padded ~3 px past x=60
    assert s[40, 40] < 127

    r = image_edit.remove_watermark(str(tmp_path), "a.png", pad=3)
    assert r["ok"] and r["watermark"]["finished"] and r["watermark"]["mask_removed"]
    assert not wm_regions.has_active(str(tmp_path / "a.png"))
    assert (tmp_path / "watermarks" / "done" / "a.png").is_file()
    assert not mask_for(str(tmp_path / "a.png"))         # just-hide mask was only the watermark: deleted
    backup = tmp_path / "removed_watermarks" / "a.png"
    assert backup.is_file() and backup.read_bytes() == orig
    with Image.open(tmp_path / "a.png") as out:
        arr = np.asarray(out.convert("RGB"))
    # the dark patch is gone (filled from the grey around it); far pixels stay
    assert arr[15, 70].astype(int).max() > 50
    assert np.array_equal(arr[50, 10], (90, 90, 90))

    # a second call has nothing to do; the first backup is kept
    try:
        image_edit.remove_watermark(str(tmp_path), "a.png", pad=3)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "no watermark mask" in str(e)
    assert backup.read_bytes() == orig

    image_edit.step(r["step"], "undo")
    assert (tmp_path / "a.png").read_bytes() == orig
    assert mask_for(str(tmp_path / "a.png")) and wm_regions.has_active(str(tmp_path / "a.png"))
    image_edit.step(r["step"], "redo")
    assert not mask_for(str(tmp_path / "a.png")) and not wm_regions.has_active(str(tmp_path / "a.png"))


def test_character_mask_is_restored_only_in_the_watermark(tmp_path):
    # subject on the left (white), background on the right (black); watermark sits on both
    _pic(tmp_path / "a.png", (200, 100), (100, 100, 100))
    base = np.zeros((100, 200), np.uint8); base[:, :120] = 255
    Image.fromarray(base, "L").save(tmp_path / f"a{MASK_SUFFIX}.png")
    wm = [Stub([(100, 30, 150, 50, 0.9, 0)])]           # overlaps subject (x<120) and background (x>=120)
    D.watermarks_only_folder(str(tmp_path), wm, log=lambda m: None)
    m = np.asarray(Image.open(mask_for(str(tmp_path / "a.png"))))
    assert m[40, 110] == 0 and m[40, 140] == 0 and m[40, 20] == 255
    cover, prior = wm_regions.read(str(tmp_path / "a.png"))
    assert prior[40, 110] == 255 and prior[40, 140] == 0  # remembered: subject vs background

    r = image_edit.remove_watermark(str(tmp_path), "a.png", pad=2)
    assert r["watermark"]["restored"] and not r["watermark"]["mask_removed"]
    m = np.asarray(Image.open(mask_for(str(tmp_path / "a.png"))))
    assert m[40, 110] == 255                             # subject, watermark gone: train there again
    assert m[40, 140] == 0                               # background stays black
    assert m[40, 20] == 255                              # the rest of the subject untouched
    assert m[80, 160] == 0


def test_hand_painted_remove_only_clears_the_watermark_pixels_it_covers(tmp_path):
    _pic(tmp_path / "a.png", (200, 120), (80, 80, 80))
    im = np.array(Image.open(tmp_path / "a.png")); im[20:50, 40:160] = 10; Image.fromarray(im).save(tmp_path / "a.png")
    D.watermarks_only_folder(str(tmp_path), [Stub([(50, 25, 150, 45, 0.9, 0)])], log=lambda m: None)
    # paint over only the left half of the watermark
    stroke = np.zeros((120, 200), np.uint8); stroke[15:55, 30:100] = 255
    r = image_edit.remove(str(tmp_path), "a.png", _png(Image.fromarray(stroke, "L")))
    assert r["watermark"]["restored"] and not r["watermark"]["finished"]
    assert wm_regions.has_active(str(tmp_path / "a.png"))
    left, _ = wm_regions.read(str(tmp_path / "a.png"))
    assert not left[30, 70] and left[30, 130]            # right half still waiting
    m = np.asarray(Image.open(mask_for(str(tmp_path / "a.png"))))
    assert m[30, 70] == 255 and m[30, 130] == 0


def test_crop_crops_the_sidecar_too(tmp_path):
    _pic(tmp_path / "a.png", (200, 150), (70, 70, 70))
    D.watermarks_only_folder(str(tmp_path), [Stub([(80, 40, 140, 80, 0.9, 0)])], log=lambda m: None)
    r = image_edit.crop(str(tmp_path), "a.png", (40, 20, 200, 150))
    assert r["size"] == [160, 130]
    cover, _ = wm_regions.read(str(tmp_path / "a.png"))
    assert cover.shape == (130, 160)
    assert cover[30, 50] and not cover[5, 5]             # box moved by (-40, -20)


def test_overwrite_with_no_watermark_drops_the_sidecar(tmp_path):
    _pic(tmp_path / "a.png", (80, 80))
    D.watermarks_only_folder(str(tmp_path), [Stub([(10, 10, 30, 30, 0.9, 0)])], log=lambda m: None)
    assert wm_regions.has_active(str(tmp_path / "a.png"))
    D.watermarks_only_folder(str(tmp_path), [Stub([])], overwrite=True, log=lambda m: None)
    assert not wm_regions.has_active(str(tmp_path / "a.png"))
    assert not (tmp_path / "watermarks").exists()
    assert not mask_for(str(tmp_path / "a.png"))


def test_restore_does_not_touch_user_edits_outside_the_watermark(tmp_path):
    # a just-hide mask, then the user blacks out something else; Remove must leave that edit
    _pic(tmp_path / "a.png", (200, 150), (90, 90, 90))
    D.watermarks_only_folder(str(tmp_path), [Stub([(20, 20, 50, 50, 0.9, 0)])], log=lambda m: None)
    path = mask_for(str(tmp_path / "a.png"))
    m = np.array(Image.open(path))
    m[100:130, 140:180] = 0                              # a later hand edit, not a watermark
    Image.fromarray(m, "L").save(path)
    image_edit.remove_watermark(str(tmp_path), "a.png", pad=2)
    out = np.asarray(Image.open(mask_for(str(tmp_path / "a.png"))))
    assert out[30, 30] == 255                            # watermark area restored
    assert (out[100:130, 140:180] == 0).all()            # the hand edit stays


def test_red_edits_are_kept_without_the_pad_growing_and_erasing_all_drops_it(tmp_path):
    p = str(tmp_path / "a.png")
    _pic(p, (80, 60))
    cover = np.zeros((60, 80), bool); cover[10:20, 10:30] = True
    wm_regions.write_sidecar(p, cover, np.full((60, 80), 255, np.uint8))
    as_loaded = lambda: np.asarray(Image.open(io.BytesIO(image_edit.watermark_stroke_png(str(tmp_path), "a.png", 3)))) > 127
    first = as_loaded()
    add = np.zeros((60, 80), np.uint8); add[40:50, 50:70] = 255          # a missed signature painted on
    erase = np.zeros((60, 80), np.uint8); erase[5:25, 5:18] = 255        # the left end rubbed out
    r = image_edit.save_watermark_edit(str(tmp_path), "a.png", _png(Image.fromarray(add)), _png(Image.fromarray(erase)), 3)
    assert r["watermark"]
    again = as_loaded()                                   # going back to the picture shows the edits
    assert again[45, 60] and not again[15, 15] and again[15, 25] and not again[erase > 0].any()
    # saving what was loaded, unchanged, changes nothing (the pad isn't baked in)
    same = wm_regions.read(p)[0].copy()
    image_edit.save_watermark_edit(str(tmp_path), "a.png", "", "", 3)
    assert (wm_regions.read(p)[0] == same).all() and (as_loaded() == again).all()
    full = _png(Image.fromarray(np.full((60, 80), 255, np.uint8)))
    r = image_edit.save_watermark_edit(str(tmp_path), "a.png", "", full, 3)
    assert not r["watermark"] and not wm_regions.has_active(p)
    assert not mask_edit.listing(str(tmp_path))[0]["watermark"]
    assert first.any()
