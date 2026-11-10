"""The picture editor's Crop and Remove (web/image_edit.py): a crop crops the picture's mask the same way, on the
upright picture, so they stay lined up; Remove fills a painted thing from what's around it and leaves the rest of
the picture alone; each change undoes and redoes to the exact file bytes; files keep their format and see-through
parts."""
import io
import os

import numpy as np
import pytest
from PIL import Image

from fizgig.web import image_edit

_real_lama_fill = image_edit.lama_fill


def _mean_fill(rgb, hole, gpu=True):
    """Stands in for LaMa (a 206 MB download) in these checks: the hole takes the mean colour of the rest."""
    out = rgb.copy()
    out[hole] = np.round(rgb[~hole].reshape(-1, 3).mean(0)).astype(np.uint8)
    return out


@pytest.fixture(autouse=True)
def _no_lama(monkeypatch):
    monkeypatch.setattr(image_edit, "lama_fill", _mean_fill)


def _png(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_hole_fills_a_closed_loop_and_grows():
    stroke = np.zeros((40, 40), bool)
    stroke[10, 10:30] = stroke[29, 10:30] = stroke[10:30, 10] = stroke[10:30, 29] = True   # a ring
    hole = image_edit.prepare_hole(stroke)
    assert hole[20, 20]                                 # inside the ring goes too
    assert hole[8, 20] and not hole[5, 20]              # grown about 2 px, no more
    assert not hole[0, 0]


def test_remove_fills_from_around_and_leaves_the_rest():
    a = np.full((64, 96, 3), 120, np.uint8)
    a[20:30, 40:50] = (250, 0, 0)                       # the thing to take out
    stroke = np.zeros((64, 96), bool)
    stroke[19:31, 39:51] = True
    out = image_edit.inpaint_rgb(a, stroke)
    assert np.abs(out[20:30, 40:50].astype(int) - 120).max() < 12
    far = np.ones((64, 96), bool)
    far[8:42, 28:62] = False
    assert np.array_equal(out[far], a[far])             # only round the stroke is touched


def test_crop_keeps_the_mask_lined_up_on_an_upright_picture(tmp_path):
    pic = np.zeros((40, 60, 3), np.uint8)               # stored 60x40, rotation flag 6: seen 40x60
    img = Image.fromarray(pic)
    exif = img.getexif()
    exif[0x0112] = 6
    img.save(tmp_path / "phone.jpg", exif=exif, quality=95)
    seen = image_edit.open_upright((tmp_path / "phone.jpg").read_bytes())
    assert seen.size == (40, 60)
    m = np.zeros((30, 20), np.uint8)                    # a half-size mask, stretched as training reads it
    m[5:10, 5:10] = 255
    Image.fromarray(m, "L").save(tmp_path / "phone-masklabel.png")
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    r = image_edit.crop(str(tmp_path), "phone.jpg", (10, 10, 30, 40))
    assert r["size"] == [20, 30]
    with Image.open(tmp_path / "phone.jpg") as im:
        assert im.size == (20, 30) and im.format == "JPEG"
        assert im.getexif().get(0x0112, 1) == 1           # upright pixels, so the flag is reset
    with Image.open(tmp_path / "phone-masklabel.png") as mk:
        mk = np.asarray(mk.convert("L"))
    assert mk.shape == (30, 20)
    assert mk[5, 5] > 200 and mk[14, 5] < 50 and mk[5, 14] < 50   # the white square moved with the crop (10..20 -> 0..10)
    image_edit.step(r["step"], "undo")
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == before
    image_edit.step(r["step"], "redo")
    with Image.open(tmp_path / "phone.jpg") as im:
        assert im.size == (20, 30)


def test_remove_keeps_format_alpha_and_undoes(tmp_path):
    a = np.full((32, 32, 4), (90, 90, 90, 255), np.uint8)
    a[:, :4, 3] = 0                                     # a see-through strip
    a[12:18, 12:18, :3] = 255
    Image.fromarray(a, "RGBA").save(tmp_path / "x.png")
    Image.new("RGB", (32, 32), (90, 90, 90)).save(tmp_path / "y.webp", lossless=True)
    orig = (tmp_path / "x.png").read_bytes()
    s = np.zeros((32, 32), np.uint8)
    s[11:19, 11:19] = 255
    r = image_edit.remove(str(tmp_path), "x.png", _png(Image.fromarray(s, "L")))
    with Image.open(tmp_path / "x.png") as im:
        out = np.asarray(im)
        assert im.mode == "RGBA"
    assert out[:, :4, 3].max() == 0 and out[15, 15, 0] < 140
    image_edit.step(r["step"], "undo")
    assert (tmp_path / "x.png").read_bytes() == orig
    r = image_edit.remove(str(tmp_path), "y.webp", _png(Image.fromarray(s, "L")))
    assert image_edit._webp_lossless((tmp_path / "y.webp").read_bytes())


def test_crop_box_is_clamped():
    assert image_edit.clamp_box((-5, -5, 500, 20), (100, 50)) == (0, 0, 100, 20)
    assert image_edit.clamp_box((30, 10, 30, 10), (100, 50)) == (30, 10, 31, 11)


def test_remove_changes_only_the_masked_pixels():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)   # noise: any stray change shows
    stroke = np.zeros((300, 400), bool)
    stroke[100:120, 150:260] = True                    # a text-like bar
    hole = image_edit.prepare_hole(stroke)
    out = image_edit.inpaint_rgb(a, stroke)
    # the fill reaches FEATHER px past the hole, for the fade; everything further out is bit-identical
    import cv2
    k = 2 * image_edit.FEATHER + 1
    reach = cv2.dilate(hole.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    assert np.array_equal(out[~reach], a[~reach])
    assert not np.array_equal(out[hole], a[hole])


def test_remove_without_lama_says_so_and_leaves_the_picture(tmp_path, monkeypatch):
    from fizgig.dataset import lama
    from fizgig.utils import hf_public
    monkeypatch.setattr(image_edit, "lama_fill", _real_lama_fill)
    monkeypatch.setattr(lama, "_model", None)
    monkeypatch.setattr(lama, "_folder", lambda: str(tmp_path / "models" / "lama"))
    monkeypatch.delenv("FIZGIG_LAMA_PATH", raising=False)

    def offline(**kw):
        raise OSError("no internet")
    monkeypatch.setattr(hf_public, "download", offline)
    pics = tmp_path / "pics"
    pics.mkdir()
    Image.new("RGB", (40, 40), (90, 90, 90)).save(pics / "x.png")
    orig = (pics / "x.png").read_bytes()
    s = np.zeros((40, 40), np.uint8)
    s[10:20, 10:20] = 255
    with pytest.raises(image_edit.LamaUnavailable) as e:
        image_edit.remove(str(pics), "x.png", _png(Image.fromarray(s, "L")), backup=True)
    assert "LaMa model" in str(e.value) and "wasn't changed" in str(e.value)
    assert (pics / "x.png").read_bytes() == orig
    assert sorted(p.name for p in pics.iterdir()) == ["x.png"]          # no backup, no mask, nothing else

    # a broken file: loading fails the same clear way
    bad = tmp_path / "models" / "lama" / lama.FILENAME
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not a model")
    monkeypatch.setenv("FIZGIG_LAMA_PATH", str(bad))
    with pytest.raises(image_edit.LamaUnavailable) as e:
        image_edit.remove(str(pics), "x.png", _png(Image.fromarray(s, "L")))
    assert "Couldn't load the LaMa model" in str(e.value)
    assert (pics / "x.png").read_bytes() == orig


@pytest.mark.skipif(not os.environ.get("FIZGIG_LAMA_PATH"), reason="set FIZGIG_LAMA_PATH to the LaMa file")
def test_real_lama_removes_a_bar(monkeypatch):
    monkeypatch.setattr(image_edit, "lama_fill", _real_lama_fill)
    a = np.full((128, 128, 3), 120, np.uint8)
    a[60:68, 20:108] = (250, 250, 250)                 # white text-like bar on grey
    stroke = np.zeros((128, 128), bool)
    stroke[60:68, 20:108] = True
    out = image_edit.inpaint_rgb(a, stroke, gpu=False)
    assert np.abs(out[60:68, 20:108].astype(int) - 120).max() < 20
