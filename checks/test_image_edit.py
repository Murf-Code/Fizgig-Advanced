"""The picture editor's Crop and Remove (web/image_edit.py): a crop crops the picture's mask the same way, on the
upright picture, so they stay lined up; Remove fills a painted thing from what's around it and leaves the rest of
the picture alone; each change undoes and redoes to the exact file bytes; files keep their format and see-through
parts."""
import io

import numpy as np
from PIL import Image

from fizgig.web import image_edit


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
