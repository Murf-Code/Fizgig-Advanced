"""Image Prep's mask editor (web/mask_edit.py): pictures and masks go out upright and come back upright, a save
replaces the picture's mask (whatever its format), an all-black one removes it, and only the folder's own pictures
are touched."""
import base64
import io
import os

import numpy as np
import pytest
from PIL import Image

from fizgig.web import mask_edit


def _png_url(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _rotated(folder):
    pic = Image.new("RGB", (60, 40), "gray")                 # stored 60x40, seen 40x60 (flag 6: a quarter turn)
    exif = pic.getexif()
    exif[0x0112] = 6
    pic.save(folder / "phone.jpg", exif=exif)


def test_listing_and_upright_picture(tmp_path):
    _rotated(tmp_path)
    Image.new("RGB", (32, 32)).save(tmp_path / "b.png")
    Image.new("L", (32, 32), 255).save(tmp_path / "b-masklabel.png")
    assert mask_edit.listing(str(tmp_path)) == [{"name": "b.png", "mask": True, "watermark": False, "backup": False}, {"name": "phone.jpg", "mask": False, "watermark": False, "backup": False}]
    with Image.open(io.BytesIO(mask_edit.picture_jpeg(str(tmp_path), "phone.jpg"))) as im:
        assert im.size == (40, 60)
    assert mask_edit.mask_png(str(tmp_path), "phone.jpg") is None


def test_save_writes_upright_and_replaces_another_format(tmp_path):
    _rotated(tmp_path)
    Image.new("L", (40, 60), 255).save(tmp_path / "phone-masklabel.jpg")
    painted = Image.new("RGBA", (20, 30), (0, 0, 0, 0))     # white on transparent, smaller than the picture
    painted.paste((255, 255, 255, 255), (0, 0, 20, 15))
    assert mask_edit.save(str(tmp_path), "phone.jpg", _png_url(painted)) == {"saved": True}
    assert not (tmp_path / "phone-masklabel.jpg").exists()
    with Image.open(tmp_path / "phone-masklabel.png") as m:
        assert m.size == (40, 60) and m.mode == "L"
        a = np.asarray(m)
    assert a[:25].mean() > 250 and a[35:].mean() < 5          # the top half white, the bottom black
    with Image.open(io.BytesIO(mask_edit.mask_png(str(tmp_path), "phone.jpg"))) as m:
        assert m.size == (40, 60)


def test_all_black_removes_the_mask(tmp_path):
    Image.new("RGB", (32, 32)).save(tmp_path / "a.png")
    Image.new("L", (32, 32), 255).save(tmp_path / "a-masklabel.png")
    assert mask_edit.save(str(tmp_path), "a.png", _png_url(Image.new("RGBA", (32, 32)))) == {"removed": True}
    assert not (tmp_path / "a-masklabel.png").exists()


def test_only_the_folders_pictures(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    Image.new("L", (8, 8), 255).save(tmp_path / "a-masklabel.png")
    for name in ("../a.png", "a-masklabel.png", "missing.png", os.path.join(str(tmp_path), "a.png")):
        with pytest.raises(ValueError):
            mask_edit.picture_jpeg(str(tmp_path), name)
