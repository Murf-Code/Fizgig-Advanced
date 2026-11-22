"""PNG metadata performance: fast dimensions without losing EXIF orientation.

Adapted from upstream shootthesound/Fizgig PR #399. The simple PNG path
must never decode pixels just to determine whether dimensions are rotated.
"""
import struct
import zlib

import pytest
from PIL import Image

from fizgig.utils import upright
from fizgig.utils.upright import png_carries_orientation, upright_size

W, H = 4, 8
ORIENTATION_6 = 6
TIFF_ORIENTATION_6 = (
    b"II\x2a\x00\x08\x00\x00\x00"
    b"\x01\x00"
    b"\x12\x01\x03\x00\x01\x00\x00\x00\x06\x00\x00\x00"
    b"\x00\x00\x00\x00"
)


def _chunk(kind, data):
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def _save(path, orientation=None):
    exif = Image.Exif()
    if orientation is not None:
        exif[0x0112] = orientation
    Image.new("RGB", (W, H)).save(path, "PNG", exif=exif if orientation else None)


def _late_exif(path):
    raw = path.read_bytes()
    path.write_bytes(raw[:-12] + _chunk(b"eXIf", TIFF_ORIENTATION_6) + raw[-12:])


def test_plain_png_never_decodes_pixels_for_dimensions(tmp_path):
    path = tmp_path / "plain.png"
    _save(path)
    with Image.open(path) as im:
        im.load = lambda *a, **k: (_ for _ in ()).throw(AssertionError("decoded pixels"))
        assert upright_size(im, str(path)) == (W, H)
        assert png_carries_orientation(str(path)) is False


def test_png_exif_orientation_kept(tmp_path):
    path = tmp_path / "rotated.png"
    _save(path, ORIENTATION_6)
    with Image.open(path) as im:
        assert upright_size(im, str(path)) == (H, W)


def test_late_exif_kept_even_after_idat(tmp_path):
    path = tmp_path / "late.png"
    _save(path)
    _late_exif(path)
    assert png_carries_orientation(str(path)) is True
    with Image.open(path) as im:
        assert "exif" not in im.info
        assert upright_size(im, str(path)) == (H, W)


def test_without_path_uses_original_logic(tmp_path):
    path = tmp_path / "fallback.png"
    _save(path)
    _late_exif(path)
    with Image.open(path) as im:
        assert upright_size(im) == (H, W)


def test_jpeg_orientation_unchanged(tmp_path):
    path = tmp_path / "photo.jpg"
    exif = Image.Exif()
    exif[0x0112] = ORIENTATION_6
    Image.new("RGB", (W, H)).save(path, "JPEG", exif=exif)
    with Image.open(path) as im:
        assert upright_size(im, str(path)) == (H, W)


@pytest.mark.parametrize("raw", [
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\x01",
    b"not a png",
])
def test_invalid_or_uncertain_inputs_use_fallback(tmp_path, raw):
    path = tmp_path / "bad.png"
    path.write_bytes(raw)
    assert png_carries_orientation(str(path)) is True


def test_text_chunk_triggers_safe_exif_fallback(tmp_path):
    path = tmp_path / "text.png"
    _save(path)
    raw = path.read_bytes()
    path.write_bytes(raw[:-12] + _chunk(b"tEXt", b"Software\x00foo") + raw[-12:])
    assert png_carries_orientation(str(path)) is True


def test_chunk_limit_triggers_safe_fallback(tmp_path, monkeypatch):
    path = tmp_path / "plain.png"
    _save(path)
    monkeypatch.setattr(upright, "_MAX_CHUNKS", 1)
    assert png_carries_orientation(str(path)) is True
