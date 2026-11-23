"""Regression checks for CPU-saving Caption Workbench thumb/preview caching."""
import io
import os

from PIL import Image

from fizgig.web import server


def _size(encoded):
    with Image.open(io.BytesIO(encoded)) as image:
        return image.size


def test_thumbnails_are_size_specific_and_persist_across_memory_eviction(tmp_path, monkeypatch):
    image_path = tmp_path / "source.jpg"
    Image.new("RGB", (900, 700), color=(30, 60, 120)).save(image_path, "JPEG")
    monkeypatch.setattr(server, "user_file", lambda name: str(tmp_path / name))
    server._thumbs.clear()

    tiny = server._thumb(str(image_path), 160)
    large = server._thumb(str(image_path), 384)
    assert tiny and large and tiny != large
    assert max(_size(tiny)) <= 160
    assert max(_size(large)) <= 384
    disk = tmp_path / "thumb_cache"
    assert len(list(disk.glob("*.bin"))) == 2

    # Both sizes can be served from disk without reopening the original photo.
    server._thumbs.clear()
    original_open = Image.open

    def forbid_source(path, *args, **kwargs):
        if str(path) == str(image_path):
            raise AssertionError("source image was decoded again")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(Image, "open", forbid_source)
        assert server._thumb(str(image_path), 160) == tiny
        assert server._thumb(str(image_path), 384) == large


def test_edit_invalidates_cached_previews(tmp_path, monkeypatch):
    image_path = tmp_path / "edited.jpg"
    Image.new("RGB", (640, 640), color="red").save(image_path, "JPEG")
    monkeypatch.setattr(server, "user_file", lambda name: str(tmp_path / name))
    server._thumbs.clear()
    first = server._thumb(str(image_path), 160)
    assert first
    old = image_path.stat()
    Image.new("RGB", (640, 640), color="blue").save(image_path, "JPEG")
    os.utime(image_path, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000_000))
    second = server._thumb(str(image_path), 160)
    assert second and second != first
    assert len(list((tmp_path / "thumb_cache").glob("*.bin"))) == 2


def test_thumbnail_ram_cache_evicts_old_previews(tmp_path, monkeypatch):
    image_path = tmp_path / "large.png"
    Image.new("RGB", (800, 800), color="green").save(image_path, "PNG")
    monkeypatch.setattr(server, "user_file", lambda name: str(tmp_path / name))
    server._thumbs.clear()
    monkeypatch.setattr(server, "_THUMB_RAM_LIMIT", 1000)
    assert server._thumb(str(image_path), 320)
    assert not server._thumbs or sum(map(len, server._thumbs.values())) <= 1000
