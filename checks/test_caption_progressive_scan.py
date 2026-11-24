"""The progressive listing must be exactly equivalent to the complete API."""
from pathlib import Path
from threading import Event

import pytest


def test_progressive_scan_matches_complete_listing(tmp_path, monkeypatch):
    from fizgig.web import server
    from fizgig.dataset import detail_crops
    from fizgig.dataset import image_dataset

    root = tmp_path / "dataset"
    root.mkdir()
    resized = root / "resized"
    resized.mkdir()
    crops = tmp_path / "detail"
    crops.mkdir()
    mask_dir = root / "masks"
    mask_dir.mkdir()
    for i in range(51):
        (root / f"image{i:03}.png").write_bytes(b"image")
        (root / f"image{i:03}.txt").write_text(f"tag {i}", encoding="utf-8")
    (root / "image000-masklabel.png").write_bytes(b"mask")
    (mask_dir / "image001-masklabel.png").write_bytes(b"mask")
    (root / "old.png").write_bytes(b"original")
    (resized / "old.png").write_bytes(b"resized")
    (resized / "old.txt").write_text("resized", encoding="utf-8")
    (crops / "crop.png").write_bytes(b"crop")
    (crops / "crop.txt").write_text("crop", encoding="utf-8")
    for name in ("clip.mp4", "clip.png", "clip.wav", "song.wav", "song.png"):
        (root / name).write_bytes(b"media")
    (root / "clip.txt").write_text("clip", encoding="utf-8")
    (root / "song.txt").write_text("song", encoding="utf-8")

    monkeypatch.setattr(server, "user_file", lambda name: str(tmp_path / name))
    monkeypatch.setattr(server.launch, "prep_subfolders", lambda _: [("resized", str(resized))])
    monkeypatch.setattr(detail_crops, "largest", lambda _: str(crops))
    monkeypatch.setattr(image_dataset, "superseded_stems", lambda p: {"old"} if Path(p) == root else set())
    reads = []
    def size(path):
        reads.append(path)
        return [512, 768]
    monkeypatch.setattr(server, "_media_size", size)
    monkeypatch.setattr(server, "_audio_seconds", lambda _: 1.5)

    handler = object.__new__(server.Handler)
    handler.json = lambda value, *args: value
    cancel = Event()
    stream = handler._caption_scan_events(str(root), ".txt", "", cancel)
    assert next(stream)["type"] == "discovering"
    manifest = next(stream)
    assert manifest["type"] == "manifest"
    assert reads == [], "directory enumeration must not open image headers"
    first = next(stream)
    assert len(first["items"]) == 24
    assert len(reads) <= 24, "only the first batch may open headers before preview"
    cancel.set()
    with pytest.raises(InterruptedError):
        next(stream)
    before = len(reads)
    assert before <= 24

    events = list(handler._caption_scan_events(str(root), ".txt", "", Event()))
    items = [item for event in events if event["type"] == "batch" for item in event["items"]]
    assert events[-1] == {"type": "done", "total": len(items)}
    assert events[1]["total"] == len(items)
    assert items == handler._captions(str(root), ".txt", "")["items"]
    assert len(items) == 55
    by_name = {it["name"]: it for it in items}
    assert by_name["clip.mp4"]["image"].endswith("clip.png")
    assert by_name["song.wav"]["duration"] == 1.5
    assert by_name["image000.png"]["mask"] == "image000-masklabel.png"
    assert by_name["image001.png"]["mask"] == "image001-masklabel.png"
    assert by_name["resized/old.png"]["sub"] == "resized"
    assert by_name["crop.png"]["crop"] == "detail"
    assert "old.png" not in by_name and "clip.wav" not in by_name
