"""Review locks persist across requests and do not erase unreadable files."""
import json
import pytest

from fizgig.web import caption_review


def test_review_lock_round_trip(tmp_path):
    folder = tmp_path / "dataset"
    folder.mkdir()
    assert caption_review.reviewed(folder) == []
    assert not caption_review.is_locked(folder, "portrait.png")
    assert caption_review.mark(folder, "portrait.png", True) == ["portrait.png"]
    assert caption_review.is_locked(folder, "portrait.png")
    assert json.loads((folder / ".fizgig_caption_reviews.json").read_text()) == ["portrait.png"]
    assert caption_review.mark(folder, "portrait.png", False) == []
    assert not caption_review.is_locked(folder, "portrait.png")


def test_review_status_fails_closed_when_corrupted(tmp_path):
    folder = tmp_path / "dataset"
    folder.mkdir()
    path = folder / ".fizgig_caption_reviews.json"
    path.write_text("not valid json")
    with pytest.raises(ValueError, match="Could not read"):
        caption_review.mark(folder, "portrait.png", True)
    assert path.read_text() == "not valid json"


def test_review_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        caption_review.mark(tmp_path, "../outside.png", True)


def test_caption_review_blocks_batch_caption_writers(tmp_path):
    from fizgig.web import captions
    from fizgig.scripts.batch_caption import _write_caption

    folder = tmp_path / "dataset"
    folder.mkdir()
    (folder / "lion.png").write_bytes(b"placeholder")
    (folder / "lion.txt").write_text("original", encoding="utf-8")
    (folder / "cat.png").write_bytes(b"placeholder")
    (folder / "cat.txt").write_text("cat", encoding="utf-8")
    caption_review.mark(folder, "lion.png", True)
    assert caption_review.is_caption_locked(folder / "lion.txt")
    assert not caption_review.is_caption_locked(folder / "cat.txt")
    result = captions.find_replace(str(folder), "original", "changed", True)
    assert result["changes"] == []
    assert (folder / "lion.txt").read_text() == "original"
    result = captions.add_text(str(folder), "prefix", "", "", True)
    assert (folder / "lion.txt").read_text() == "original"
    assert (folder / "cat.txt").read_text().startswith("prefix")
    with pytest.raises(ValueError, match="review lock"):
        _write_caption(str(folder / "lion.png"), "overwrite")
    assert (folder / "lion.txt").read_text() == "original"


def test_unreadable_review_ledger_blocks_legacy_batch(tmp_path):
    from fizgig.web import captions
    (tmp_path / "lion.txt").write_text("before", encoding="utf-8")
    (tmp_path / ".fizgig_caption_reviews.json").write_text("corrupt")
    with pytest.raises(ValueError):
        captions.find_replace(str(tmp_path), "before", "after", True)
    assert (tmp_path / "lion.txt").read_text() == "before"


def test_ai_append_still_respects_review_locks(tmp_path):
    from fizgig.scripts.batch_caption import _write_caption
    folder = tmp_path / "dataset"
    folder.mkdir()
    image = folder / "lion.png"
    image.write_bytes(b"placeholder")
    _write_caption(str(image), "first")
    _write_caption(str(image), "second", append=True)
    assert (folder / "lion.txt").read_text() == "first\nsecond"
    _write_caption(str(image), "second", append=True)
    assert (folder / "lion.txt").read_text().count("second") == 1
    caption_review.mark(folder, image.name, True)
    with pytest.raises(ValueError, match="review lock"):
        _write_caption(str(image), "third", append=True)
    assert "third" not in (folder / "lion.txt").read_text()


def test_reviewed_detail_crop_is_not_overwritten_by_caption_mirror(tmp_path):
    from fizgig.dataset import detail_crops

    for label in ("0.25mp", "0.5mp"):
        d = tmp_path / detail_crops.DIR / label
        d.mkdir(parents=True)
        (d / "lion.png").write_bytes(b"picture")
        (d / "cat.png").write_bytes(b"picture")
    small = tmp_path / detail_crops.DIR / "0.25mp"
    large = tmp_path / detail_crops.DIR / "0.5mp"
    (small / "lion.txt").write_text("reviewed text", encoding="utf-8")
    (large / "lion.txt").write_text("new shared text", encoding="utf-8")
    (small / "cat.txt").write_text("older", encoding="utf-8")
    (large / "cat.txt").write_text("updated", encoding="utf-8")
    caption_review.mark(small, "lion.png", True)
    assert detail_crops.share_captions(str(tmp_path)) == 1
    assert (small / "lion.txt").read_text() == "reviewed text"
    assert (small / "cat.txt").read_text() == "updated"
