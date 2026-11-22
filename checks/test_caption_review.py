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
