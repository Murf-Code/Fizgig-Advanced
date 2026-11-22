"""Optimistic multi-caption edit and grouped undo/redo regression tests."""
import pytest

from fizgig.web import caption_batch_edit, caption_review


def _dataset(tmp_path):
    folder = tmp_path / "captions"
    folder.mkdir()
    for name, caption in (("lion", "lion, blue"), ("cat", "cat, grey")):
        (folder / (name + ".png")).write_bytes(b"image")
        (folder / (name + ".txt")).write_text(caption, encoding="utf-8")
    return folder


def _row(folder, name, expected, text):
    return {"folder": str(folder), "name": name + ".png", "expected": expected, "text": text}


def test_grouped_caption_edit_and_reverse(tmp_path):
    folder = _dataset(tmp_path)
    forward = [_row(folder, "lion", "lion, blue", "lion, red"),
               _row(folder, "cat", "cat, grey", "cat, orange")]
    assert len(caption_batch_edit.apply(forward)) == 2
    assert (folder / "lion.txt").read_text() == "lion, red"
    assert (folder / "cat.txt").read_text() == "cat, orange"
    backward = [_row(folder, "lion", "lion, red", "lion, blue"),
                _row(folder, "cat", "cat, orange", "cat, grey")]
    assert len(caption_batch_edit.apply(backward)) == 2
    assert (folder / "lion.txt").read_text() == "lion, blue"
    assert (folder / "cat.txt").read_text() == "cat, grey"


def test_conflict_prevents_entire_batch(tmp_path):
    folder = _dataset(tmp_path)
    entries = [_row(folder, "lion", "lion, blue", "lion, red"),
               _row(folder, "cat", "stale caption", "cat, orange")]
    with pytest.raises(ValueError, match="changed outside Studio"):
        caption_batch_edit.apply(entries)
    assert (folder / "lion.txt").read_text() == "lion, blue"


def test_review_lock_prevents_entire_batch(tmp_path):
    folder = _dataset(tmp_path)
    caption_review.mark(folder, "cat.png", True)
    with pytest.raises(ValueError, match="locked"):
        caption_batch_edit.apply([_row(folder, "lion", "lion, blue", "lion, red"),
                                  _row(folder, "cat", "cat, grey", "cat, orange")])
    assert (folder / "lion.txt").read_text() == "lion, blue"
    assert (folder / "cat.txt").read_text() == "cat, grey"


def test_duplicate_stems_and_bad_paths_rejected(tmp_path):
    folder = _dataset(tmp_path)
    entries = [_row(folder, "lion", "lion, blue", "lion, red"),
               _row(folder, "lion", "lion, blue", "lion, orange")]
    with pytest.raises(ValueError, match="same caption"):
        caption_batch_edit.apply(entries)
    with pytest.raises(ValueError, match="filename"):
        caption_batch_edit.apply([{"folder": str(folder), "name": "../escape.png",
                                   "expected": None, "text": "bad"}])


def test_rollback_if_second_atomic_write_fails(tmp_path, monkeypatch):
    folder = _dataset(tmp_path)
    real = caption_batch_edit.write_atomic
    attempts = []

    def fail_second(path, value):
        attempts.append(path)
        if len(attempts) == 2:
            raise OSError("simulated write error")
        return real(path, value)

    monkeypatch.setattr(caption_batch_edit, "write_atomic", fail_second)
    with pytest.raises(OSError, match="simulated"):
        caption_batch_edit.apply([_row(folder, "lion", "lion, blue", "lion, red"),
                                  _row(folder, "cat", "cat, grey", "cat, orange")])
    assert (folder / "lion.txt").read_text() == "lion, blue"
    assert (folder / "cat.txt").read_text() == "cat, grey"


def test_missing_caption_created_and_undo_deletes(tmp_path):
    folder = _dataset(tmp_path)
    (folder / "lion.txt").unlink()
    caption_batch_edit.apply([_row(folder, "lion", None, "new caption")])
    assert (folder / "lion.txt").read_text() == "new caption"
    caption_batch_edit.apply([_row(folder, "lion", "new caption", None)])
    assert not (folder / "lion.txt").exists()
