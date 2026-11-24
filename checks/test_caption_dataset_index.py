"""Large dataset metadata cache: correctness is more important than stale speed."""
from pathlib import Path
import os

from fizgig.web.caption_index import FolderIndex


def test_persistent_index_reuses_unchanged_and_not_changed_files(tmp_path):
    folder = tmp_path / "dataset"
    folder.mkdir()
    db = tmp_path / "index.sqlite3"
    for i in range(1000):
        (folder / f"{i}.txt").write_text(f"caption {i}", encoding="utf-8")
    reads = []
    def reader(path):
        reads.append(path)
        return Path(path).read_text(encoding="utf-8")
    def scan():
        with FolderIndex(db, folder) as index:
            return {n: index.read(n, "caption", reader) for n in index.names}
    first = scan()
    assert len(reads) == 1000
    assert scan() == first
    assert len(reads) == 1000
    (folder / "0.txt").write_text("changed", encoding="utf-8")
    (folder / "1.txt").unlink()
    (folder / "new.txt").write_text("added", encoding="utf-8")
    result = scan()
    assert result["0.txt"] == "changed" and result["new.txt"] == "added"
    assert "1.txt" not in result
    assert len(reads) == 1002


def test_caption_change_does_not_invalidate_dimensions(tmp_path):
    folder = tmp_path / "dataset"
    folder.mkdir()
    db = tmp_path / "index.sqlite3"
    (folder / "a.png").write_bytes(b"fake image")
    (folder / "a.txt").write_text("original", encoding="utf-8")
    with FolderIndex(db, folder) as index:
        assert index.read("a.png", "size", lambda _: [500, 600]) == [500, 600]
        assert index.read("a.txt", "caption", lambda _: "original") == "original"
    (folder / "a.txt").write_text("new longer caption", encoding="utf-8")
    def forbidden(_):
        raise AssertionError("unchanged image was reopened")
    with FolderIndex(db, folder) as index:
        assert index.read("a.png", "size", forbidden) == [500, 600]
        assert index.read("a.txt", "caption", lambda p: Path(p).read_text()) == "new longer caption"
        assert index.read("missing.txt", "caption", forbidden) is None


def test_unavailable_cache_falls_back_to_files(tmp_path):
    (tmp_path / "a.txt").write_text("source", encoding="utf-8")
    with FolderIndex(tmp_path / "missing" / "index.sqlite3", tmp_path) as index:
        assert index.read("a.txt", "caption", lambda p: Path(p).read_text()) == "source"
