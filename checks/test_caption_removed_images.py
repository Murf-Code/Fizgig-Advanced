"""Caption Studio removed-image journal and safe recovery across application restarts."""
import json

from PIL import Image

from fizgig.web import removed_images


def test_removal_journal_survives_reload_and_reconciles(tmp_path):
    folder = tmp_path / "train"
    removed = folder / "removed"
    removed.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(removed / "example.png")
    (removed / "example.txt").write_text("caption", encoding="utf-8")
    move = [["example.png", "example.png"], ["example.txt", "example.txt"]]
    ident = removed_images.record(folder, move)
    assert ident
    items = removed_images.listing(folder)
    assert len(items) == 1
    assert items[0]["id"] == ident
    assert items[0]["moved"] == move
    assert removed_images.lookup(folder, ident)["name"] == "example.png"
    removed_images.reconcile(folder, move, ["example.png"])
    pending = removed_images.lookup(folder, ident)
    assert pending["moved"] == [["example.txt", "example.txt"]]
    removed_images.reconcile(folder, move, ["example.txt"])
    assert removed_images.listing(folder) == []


def test_colliding_removed_names_preserve_original_name(tmp_path):
    folder = tmp_path / "dataset"
    removed = folder / "removed"
    removed.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(removed / "portrait.png")
    Image.new("RGB", (16, 16)).save(removed / "portrait (2).png")
    a = removed_images.record(folder, [["portrait.png", "portrait.png"]])
    b = removed_images.record(folder, [["portrait.png", "portrait (2).png"]])
    by_id = {x["id"]: x for x in removed_images.listing(folder)}
    assert by_id[a]["moved"] == [["portrait.png", "portrait.png"]]
    assert by_id[b]["moved"] == [["portrait.png", "portrait (2).png"]]
    assert len(by_id) == 2


def test_older_removal_has_safe_same_name_fallback(tmp_path):
    root = tmp_path / "dataset"
    folder = root / "removed"
    folder.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(folder / "old.webp")
    (folder / "old.caption").write_text("prior caption", encoding="utf-8")
    entries = removed_images.listing(root, ".caption")
    assert entries == [{
        "id": "legacy:old.webp", "name": "old.webp",
        "moved": [["old.webp", "old.webp"], ["old.caption", "old.caption"]],
        "time": 0, "legacy": True,
    }]


def test_journal_rejects_path_traversal_inputs(tmp_path):
    folder = tmp_path / "dataset"
    moved = [["../outside.png", "inside.png"], ["valid.png", "../../outside.png"]]
    assert removed_images.record(folder, moved) == ""
    assert not (folder / "removed" / ".fizgig_caption_removals.json").exists()


def test_corrupt_journal_fails_to_legacy_listing_without_overwrite(tmp_path):
    folder = tmp_path / "dataset"
    rem = folder / "removed"
    rem.mkdir(parents=True)
    (rem / ".fizgig_caption_removals.json").write_text("{garbage", encoding="utf-8")
    Image.new("RGB", (16, 16)).save(rem / "legacy.png")
    items = removed_images.listing(folder)
    assert len(items) == 1 and items[0]["legacy"]
    assert (rem / ".fizgig_caption_removals.json").read_text() == "{garbage"
