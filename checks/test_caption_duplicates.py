"""CPU-only Caption Studio similarity groups and collision-safe tag merges."""
import os
import time

import numpy as np
import pytest
from PIL import Image

from fizgig.web import caption_duplicates, caption_review


def picture(root, name, size=(72, 64), kind="pattern"):
    root.mkdir(parents=True, exist_ok=True)
    if kind == "pattern":
        y, x = np.indices((size[1], size[0]))
        rgb = np.stack(((x * 7 + y * 2) % 256, (x * 3 + y * 13) % 256,
                        (x * 11 + y * 3) % 256), axis=2).astype("uint8")
        rgb[7:24, 12:45] = (10, 250, 15)
        img = Image.fromarray(rgb, "RGB")
    else:
        img = Image.new("RGB", size, kind)
    target = root / name
    img.save(target)
    return str(target)


def await_done(root):
    for i in range(250):
        value = caption_duplicates.status(str(root), 4)
        if value["stage"] in ("done", "error"):
            return value
        time.sleep(0.02)
    raise AssertionError("duplicate scan did not finish")


def test_hash_detects_resized_copy_and_discards_different_solid_colors(tmp_path):
    root = tmp_path / "photos"
    src = picture(root, "a.png")
    with Image.open(src) as image:
        image.resize((144, 128), Image.Resampling.LANCZOS).save(root / "b.png")
    red = picture(root, "red.png", kind="red")
    blue = picture(root, "blue.png", kind="blue")
    sig_a = caption_duplicates.fingerprint(src)
    sig_b = caption_duplicates.fingerprint(str(root / "b.png"))
    assert caption_duplicates.apart(sig_a, sig_b) <= 16
    assert caption_duplicates.apart(
        caption_duplicates.fingerprint(red), caption_duplicates.fingerprint(blue)) > 4
    start = caption_duplicates.start(str(root), [src, str(root / "b.png"), red, blue])
    assert start["stage"] in ("scanning", "matching", "done")
    r = await_done(root)
    assert r["stage"] == "done", r.get("error")
    assert r["scanned"] == 4
    groups = caption_duplicates.status(str(root), 16)
    assert any({os.path.basename(i["path"]) for i in g["items"]} == {"a.png", "b.png"}
               for g in groups["groups"])
    assert all("blue.png" not in {i["name"] for i in g["items"]}
               for g in caption_duplicates.status(str(root), 4)["groups"])


def test_scan_threshold_changes_without_redecoding_images(tmp_path, monkeypatch):
    root = tmp_path / "data"
    a = picture(root, "a.png")
    b = picture(root, "b.png")
    names = [a, b]
    caption_duplicates.start(str(root), names)
    assert await_done(root)["stage"] == "done"
    monkeypatch.setattr(caption_duplicates, "fingerprint", lambda _: (
        (_ for _ in ()).throw(AssertionError("threshold adjustment redecoded picture"))))
    high = caption_duplicates.status(str(root), 0)
    low = caption_duplicates.status(str(root), 16)
    assert high["stage"] == low["stage"] == "done"
    assert high["pair_count"] <= low["pair_count"]


def test_scan_excludes_masks_backups_and_paths_outside_folder(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    pic = picture(root, "subject.png")
    mask = picture(root, "subject-masklabel.png")
    extra = tmp_path / "unrelated"
    outsider = picture(extra, "outside.png")
    with pytest.raises(ValueError, match="outside"):
        caption_duplicates.checked_path(str(root), outsider)
    with pytest.raises(ValueError, match="source photos"):
        caption_duplicates.checked_path(str(root), mask)
    with pytest.raises(ValueError, match="Choose"):
        caption_duplicates.start("", [pic])
    scan_root, paths, _ = caption_duplicates.paths_for_scan(
        str(root), [pic, pic, mask, outsider])
    assert scan_root == str(root)
    assert paths == [pic]


def test_scan_separates_dataset_subfolders_and_never_touches_removed(tmp_path):
    root = tmp_path / "dataset"
    pic = picture(root / "subject_a", "portrait.png")
    pic2 = picture(root / "subject_b", "portrait.png")
    pic3 = picture(root / "removed", "portrait.png")
    assert len(caption_duplicates.paths_for_scan(str(root), [pic, pic2, pic3])[1]) == 2
    assert not any("removed" in p for p in
                   caption_duplicates.paths_for_scan(str(root), [pic, pic2, pic3])[1])


def test_merge_adds_unique_casefolded_tags_and_preserves_multiline_tail(tmp_path):
    root = tmp_path / "data"
    a = picture(root, "keep.png")
    b = picture(root, "donor.png")
    c = picture(root, "third.png")
    (root / "keep.txt").write_text("lion, cosmic mane\nMetadata: keep this line exactly\n", encoding="utf-8")
    (root / "donor.txt").write_text("Lion, blue eyes, smiling", encoding="utf-8")
    (root / "third.txt").write_text("blue eyes, fluffy tail", encoding="utf-8")
    expected = {
        a: "lion, cosmic mane\nMetadata: keep this line exactly\n",
        b: "Lion, blue eyes, smiling",
        c: "blue eyes, fluffy tail",
    }
    preview = caption_duplicates.merge_tags(str(root), a, [b, c], expected=expected)
    assert not preview["applied"]
    assert preview["added"] == ["blue eyes", "smiling", "fluffy tail"]
    assert preview["caption"] == ("lion, cosmic mane, blue eyes, smiling, fluffy tail\n"
                                  "Metadata: keep this line exactly\n")
    assert (root / "keep.txt").read_text(encoding="utf-8") == expected[a]
    applied = caption_duplicates.merge_tags(str(root), a, [b, c], expected=expected,
                                            apply=True)
    assert applied["applied"] and applied["added_count"] == 3
    assert (root / "keep.txt").read_text(encoding="utf-8") == preview["caption"]
    assert (root / "donor.txt").read_text(encoding="utf-8") == expected[b]
    assert os.path.isfile(b) and os.path.isfile(c)


def test_merge_rejects_stale_snapshot_without_modifying_keeper(tmp_path):
    root = tmp_path / "data"
    a, b = picture(root, "keep.png"), picture(root, "donor.png")
    (root / "keep.txt").write_text("cat", encoding="utf-8")
    (root / "donor.txt").write_text("long tail", encoding="utf-8")
    expected = {a: "cat", b: "long tail"}
    (root / "donor.txt").write_text("a newer edited caption", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        caption_duplicates.merge_tags(root, a, [b], expected=expected, apply=True)
    assert (root / "keep.txt").read_text(encoding="utf-8") == "cat"


def test_merge_refuses_prose_or_reviewed_keeper(tmp_path):
    root = tmp_path / "data"
    a, b = picture(root, "keep.png"), picture(root, "donor.png")
    (root / "keep.txt").write_text("a cat with a blue coat.", encoding="utf-8")
    (root / "donor.txt").write_text("eyes, blue fur", encoding="utf-8")
    expected = {a: "a cat with a blue coat.", b: "eyes, blue fur"}
    with pytest.raises(ValueError, match="prose"):
        caption_duplicates.merge_tags(root, a, [b], expected=expected, apply=True)
    (root / "keep.txt").write_text("cat, blue coat", encoding="utf-8")
    expected[a] = "cat, blue coat"
    caption_review.mark(root, "keep.png", True)
    with pytest.raises(ValueError, match="reviewed"):
        caption_duplicates.merge_tags(root, a, [b], expected=expected, apply=True)
    assert (root / "keep.txt").read_text(encoding="utf-8") == expected[a]


def test_merge_respects_custom_separator_and_missing_source_caption(tmp_path):
    root = tmp_path / "data"
    a, b, c = (picture(root, n) for n in ("a.png", "b.png", "c.png"))
    (root / "a.caption").write_text("cat; blue fur", encoding="utf-8")
    (root / "b.caption").write_text("blue fur; sharp teeth", encoding="utf-8")
    expected = {a: "cat; blue fur", b: "blue fur; sharp teeth", c: None}
    result = caption_duplicates.merge_tags(root, a, [b, c], ext=".caption",
                                           delimiter=";", expected=expected, apply=True)
    assert result["added"] == ["sharp teeth"]
    assert (root / "a.caption").read_text(encoding="utf-8") == "cat; blue fur; sharp teeth"
    assert not (root / "c.caption").exists()
