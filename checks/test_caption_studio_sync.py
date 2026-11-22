"""Regression tests for Caption Studio's live caption queue (no GPU needed)."""
import json
from fizgig.web import image_loss


def _run(tmp_path):
    root = tmp_path / "pictures"
    root.mkdir()
    out = tmp_path / "output"
    out.mkdir()
    return root, {"image_folder": str(root), "output_dir": str(out), "result": "running"}


def test_studio_queues_updates_from_main_and_subfolder(tmp_path):
    root, run = _run(tmp_path)
    sub = root / "outfits"
    sub.mkdir()
    assert image_loss.queue_for_folder(str(root), "a.png", "first", ".txt", run)
    assert image_loss.queue_for_folder(str(sub), "b.webp", "second", ".txt", run)
    p = tmp_path / "output" / "loss_log" / "caption_updates.json"
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": "first", "outfits/b": "second"}
    assert image_loss.queue_for_folder(str(root), "a.png", "new", ".txt", run)
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": "new", "outfits/b": "second"}


def test_studio_does_not_enqueue_for_unrelated_folder_or_finished_run(tmp_path):
    root, run = _run(tmp_path)
    other = tmp_path / "unrelated"
    other.mkdir()
    assert not image_loss.queue_for_folder(str(other), "a.png", "x", ".txt", run)
    run["result"] = "finished"
    assert not image_loss.queue_for_folder(str(root), "a.png", "x", ".txt", run)
    assert not (tmp_path / "output" / "loss_log" / "caption_updates.json").exists()


def test_studio_keeps_empty_caption_in_queue(tmp_path):
    root, run = _run(tmp_path)
    assert image_loss.queue_for_folder(str(root), "a.png", "", ".txt", run)
    p = tmp_path / "output" / "loss_log" / "caption_updates.json"
    assert json.loads(p.read_text(encoding="utf-8"))["a"] == ""
