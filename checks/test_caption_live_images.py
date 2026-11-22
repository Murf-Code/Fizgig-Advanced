"""CPU-only queue checks for live image edits at epoch boundaries."""
import json
from types import SimpleNamespace

from fizgig.web import image_loss
from fizgig.families import loss_watch


def test_image_queue_merges_and_preserves_subfolder_keys(tmp_path):
    folder = tmp_path / "dataset"
    folder.mkdir()
    sub = folder / "faces"
    sub.mkdir()
    output = tmp_path / "run"
    run = {"result": "running", "output_dir": str(output), "image_folder": str(folder)}
    assert image_loss.queue_image_for_folder(str(folder), "a.png", "updated", run)
    assert image_loss.queue_image_for_folder(str(sub), "b.jpg", "removed", run)
    assert image_loss.queue_image_for_folder(str(folder), "a.png", "removed", run)
    p = output / "loss_log" / "image_updates.json"
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": "removed", "faces/b": "removed"}


def test_removed_image_excluded_at_epoch_without_vae(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    out = tmp_path / "run"
    q = out / "loss_log" / "image_updates.json"
    q.parent.mkdir(parents=True)
    q.write_text('{"photo": "removed"}', encoding="utf-8")
    monkeypatch.setattr(loss_watch, "_items", lambda group: {"photo": SimpleNamespace()})
    watch = loss_watch.Watch.__new__(loss_watch.Watch)
    watch.output_dir = str(out)
    watch.group = object()
    watch.folders = {"": str(root)}
    watch.image_dir = str(root)
    watch.removed_keys = set()
    watch.vae_path = None
    watch.watch = None
    watch.reg = set()
    watch._images(1, None, None, False)
    assert watch.removed_keys == {"photo"}
    assert watch.excluded({"item_keys": ["photo"]})
    assert not q.exists()


def test_failed_image_refresh_is_queued_and_not_trained(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "photo.png").write_bytes(b"dummy")
    out = tmp_path / "run"
    q = out / "loss_log" / "image_updates.json"
    q.parent.mkdir(parents=True)
    q.write_text('{"photo": "updated"}', encoding="utf-8")
    item = SimpleNamespace(bucket_size=(256, 256))
    monkeypatch.setattr(loss_watch, "_items", lambda group: {"photo": item})
    watch = loss_watch.Watch.__new__(loss_watch.Watch)
    watch.output_dir = str(out)
    watch.group = object()
    watch.folders = {"": str(root)}
    watch.image_dir = str(root)
    watch.removed_keys = set()
    watch.vae_path = None
    watch.watch = None
    watch.reg = set()
    watch._images(1, None, None, False)
    assert watch.excluded({"item_keys": ["photo"]})
    assert json.loads(q.read_text(encoding="utf-8")) == {"photo": "updated"}
