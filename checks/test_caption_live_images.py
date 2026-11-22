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


def test_source_scan_detects_caption_and_image_edits(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    picture = root / "photo.png"
    picture.write_bytes(b"old-pixels")
    caption = root / "photo.txt"
    caption.write_text("original", encoding="utf-8")
    monkeypatch.setattr(loss_watch, "_items", lambda group: {"photo": SimpleNamespace()})
    watch = loss_watch.Watch.__new__(loss_watch.Watch)
    watch.group = object()
    watch.image_dir = str(root)
    watch.caption_ext = ".txt"
    watch.folders = {"": str(root)}
    watch._source_snapshot = watch._source_state()
    assert watch._scan_sources() == ({}, {})
    caption.write_text("changed", encoding="utf-8")
    picture.write_bytes(b"new-pixels-and-contents")
    img, txt = watch._scan_sources()
    assert img == {"photo": "updated"}
    assert txt == {"photo": "changed"}
    picture.unlink()
    img, txt = watch._scan_sources()
    assert img == {"photo": "removed"}
    assert txt == {}


def test_claimed_queue_is_recovered_without_overwriting_newer_updates(tmp_path):
    q = tmp_path / "caption_updates.json"
    processing = tmp_path / "caption_updates.json.processing"
    processing.write_text('{"a": "older", "b": "recovered"}', encoding="utf-8")
    q.write_text('{"a": "newer"}', encoding="utf-8")
    loss_watch._recover_processing(str(q), {"c": "detected"})
    assert json.loads(q.read_text(encoding="utf-8")) == {
        "a": "newer", "b": "recovered", "c": "detected"
    }
    assert not processing.exists()


def test_edited_still_is_reencoded_into_same_cache_path(tmp_path, monkeypatch):
    from PIL import Image
    import torch
    from safetensors.torch import load_file, save_file

    root = tmp_path / "dataset"
    root.mkdir()
    Image.new("RGB", (32, 32), "blue").save(root / "photo.png")
    cache_path = tmp_path / "photo_cache.safetensors"
    save_file({"latent_4x4": torch.zeros(4, 4, 4)}, str(cache_path))
    output = tmp_path / "run"
    q = output / "loss_log" / "image_updates.json"
    q.parent.mkdir(parents=True)
    q.write_text('{"photo": "updated"}', encoding="utf-8")

    class Driver:
        description = SimpleNamespace(arch_id="krea2", rgba=False, text_cache_trim=False)
        def load_vae(self, path, device):
            return object()
        def encode_images(self, vae, images):
            assert images[0].shape[:2] == (32, 32)
            return [torch.ones(4, 4, 4)]

    item = SimpleNamespace(item_key="photo", bucket_size=(32, 32),
                           original_size=(32, 32), latent_cache_path=str(cache_path))
    monkeypatch.setattr(loss_watch, "_items", lambda group: {"photo": item})
    watch = loss_watch.Watch.__new__(loss_watch.Watch)
    watch.group = object()
    watch.output_dir = str(output)
    watch.folders = {"": str(root)}
    watch.image_dir = str(root)
    watch.vae_path = str(tmp_path / "vae")
    watch.driver = Driver()
    watch.removed_keys = set()
    watch._images(1, None, torch.device("cpu"), False)
    assert watch.removed_keys == set()
    assert torch.equal(load_file(str(cache_path))["latent_4x4"], torch.ones(4, 4, 4))
    assert not q.exists()
    assert not (tmp_path / "photo_cache.safetensors.live-update").exists()
