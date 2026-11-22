"""CPU-only checks for safely adding dataset pictures to a running fixed-step schedule."""
import random
import os
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from fizgig.dataset.image_dataset import BucketBatchManager, ItemInfo
from fizgig.families import loss_watch


class FakeDriver:
    supports_references = False
    description = SimpleNamespace(arch_id="krea2", text_cache_trim=False, edit_references=1)

    def __init__(self):
        self.text_hook = None

    def load_vae(self, path, device):
        return object()

    def encode_images(self, vae, images):
        return [torch.ones((4, 4, 4)) for _ in images]

    def load_text_encoder(self, path, device):
        return object()

    def encode_text(self, te, captions):
        if self.text_hook:
            self.text_hook()
        return [{"cond__mock": torch.ones(1, 4)} for _ in captions]

    def unload_text_encoder(self, te):
        pass


def live_setup(tmp_path):
    root = tmp_path / "pictures"
    root.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    Image.new("RGB", (256, 256), "blue").save(root / "old.png")
    (root / "old.txt").write_text("old concept")
    (root / "new.txt").write_text("new concept")
    old = ItemInfo("old", "old concept", (256, 256), (256, 256))
    old.latent_cache_path = str(cache / "old_0256x0256_krea2.safetensors")
    old.text_encoder_output_cache_path = str(cache / "old_krea2_te.safetensors")
    save_file({"latent_4x4": torch.zeros(4, 4, 4)}, old.latent_cache_path)
    save_file({"cond__mock": torch.ones(1, 4)}, old.text_encoder_output_cache_path)
    bm = BucketBatchManager({(256, 256): [old]}, 1)
    dataset = SimpleNamespace(
        image_directory=str(root), caption_extension=".txt", mask_directory=None,
        batch_manager=bm, item_prefix="", num_repeats=1, architecture="krea2",
        cache_directory=str(cache), resolution=(256, 256), enable_bucket=False,
        bucket_no_upscale=False, reso_steps=16, loss_masks=False, is_reg=False,
        holdout_only=False, holdout_stems=set(), has_control=False, loss_weight=1.0,
        extra_extensions=(), datasource=SimpleNamespace(captions_optional=False),
    )
    dataset.get_latent_cache_path = lambda item: str(cache / (
        f"{os.path.splitext(os.path.basename(item.item_key))[0]}_{item.original_size[0]:04d}x{item.original_size[1]:04d}_krea2.safetensors"))
    dataset.get_text_encoder_output_cache_path = lambda item: str(
        cache / f"{os.path.splitext(os.path.basename(item.item_key))[0]}_krea2_te.safetensors")
    watch = loss_watch.Watch.__new__(loss_watch.Watch)
    watch.group = SimpleNamespace(datasets=[dataset])
    watch.driver = FakeDriver()
    watch.output_dir = str(tmp_path / "run")
    watch.image_dir = str(root)
    watch.folders = {"": str(root)}
    watch.caption_ext = ".txt"
    watch.vae_path = "mock-vae"
    watch.te_path = "mock-te"
    watch.folder_weights = {}
    watch.removed_keys = set()
    watch.invalid_mask_keys = set()
    watch.unsupported_media_keys = set()
    watch.stale_caption_keys = set()
    watch._pending_reference_text = {}
    watch.watch = None
    watch.reg = set()
    watch._source_snapshot = watch._source_state()
    return root, dataset, watch


def test_new_image_encodes_and_joins_existing_fixed_bucket(tmp_path):
    root, ds, watch = live_setup(tmp_path)
    Image.new("RGB", (256, 256), "orange").save(root / "new.png")
    assert watch._unscheduled_inputs() == ["new"]
    before = len(ds.batch_manager)
    watch._admit_new_images(1, None, torch.device("cpu"), False)
    assert len(ds.batch_manager) == before
    assert "new" in {it.item_key for it in ds.batch_manager.live_pools[(256, 256)]}
    assert watch._unscheduled_inputs() == []
    assert (tmp_path / "cache" / "new_0256x0256_krea2.safetensors").exists()
    assert (tmp_path / "cache" / "new_krea2_te.safetensors").exists()
    drawn = set()
    for seed in range(12):
        ds.batch_manager.shuffle(random.Random(seed))
        drawn.add(ds.batch_manager.buckets[(256, 256)][0].item_key)
        assert len(ds.batch_manager) == before
    assert drawn == {"old", "new"}


def test_image_without_caption_is_not_scheduled(tmp_path):
    root, ds, watch = live_setup(tmp_path)
    (root / "new.txt").unlink()
    Image.new("RGB", (256, 256)).save(root / "new.png")
    watch._admit_new_images(1, None, torch.device("cpu"), False)
    assert not ds.batch_manager.live_pools


def test_source_changed_while_encoding_is_not_admitted(tmp_path):
    root, ds, watch = live_setup(tmp_path)
    Image.new("RGB", (256, 256), "green").save(root / "new.png")
    def edit_mid_encode():
        (root / "new.txt").write_text("new caption changed during encoding")
    watch.driver.text_hook = edit_mid_encode
    watch._admit_new_images(1, None, torch.device("cpu"), False)
    assert not ds.batch_manager.live_pools
    assert watch._unscheduled_inputs() == ["new"]


def test_media_and_pair_edits_are_detected_for_secondary_dataset(tmp_path):
    root, ds, watch = live_setup(tmp_path)
    ds.item_prefix = "s2:"
    for it in ds.batch_manager.buckets[(256, 256)]:
        it.item_key = "s2:old"
    ds.has_control = True
    before = tmp_path / "before"
    before.mkdir()
    Image.new("RGB", (256, 256), "red").save(before / "old.png")
    ds.control_directory = str(before)
    watch._owner_cache = None
    watch._source_snapshot = watch._source_state()
    Image.new("RGB", (256, 256), "yellow").save(before / "old.png")
    changes, txt = watch._scan_sources()
    assert changes == {"s2:old": "updated"}
    assert txt == {}


def test_secondary_subject_new_cache_name_has_no_training_prefix(tmp_path):
    root, ds, watch = live_setup(tmp_path)
    ds.item_prefix = "s2:"
    for it in ds.batch_manager.buckets[(256, 256)]:
        it.item_key = "s2:old"
    watch._owner_cache = None
    watch._source_snapshot = watch._source_state()
    Image.new("RGB", (256, 256), "orange").save(root / "new.png")
    watch._admit_new_images(1, None, torch.device("cpu"), False)
    keys = {it.item_key for it in ds.batch_manager.live_pools[(256, 256)]}
    assert "s2:new" in keys
    new_item = next(it for it in ds.batch_manager.live_pools[(256, 256)] if it.item_key == "s2:new")
    assert os.path.basename(new_item.latent_cache_path) == "new_0256x0256_krea2.safetensors"
    assert os.path.basename(new_item.text_encoder_output_cache_path) == "new_krea2_te.safetensors"
