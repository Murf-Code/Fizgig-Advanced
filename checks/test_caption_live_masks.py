"""CPU-only regression tests for masks edited, added or removed during active training."""
from types import SimpleNamespace

from PIL import Image

from fizgig.families import loss_watch


def setup_watch(tmp_path, *, prefix="", other_mask_folder=False):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (40, 32), "blue").save(images / "photo.png")
    mask_folder = tmp_path / "mask_inputs" if other_mask_folder else images
    if other_mask_folder:
        mask_folder.mkdir()
    item = SimpleNamespace(item_key=prefix + "photo", original_size=(40, 32),
                           bucket_size=(32, 32), mask_path=None)
    dataset = SimpleNamespace(
        image_directory=str(images),
        mask_directory=str(mask_folder) if other_mask_folder else None,
        item_prefix=prefix, loss_masks=True, is_reg=False,
        batch_manager=SimpleNamespace(buckets={(32, 32): [item, item]}),
    )
    watch = loss_watch.Watch.__new__(loss_watch.Watch)
    watch.group = SimpleNamespace(datasets=[dataset])
    watch.removed_keys = set()
    watch.invalid_mask_keys = set()
    watch.reg = set()
    watch.watch = None
    watch._mask_snapshot = watch._mask_state()
    return watch, item, mask_folder


def test_new_mask_added_during_training_is_used_at_next_boundary(tmp_path):
    watch, item, folder = setup_watch(tmp_path)
    assert item.mask_path is None
    path = folder / "photo-masklabel.png"
    Image.new("L", (40, 32), 255).save(path)
    assert item.mask_path is None
    watch._refresh_masks()
    assert item.mask_path == str(path.resolve())
    assert not watch.excluded({"item_keys": ["photo"]})


def test_bad_mask_blocks_picture_until_corrected(tmp_path):
    watch, item, folder = setup_watch(tmp_path)
    path = folder / "photo-masklabel.png"
    Image.new("L", (8, 32), 255).save(path)
    watch._refresh_masks()
    assert item.mask_path is None
    assert watch.invalid_mask_keys == {"photo"}
    assert watch.excluded({"item_keys": ["photo"]})
    assert watch.multiplier({"item_keys": ["photo"]}) == 0.0
    Image.new("L", (40, 32), 255).save(path)
    watch._refresh_masks()
    assert item.mask_path == str(path.resolve())
    assert watch.invalid_mask_keys == set()
    assert not watch.excluded({"item_keys": ["photo"]})


def test_deleted_mask_switches_to_unmasked_loss(tmp_path):
    watch, item, folder = setup_watch(tmp_path)
    path = folder / "photo-masklabel.png"
    Image.new("L", (40, 32), 255).save(path)
    watch._refresh_masks()
    path.unlink()
    watch._refresh_masks()
    assert item.mask_path is None
    assert not watch.excluded({"item_keys": ["photo"]})


def test_secondary_dataset_mask_folder_and_key_prefix(tmp_path):
    watch, item, folder = setup_watch(tmp_path, prefix="s2:", other_mask_folder=True)
    path = folder / "photo-masklabel.png"
    Image.new("L", (40, 32), 255).save(path)
    watch._refresh_masks()
    assert item.mask_path == str(path.resolve())
    assert "s2:photo" in watch._mask_snapshot


def test_stale_mask_state_is_not_applied_mid_epoch(tmp_path):
    watch, item, folder = setup_watch(tmp_path)
    path = folder / "photo-masklabel.png"
    Image.new("L", (40, 32), 255).save(path)
    assert item.mask_path is None  # the watch applies file changes at a boundary
    watch._refresh_masks()
    assert item.mask_path is not None
