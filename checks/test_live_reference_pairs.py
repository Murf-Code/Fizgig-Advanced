"""CPU regressions for paired still-image and reference-aware text updates."""
from types import SimpleNamespace

from PIL import Image
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open

from fizgig.families import loss_watch
from fizgig.families import cache


class PairDriver:
    supports_references = True
    description = SimpleNamespace(
        arch_id="qwen_image21", key="qwen_image21", edit_references=2,
        rgba=False, text_cache_trim=False)

    def __init__(self):
        self.refs_seen = 0

    def load_vae(self, path, device):
        return object()

    def encode_images(self, vae, images):
        assert len(images) == 1
        return [torch.ones(4, 4, 4) * 7]

    def load_reference_text_encoder(self, path, device):
        return object()

    def encode_text_with_references(self, te, captions, references):
        assert captions == ["updated caption"]
        assert len(references[0]) == 1
        self.refs_seen += 1
        return [{"cond__test": torch.ones(1, 8) * 9}]

    def unload_text_encoder(self, te):
        pass


def paired_case(tmp_path):
    root = tmp_path / "photos"
    before = tmp_path / "before"
    caches = tmp_path / "caches"
    root.mkdir(); before.mkdir(); caches.mkdir()
    Image.new("RGB", (32, 32), "red").save(root / "photo.png")
    Image.new("RGB", (32, 32), "blue").save(before / "photo.png")
    (root / "photo.txt").write_text("updated caption")
    item = SimpleNamespace(item_key="s2:photo", caption="old caption",
        original_size=(32, 32), bucket_size=(32, 32),
        latent_cache_path=str(caches / "photo_0032x0032_qwen_image21.safetensors"),
        text_encoder_output_cache_path=str(caches / "photo_qwen_image21_te.safetensors"))
    save_file({"latent_4x4": torch.zeros(4, 4, 4),
               "latent_control_0_4x4": torch.zeros(4, 4, 4)}, item.latent_cache_path)
    cache.save_cond(PairDriver.description, item,
                    {"cond__test": torch.zeros(1, 8)}, refs="32x32")
    bm = SimpleNamespace(buckets={(32, 32): [item]})
    ds = SimpleNamespace(item_prefix="s2:", image_directory=str(root),
        control_directory=str(before), extra_reference_directory=None,
        batch_manager=bm, is_reg=False, has_control=True,
        caption_extension=".txt", extra_extensions=(),
        resolution=(32,32), enable_bucket=False, bucket_no_upscale=False, reso_steps=16)
    ds.get_latent_cache_path=lambda i: str(caches / (
        f"{i.item_key.split('/')[-1].split('.')[0]}_{i.original_size[0]:04d}x"
        f"{i.original_size[1]:04d}_qwen_image21.safetensors"))
    driver=PairDriver()
    watch=loss_watch.Watch.__new__(loss_watch.Watch)
    watch.group=SimpleNamespace(datasets=[ds]);watch.driver=driver
    watch.output_dir=str(tmp_path / "run");watch.image_dir=str(root)
    watch.folders={"s2:":str(root)};watch.caption_ext=".txt"
    watch.removed_keys=set();watch.invalid_mask_keys=set()
    watch.unsupported_media_keys=set();watch.stale_caption_keys=set()
    watch._pending_reference_text={};watch.auto_recaption=False
    watch.watch=None;watch.reg=set()
    watch.vae_path="vae-model";watch.te_path="encoder-model"
    return watch, item, driver, root, before


def test_reference_edits_rebuild_control_latent_and_vision_text(tmp_path):
    watch,item,driver,root,before=paired_case(tmp_path)
    watch._images(1,None,torch.device("cpu"),False,{"s2:photo":"updated"})
    latent=load_file(item.latent_cache_path)
    assert (latent["latent_control_0_4x4"]==7).all()
    assert "s2:photo" in watch.removed_keys
    assert watch._pending_reference_text=={"s2:photo":"updated caption"}
    watch._captions(1,None,torch.device("cpu"),False)
    assert driver.refs_seen==1
    assert "s2:photo" not in watch.removed_keys
    assert watch.stale_caption_keys == set()
    assert watch._pending_reference_text == {}
    with safe_open(item.text_encoder_output_cache_path,framework="pt") as f:
        assert f.metadata()["caption1"]=="updated caption"
        assert f.metadata()["reference_sizes"]=="32x32"
    assert (load_file(item.text_encoder_output_cache_path)["cond__test"]==9).all()


def test_pair_change_detected_without_target_edit(tmp_path):
    watch,item,driver,root,before=paired_case(tmp_path)
    watch._source_snapshot=watch._source_state()
    assert watch._scan_sources()==({}, {})
    Image.new("RGB", (32, 32), "green").save(before / "photo.png")
    imgs,caps=watch._scan_sources()
    assert imgs=={"s2:photo":"updated"}
    assert caps=={}


def test_missing_before_image_keeps_item_excluded_and_retry_queued(tmp_path):
    watch,item,driver,root,before=paired_case(tmp_path)
    (before / "photo.png").unlink()
    watch._images(1,None,torch.device("cpu"),False,{"s2:photo":"updated"})
    assert "s2:photo" in watch.removed_keys
    import json
    p=tmp_path / "run" / "loss_log" / "image_updates.json"
    assert json.loads(p.read_text())=={"s2:photo":"updated"}
    lat=load_file(item.latent_cache_path)
    assert not lat["latent_control_0_4x4"].any()


def test_newer_caption_during_encoding_is_not_committed(tmp_path):
    watch,item,driver,root,before=paired_case(tmp_path)
    watch._images(1,None,torch.device("cpu"),False,{"s2:photo":"updated"})
    original=driver.encode_text_with_references
    def modify(te,captions,references):
        (root / "photo.txt").write_text("a newer caption")
        return original(te,captions,references)
    driver.encode_text_with_references=modify
    import pytest
    with pytest.raises(ValueError,match="changed while being encoded"):
        watch._captions(1,None,torch.device("cpu"),False)
    assert "s2:photo" in watch.removed_keys
    assert item.caption=="old caption"
