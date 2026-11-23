"""Cache preflight: skip clean stages and repair only stale cache classes."""
import os
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from fizgig.dataset.image_dataset import source_stamp
from fizgig.families.cache import encoder_stamp, vae_stamp
from fizgig.families.cache_preflight import scan


@pytest.fixture
def cached_still(tmp_path):
    img = tmp_path / "photos"
    cache = tmp_path / "cache"
    img.mkdir()
    cache.mkdir()
    picture = img / "one.png"
    Image.new("RGB", (64, 64), "navy").save(picture)
    caption = img / "one.txt"
    caption.write_text("lion, blue mane", encoding="utf8")
    vae = tmp_path / "vae.bin"
    text = tmp_path / "text.bin"
    vae.write_bytes(b"vae version 1")
    text.write_bytes(b"text version 1")
    desc = SimpleNamespace(media=("photo",), arch_id="krea2drv", bucket_step=16)
    config = (f'[general]\nresolution = [64, 64]\ncaption_extension = ".txt"\n'
              f'enable_bucket = true\nbucket_no_upscale = true\n'
              f'[[datasets]]\nimage_directory = "{img.as_posix()}"\n'
              f'cache_directory = "{cache.as_posix()}"\n')
    latent = cache / "one_0064x0064_krea2drv.safetensors"
    txtcache = cache / "one_krea2drv_te.safetensors"

    def save():
        save_file({"latent_8x8": torch.zeros(2, 8, 8)}, str(latent),
                  metadata={"architecture": "krea2drv", "source": source_stamp(str(picture)),
                            "vae_encoder": vae_stamp(str(vae))})
        save_file({"cond__hidden_states": torch.ones(1, 1)}, str(txtcache),
                  metadata={"architecture": "krea2drv", "caption1": caption.read_text(),
                            "text_encoder": encoder_stamp(str(text))})

    save()

    def inspect(cfg=None, held=None):
        return scan(desc, cfg or config, held_paths=held, vae_model=str(vae), text_model=str(text))

    return SimpleNamespace(image=picture, caption=caption, latent=latent, txtcache=txtcache,
                           cache=cache, vae=vae, text=text, config=config, inspect=inspect, save=save)


def test_complete_cache_requires_no_encoding(cached_still):
    assert cached_still.inspect() == {"checked": 1, "latents": 0, "text": 0}


def test_caption_change_only_invalidates_text(cached_still):
    cached_still.caption.write_text("lion, indigo mane", encoding="utf8")
    assert cached_still.inspect() == {"checked": 1, "latents": 0, "text": 1}


def test_image_edit_only_invalidates_latents(cached_still):
    s = os.stat(cached_still.image)
    os.utime(cached_still.image, ns=(s.st_atime_ns, s.st_mtime_ns + 2_000_000_000))
    assert cached_still.inspect() == {"checked": 1, "latents": 1, "text": 0}


def test_missing_cache_and_corrupt_header_are_detected(cached_still):
    cached_still.latent.unlink()
    assert cached_still.inspect()["latents"] == 1
    cached_still.save()
    cached_still.txtcache.write_bytes(b"not safetensors")
    assert cached_still.inspect()["text"] == 1


def test_changed_resolution_only_invalidates_latents(cached_still):
    # A larger target with upscaling requires different latent geometry.
    changed = cached_still.config.replace("resolution = [64, 64]", "resolution = [128, 128]")
    changed = changed.replace("bucket_no_upscale = true", "bucket_no_upscale = false")
    assert cached_still.inspect(changed) == {"checked": 1, "latents": 1, "text": 0}


def test_different_vae_invalidates_provenanced_latents(cached_still):
    cached_still.vae.write_bytes(b"vae version 2 is different")
    assert cached_still.inspect() == {"checked": 1, "latents": 1, "text": 0}


def test_different_text_encoder_invalidates_cached_caption(cached_still):
    cached_still.text.write_bytes(b"text version 2 is different")
    assert cached_still.inspect() == {"checked": 1, "latents": 0, "text": 1}


def test_holdout_cache_is_checked_in_its_separate_directory(cached_still):
    from fizgig.training.holdout import HOLDOUT_CACHE

    held_dir = cached_still.cache / HOLDOUT_CACHE
    held_dir.mkdir()
    cached_still.latent.replace(held_dir / cached_still.latent.name)
    cached_still.txtcache.replace(held_dir / cached_still.txtcache.name)
    assert cached_still.inspect(held=[str(cached_still.image)]) == {
        "checked": 1, "latents": 0, "text": 0
    }


def test_an_unhandled_caption_transform_runs_the_standard_stage(cached_still):
    transformed = cached_still.config.replace(
        'caption_extension = ".txt"', 'caption_extension = ".txt"\ntag_shuffle = true'
    )
    assert cached_still.inspect(transformed) is None


def test_removed_picture_caches_are_flagged_for_cleanup(cached_still):
    orphan = cached_still.cache / "deleted_0064x0064_krea2drv.safetensors"
    orphan.write_bytes(b"old")
    assert cached_still.inspect()["latents"] == 1


def test_missing_model_file_never_skips_expensive_stage(cached_still):
    cached_still.vae.unlink()
    assert cached_still.inspect() is None
