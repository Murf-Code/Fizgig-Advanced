"""Cheap, conservative cache validation for the Start plan.

This intentionally supports *plain still-image* datasets only. An uncertain, missing or stale
entry makes its stage run through the existing encoder's --skip_existing checks. A proven clean
stage can be omitted entirely, so the VAE / text encoder and image-decoding workers never start.

The inspection reads image headers and safetensors headers, never image pixels or tensor data.
Unlike the old resume check, it compares the source stamp, caption, bucket and encoder identity.
No persistent manifest: the cache files remain the source of truth (including across resumes).
"""
import os

import toml
from PIL import Image
from safetensors import safe_open

from fizgig.dataset.caption_file import read_caption
from fizgig.dataset.image_dataset import (
    BucketSelector, IMAGE_EXTENSIONS, ImageDataset, glob_images,
)
from fizgig.training.holdout import HOLDOUT_CACHE, held_stems
from fizgig.utils.upright import orientation

# A special encoding mode may need more than these two ordinary cache files. Fall back
# to the regular stage when the generated dataset requests one.
_SPECIAL = (
    "control_directory", "extra_reference_directory", "caption_replace",
    "tag_shuffle", "tag_dropout", "caption_dropout", "keep_words",
    "caption_variants", "caption_variant_cap", "clip_megapixels",
)


def _header(path):
    """Image dimensions as open_upright() supplies them, and the existing source stamp.

    PIL.open reads the header only. Orientation is the only operation that can affect
    dimensions without decoding the image. A corrupt/unreadable header triggers caching.
    """
    with Image.open(path) as im:
        w, h = im.size
        rotate = orientation(im)
    if rotate in (5, 6, 7, 8):
        w, h = h, w
    st = os.stat(path)
    stamp = f"{st.st_size}:{int(st.st_mtime)}"
    if rotate != 1:
        stamp += ":upright"
    return w, h, stamp


def _metadata(path):
    try:
        with safe_open(path, framework="pt") as f:
            return f.metadata() or {}, tuple(f.keys())
    except Exception:
        return None, ()


def _latent_ok(path, source, stamp, bucket, arch, vae_stamp):
    md, keys = _metadata(path)
    if md is None or not any(k.startswith("latent_") and not k.startswith("latent_control_") for k in keys):
        return False
    if md.get("architecture") != arch:
        return False
    if md.get("vae_encoder") and md["vae_encoder"] != vae_stamp:
        return False
    if md.get("source"):
        if md["source"] != stamp:
            return False
    else:
        # Legacy caches without a source stamp used file modification time.
        try:
            if os.stat(source).st_mtime > os.stat(path).st_mtime:
                return False
        except OSError:
            return False
    # Same validation as the encoder's --skip_existing, including patch/latent factor.
    return ImageDataset.latent_cache_matches_reso(path, bucket, arch) is True


def _text_ok(path, caption, arch, encoder_stamp):
    md, keys = _metadata(path)
    return (md is not None and md.get("architecture") == arch
            and any(k.startswith("cond__") for k in keys)
            and md.get("caption1") == caption and not md.get("reference_sizes")
            and (not md.get("text_encoder") or md["text_encoder"] == encoder_stamp))


def scan(desc, toml_text, *, held_paths=None, vae_model="", text_model=""):
    """Return {checked, latents, text}, or None when a fast decision would be unsafe.

    Counters are the *number of items needing a stage*, not the number of bytes to encode.
    Existing stage implementations still decide which particular items to encode and clean.
    The hold-out selection is supplied from this Start, since the on-disk hold-out list may
    still describe an earlier run (or have been emptied when a paused run stopped).
    """
    if set(getattr(desc, "media", ())) != {"photo"}:
        return None
    if not (vae_model and text_model and os.path.isfile(vae_model) and os.path.isfile(text_model)):
        return None
    try:
        parsed = toml.loads(toml_text)
        general = parsed.get("general") or {}
        sets = parsed.get("datasets") or []
        if not sets or any(general.get(k) for k in _SPECIAL):
            return None
        resolution = tuple(int(v) for v in general["resolution"])
        if len(resolution) != 2 or min(resolution) <= 0:
            return None
        ext = general.get("caption_extension") or ".txt"
        if not isinstance(ext, str) or not ext.startswith("."):
            return None
        enabled = bool(general.get("enable_bucket", True))
        no_upscale = bool(general.get("bucket_no_upscale", False))
        selector = BucketSelector(resolution, enabled, no_upscale, desc.bucket_step)
        from fizgig.families.cache import encoder_stamp, vae_stamp
        txt_stamp = encoder_stamp(text_model)
        latent_stamp = vae_stamp(vae_model)
        count = stale_lat = stale_text = 0
        accepted = {e.lower() for e in IMAGE_EXTENSIONS}
        for ds in sets:
            if any(ds.get(k) for k in _SPECIAL) or any(
                k in ds for k in ("resolution", "enable_bucket", "bucket_no_upscale")
            ):
                return None
            folder, cache_dir = ds.get("image_directory"), ds.get("cache_directory")
            if not folder or not cache_dir or not os.path.isdir(folder):
                return None
            # The same picture list as the dataset worker, including captionless filtering and
            # the superseded-image exclusion used by Image Prep.
            paths = glob_images(folder, caption_extension=ext)
            held = held_stems(general.get("holdout_file"), folder, held_paths)
            expected_lat, expected_text = set(), set()
            for source in paths:
                stem = os.path.splitext(os.path.basename(source))[0]
                actual_dir = os.path.join(cache_dir, HOLDOUT_CACHE) if stem in held else cache_dir
                w, h, stamp = _header(source)
                bucket = selector.get_bucket_resolution((w, h))
                lat = os.path.join(actual_dir, f"{stem}_{w:04d}x{h:04d}_{desc.arch_id}.safetensors")
                txt = os.path.join(actual_dir, f"{stem}_{desc.arch_id}_te.safetensors")
                expected_lat.add(os.path.normcase(os.path.normpath(lat)))
                expected_text.add(os.path.normcase(os.path.normpath(txt)))
                caption = read_caption(os.path.splitext(source)[0] + ext).strip()
                count += 1
                if not _latent_ok(lat, source, stamp, bucket, desc.arch_id, latent_stamp):
                    stale_lat += 1
                if not _text_ok(txt, caption, desc.arch_id, txt_stamp):
                    stale_text += 1

            # An orphan is a stale entry as well: an image removed from the dataset must
            # not leave extra caches behind. The encoder's cleanup handles these.
            directories = [cache_dir]
            if held:
                directories.append(os.path.join(cache_dir, HOLDOUT_CACHE))
            for directory in directories:
                if not os.path.isdir(directory):
                    continue
                for name in os.listdir(directory):
                    path = os.path.normcase(os.path.normpath(os.path.join(directory, name)))
                    if name.endswith(f"_{desc.arch_id}.safetensors") and path not in expected_lat:
                        stale_lat += 1
                    if name.endswith(f"_{desc.arch_id}_te.safetensors") and path not in expected_text:
                        stale_text += 1
        if count == 0:
            return None
        return {"checked": count, "latents": stale_lat, "text": stale_text}
    except Exception:
        # A metadata scan must never stop Start: the full cache stages already know how to
        # validate and repair their own dataset types. An unknown case uses that safe path.
        return None
