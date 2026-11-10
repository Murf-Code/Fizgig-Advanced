"""Watermark-only regions, saved beside Make masks' training masks.

The training mask blacks out watermarks together with anything else the user hid (a character
mask, a painted edit). That mix can't be unpicked later, so when a watermark is found this
module writes a sidecar in the folder's watermarks/ sub-folder: which pixels were the
watermark, and what the training mask was there beforehand. The Remove tool uses that to
paint a red stroke, and after a fill it can put those mask pixels back without touching the
user's other edits.

The sidecar is a LA PNG, same size as the picture: L is the previous mask value (white if
there was no mask), A is 255 on the watermark and 0 everywhere else. Pictures themselves
don't change. watermarks/ and the backup folder removed_watermarks/ are Fizgig's own
sub-folders (families/launch.py) and are never trained on.
"""
import os

import numpy as np
from PIL import Image

from fizgig.dataset.masks import MASK_SUFFIX, MIN_COVER, mask_for, _EXTS as MASK_EXTS

WM_DIR = "watermarks"                 # <folder>/watermarks/<stem>.png
WM_DONE_DIR = "done"                  # after a successful Remove: watermarks/done/<stem>.png
BACKUP_DIR = "removed_watermarks"     # copies of the original pictures (and their masks)
DEFAULT_PAD = 3                       # extra px around the region when it's loaded into Remove
PAD_MIN, PAD_MAX = 2, 5


def clamp_pad(pad, default=DEFAULT_PAD):
    try:
        n = int(round(float(pad)))
    except (TypeError, ValueError):
        n = default
    return max(PAD_MIN, min(PAD_MAX, n))


def _folder(image_path):
    return os.path.dirname(str(image_path))


def _stem(image_path):
    return os.path.splitext(os.path.basename(str(image_path)))[0]


def _dir(image_path):
    return os.path.join(_folder(image_path), WM_DIR)


def active_path(image_path):
    """The sidecar still waiting to be removed, or the path it would be written to."""
    return os.path.join(_dir(image_path), _stem(image_path) + ".png")


def done_path(image_path):
    return os.path.join(_dir(image_path), WM_DONE_DIR, _stem(image_path) + ".png")


def has_active(image_path):
    return os.path.isfile(active_path(image_path))


def pad_cover(cover, pad=DEFAULT_PAD):
    """Grow a bool (H, W) cover by `pad` px (clamped to 2-5)."""
    import cv2
    n = clamp_pad(pad)
    m = np.asarray(cover).astype(bool)
    k = 2 * n + 1
    return cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)


def encode(cover, prior):
    """cover bool (H, W) and prior mask uint8 (H, W) -> a LA PIL image."""
    c = np.asarray(cover).astype(bool)
    p = np.asarray(prior, np.uint8)
    if p.shape[:2] != c.shape[:2]:
        p = np.asarray(Image.fromarray(p, "L").resize((c.shape[1], c.shape[0]), Image.BILINEAR))
    la = np.empty(c.shape + (2,), np.uint8)
    la[..., 0] = p
    la[..., 1] = np.where(c, 255, 0)
    return Image.fromarray(la, "LA")


def decode(im):
    """A sidecar image -> (cover bool, prior uint8), both (H, W)."""
    la = np.asarray(im.convert("LA"))
    return la[..., 1] > 127, la[..., 0].copy()


def read(image_path):
    """(cover, prior) of the active sidecar, or None when there isn't one."""
    path = active_path(image_path)
    if not os.path.isfile(path):
        return None
    with Image.open(path) as im:
        im.load()
        return decode(im)


def write_sidecar(image_path, cover, prior):
    """Write (or replace) the active sidecar. Drops a leftover done sidecar for the same picture.
    cover must have some True pixels; otherwise the sidecar is dropped."""
    c = np.asarray(cover).astype(bool)
    if not c.any():
        drop_sidecar(image_path)
        return None
    path = active_path(image_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    encode(c, prior).save(tmp, "PNG")
    os.replace(tmp, path)
    done = done_path(image_path)
    if os.path.isfile(done):
        os.remove(done)
        _rmdir_empty(os.path.dirname(done))
    return path


def drop_sidecar(image_path):
    """Remove the active sidecar and a done one, if either is there."""
    gone = 0
    for path in (active_path(image_path), done_path(image_path)):
        if os.path.isfile(path):
            os.remove(path)
            gone += 1
            _rmdir_empty(os.path.dirname(path))
    _rmdir_empty(_dir(image_path))
    return gone


def mark_done(image_path):
    """The watermark has been removed: move the active sidecar into watermarks/done/."""
    src = active_path(image_path)
    if not os.path.isfile(src):
        return None
    dest = done_path(image_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    os.replace(src, dest)
    _rmdir_empty(_dir(image_path))
    return dest


def restore_active(image_path):
    """Undo of mark_done: move watermarks/done/<stem>.png back to watermarks/."""
    src = done_path(image_path)
    if not os.path.isfile(src):
        return None
    dest = active_path(image_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    os.replace(src, dest)
    _rmdir_empty(os.path.dirname(src))
    return dest


def _rmdir_empty(path):
    try:
        if os.path.isdir(path) and not os.listdir(path):
            os.rmdir(path)
    except OSError:
        pass


def unique_dest(folder, name):
    """A path in `folder` for `name` that doesn't overwrite a file already there (name, then name (2), ...)."""
    os.makedirs(folder, exist_ok=True)
    dest, k = os.path.join(folder, name), 2
    while os.path.exists(dest):
        stem, ext = os.path.splitext(name)
        dest, k = os.path.join(folder, f"{stem} ({k}){ext}"), k + 1
    return dest


def backup_originals(image_path):
    """Copy the picture (and its training mask and watermark sidecar, if any) into <folder>/removed_watermarks/,
    never over a file already there. Returns the paths written. A second Remove of the same picture leaves the
    first copy (the true original) as it is."""
    import shutil
    folder = os.path.join(_folder(image_path), BACKUP_DIR)
    written = []
    name = os.path.basename(image_path)
    dest = os.path.join(folder, name)
    if not os.path.isfile(dest):
        os.makedirs(folder, exist_ok=True)
        shutil.copy2(image_path, dest)
        written.append(dest)
    mask = mask_for(image_path)
    if mask:
        mname = os.path.basename(mask)
        mdest = os.path.join(folder, mname)
        if not os.path.isfile(mdest):
            os.makedirs(folder, exist_ok=True)
            shutil.copy2(mask, mdest)
            written.append(mdest)
    sidecar = active_path(image_path)
    if os.path.isfile(sidecar):
        sdest = backup_sidecar_path(image_path)
        if not os.path.isfile(sdest):
            os.makedirs(os.path.dirname(sdest), exist_ok=True)
            shutil.copy2(sidecar, sdest)
            written.append(sdest)
    return written


def backup_path(image_path):
    return os.path.join(_folder(image_path), BACKUP_DIR, os.path.basename(str(image_path)))


def has_backup(image_path):
    return os.path.isfile(backup_path(image_path))


def backup_sidecar_path(image_path):
    """Where the watermark sidecar is kept inside removed_watermarks/ (watermarks/<stem>.png)."""
    return os.path.join(_folder(image_path), BACKUP_DIR, WM_DIR, _stem(image_path) + ".png")


def backup_mask_path(image_path):
    """The training mask copied into removed_watermarks/, if one is there."""
    folder = os.path.join(_folder(image_path), BACKUP_DIR)
    if not os.path.isdir(folder):
        return None
    stem = _stem(image_path) + MASK_SUFFIX
    low = {n.lower(): n for n in os.listdir(folder)}
    for ext in MASK_EXTS:
        real = low.get((stem + ext).lower())
        if real:
            return os.path.join(folder, real)
    return None


def resize_pair(cover, prior, size):
    """Stretch (cover, prior) to size (w, h): cover by nearest neighbour, prior the way training stretches a mask."""
    w, h = size
    if cover.shape[1] == w and cover.shape[0] == h:
        return cover, prior
    c = np.asarray(Image.fromarray(cover.astype(np.uint8) * 255, "L").resize((w, h), Image.NEAREST)) > 127
    p = np.asarray(Image.fromarray(prior, "L").resize((w, h), Image.BILINEAR))
    return c, p


def restore_mask_array(mask, cover, prior):
    """Put `prior` back on `mask` only where `cover` is True. All (H, W) uint8. The picture's other mask
    edits stay as they are."""
    out = np.array(mask, np.uint8, copy=True)
    c = np.asarray(cover).astype(bool)
    out[c] = np.asarray(prior, np.uint8)[c]
    return out


def mask_is_all_white(mask):
    """True when a training mask no longer hides anything (it can be deleted; the picture trains as normal)."""
    m = np.asarray(mask, np.uint8)
    return (not m.size) or (not (m < 255).any()) or (float(m.mean()) / 255.0 >= 1.0 - MIN_COVER)


def remaining_cover(cover, removed):
    """Watermark pixels not in `removed` (the hole that was filled)."""
    return np.asarray(cover).astype(bool) & ~np.asarray(removed).astype(bool)
