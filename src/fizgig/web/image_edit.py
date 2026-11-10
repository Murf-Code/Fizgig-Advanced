"""The picture editor's picture tools (Image Prep's Edit masks): Crop and Remove change the picture file itself.

Everything happens on the picture as it is seen (upright, a phone photo's rotation flag applied), as the mask editor
works, and the file is written back in its own format with its rotation flag reset. Crop crops the picture's masks
the same way, so they stay lined up. Remove rebuilds what was painted over from what is around it with LaMa
(dataset/lama.py, an Apache-2.0 model fetched on first use). When LaMa can't be downloaded or loaded, Remove says so
and the picture is left as it was.

Each change is a step kept in memory (the file's bytes before and after, its masks' too, at most UNDO_STEPS of
them), so the page can undo and redo it. Nothing is kept on disk and the steps go when Fizgig closes. Restore Original
Image puts back the first copy from removed_watermarks/ (made before Remove), as one of those steps; the copy itself
stays. Every write is atomic (a temporary file, then a replace), so a failed save never leaves a broken picture."""
import io
import itertools
import os
import threading
from collections import OrderedDict

import numpy as np
from PIL import Image

from fizgig.dataset.masks import MASK_SUFFIX, MASKS_DIR, mask_for, read_mask, _EXTS as MASK_EXTS
from fizgig.dataset import lama, wm_regions
from fizgig.dataset.lama import LamaUnavailable  # noqa: F401  (Remove's error when LaMa can't run)
from fizgig.utils.upright import ORIENTATION, upright, upright_size

UNDO_STEPS = 30          # image steps kept, across all pictures
MAX_WINDOW = 1024        # a Remove window bigger than this is worked on scaled down (LaMa's sweet spot)
FEATHER = 3              # px: the filled part fades into the picture over this much


# ---- files -------------------------------------------------------------------------------------------------------

def _picture(folder, name):
    from fizgig.web.mask_edit import _picture as pic
    return pic(folder, name)


def mask_files(picture_path):
    """Every mask file of a picture: beside it, and in the folder's masks/ sub-folder."""
    folder, base = os.path.split(picture_path)
    stem = os.path.splitext(base)[0] + MASK_SUFFIX
    out = []
    for where in (folder, os.path.join(folder, MASKS_DIR)):
        if not os.path.isdir(where):
            continue
        low = {n.lower(): n for n in os.listdir(where)}
        for ext in MASK_EXTS:
            real = low.get((stem + ext).lower())
            if real:
                out.append(os.path.join(where, real))
    return out


def sidecar_files(picture_path):
    """The picture's watermark sidecar, active or already applied (done), if the file is there."""
    return [p for p in (wm_regions.active_path(picture_path), wm_regions.done_path(picture_path))
            if os.path.isfile(p)]


def _read(path):
    with open(path, "rb") as f:
        return f.read()


def write_atomic(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".fzedit.part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _webp_lossless(data):
    head = data[:4096]
    return head[:4] == b"RIFF" and head[8:12] == b"WEBP" and b"VP8L" in head and b"VP8 " not in head


def encode_like(img, original_bytes):
    """`img` (upright) as bytes in the original file's format: JPEG at quality 95 with full colour detail, WEBP
    lossless when the original was (else quality 95), PNG and others as they are. The ICC profile and the EXIF data are
    kept, the rotation flag reset (the pixels are upright now)."""
    with Image.open(io.BytesIO(original_bytes)) as src:
        fmt = (src.format or "PNG").upper()
        icc = src.info.get("icc_profile")
        try:
            exif = src.getexif()
            if ORIENTATION in exif:
                exif[ORIENTATION] = 1
            exif = exif.tobytes() if len(exif) else None
        except Exception:
            exif = None
    kw = {}
    if icc:
        kw["icc_profile"] = icc
    if exif and fmt in ("JPEG", "WEBP", "PNG", "TIFF"):
        kw["exif"] = exif
    if fmt == "JPEG":
        if img.mode not in ("RGB", "L", "CMYK"):
            img = img.convert("RGB")
        kw.update(quality=95, subsampling=0)
    elif fmt == "WEBP":
        kw.update(lossless=True) if _webp_lossless(original_bytes) else kw.update(quality=95)
    elif fmt == "BMP" and img.mode not in ("RGB", "L", "P", "1"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


def open_upright(data):
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        return upright(im).copy()


# ---- undo / redo: each step's files before and after -------------------------------------------------------------

_steps = OrderedDict()
_ids = itertools.count(1)
_lock = threading.Lock()


def _remember(files_before, files_after):
    """files_*: {path: bytes or None (no file)} -> the step's id."""
    with _lock:
        sid = next(_ids)
        _steps[sid] = (files_before, files_after)
        while len(_steps) > UNDO_STEPS:
            _steps.popitem(last=False)
        return sid


def _put(files):
    for path, data in files.items():
        if data is None:
            if os.path.exists(path):
                os.remove(path)
        else:
            write_atomic(path, data)


def step(sid, direction):
    """Undo ("undo") or redo ("redo") the image step `sid`: its files go back to how they were before or after it."""
    with _lock:
        s = _steps.get(int(sid))
    if s is None:
        raise ValueError("That change is too far back to undo (only the last 30 picture changes are kept).")
    _put(s[0] if direction == "undo" else s[1])
    return {"ok": True}


def forget():
    with _lock:
        _steps.clear()


# ---- crop --------------------------------------------------------------------------------------------------------

def clamp_box(box, size):
    """(left, top, right, bottom) rounded and kept inside a (w, h) picture, at least 1 px each way."""
    w, h = size
    l, t, r, b = (int(round(float(v))) for v in box)
    l, r = sorted((l, r))
    t, b = sorted((t, b))
    l, t = max(0, min(l, w - 1)), max(0, min(t, h - 1))
    r, b = max(l + 1, min(r, w)), max(t + 1, min(b, h))
    return l, t, r, b


def crop_mask_bytes(data, pic_size, box):
    """A mask file's bytes cropped to `box` of the upright picture (a mask of another size is stretched to the picture
    first, as training reads it), in the mask file's own format."""
    with Image.open(io.BytesIO(data)) as m:
        m.load()
        fmt = m.format or "PNG"
        m = m.copy()
    if m.size != pic_size:
        m = m.resize(pic_size, Image.BILINEAR)
    m = m.crop(box)
    if fmt.upper() == "JPEG" and m.mode not in ("L", "RGB"):
        m = m.convert("L")
    buf = io.BytesIO()
    m.save(buf, fmt)
    return buf.getvalue()


def crop_sidecar_bytes(data, pic_size, box):
    """A watermark sidecar cropped to `box`: the cover is resized by nearest neighbour so it stays sharp."""
    with Image.open(io.BytesIO(data)) as m:
        m.load()
        m = m.convert("LA").copy()
    if m.size != pic_size:
        cover, prior = wm_regions.decode(m)
        cover, prior = wm_regions.resize_pair(cover, prior, pic_size)
        m = wm_regions.encode(cover, prior)
    m = m.crop(box)
    buf = io.BytesIO()
    m.save(buf, "PNG")
    return buf.getvalue()


def crop(folder, name, box):
    """Crop the picture to box (left, top, right, bottom on the upright picture), and its masks with it."""
    pic = _picture(folder, name)
    before = {pic: _read(pic)}
    img = open_upright(before[pic])
    box = clamp_box(box, img.size)
    if box == (0, 0) + img.size:
        raise ValueError("The crop box is the whole picture: nothing to crop.")
    after = {pic: encode_like(img.crop(box), before[pic])}
    for m in mask_files(pic):
        before[m] = _read(m)
        after[m] = crop_mask_bytes(before[m], img.size, box)
    for s in sidecar_files(pic):
        before[s] = _read(s)
        after[s] = crop_sidecar_bytes(before[s], img.size, box)
    _put(after)
    return {"ok": True, "step": _remember(before, after), "size": [box[2] - box[0], box[3] - box[1]]}


# ---- remove ------------------------------------------------------------------------------------------------------

def prepare_hole(mask):
    """A painted stroke (bool HxW) -> what to fill: anything the stroke closes in is filled too (a loop drawn round a
    thing removes all of it), then grown about 2 px so the thing's edge goes too."""
    import cv2
    m = np.asarray(mask).astype(bool)
    h, w = m.shape
    outside = np.ones((h + 2, w + 2), np.uint8)
    outside[1:-1, 1:-1] = (~m).astype(np.uint8)          # flood from the border through what isn't painted
    flood = outside.copy()
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 2)
    hole = m | (flood[1:-1, 1:-1] != 2)
    hole = cv2.dilate(hole.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return hole.astype(bool)


def window(hole, size):
    """The part of the picture to work on: the hole's box with a margin of 3/4 its longer side (32 to 512 px)."""
    ys, xs = np.nonzero(hole)
    l, r, t, b = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad = int(max(32, min(512, 0.75 * max(r - l, b - t))))
    w, h = size
    return max(0, l - pad), max(0, t - pad), min(w, r + pad), min(h, b + pad)


def lama_fill(rgb, hole, gpu=True):
    """LaMa rebuilds the hole (rgb's sides are multiples of 8). Raises lama.LamaUnavailable."""
    return lama.inpaint(rgb, hole, gpu=gpu)


def inpaint_rgb(rgb, stroke, fill=None, gpu=True):
    """rgb uint8 HxWx3 and the painted stroke (bool HxW) -> rgb with the stroke's things removed. Only a window round
    the stroke is worked on (scaled down past MAX_WINDOW px, padded to a multiple of 8); the filled part is pasted
    back fading into the picture over FEATHER px."""
    import cv2
    hole = prepare_hole(stroke)
    if not hole.any():
        return rgb.copy()
    h, w = hole.shape
    l, t, r, b = window(hole, (w, h))
    # filled a little past the hole, so the fade-out at its edge blends filled pixels, not the thing being removed
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * FEATHER + 1, 2 * FEATHER + 1))
    wide = cv2.dilate(hole.astype(np.uint8), grow).astype(bool)
    sub, sub_hole = rgb[t:b, l:r], wide[t:b, l:r]
    ww, wh = r - l, b - t
    k = min(1.0, MAX_WINDOW / max(ww, wh))
    if k < 1:
        sw, sh = max(8, int(ww * k)), max(8, int(wh * k))
        small = np.asarray(Image.fromarray(sub).resize((sw, sh), Image.LANCZOS))
        small_hole = np.asarray(Image.fromarray(sub_hole.astype(np.uint8) * 255).resize((sw, sh), Image.NEAREST)) > 0
    else:
        small, small_hole = sub, sub_hole
    ph, pw = (-small.shape[0]) % 8, (-small.shape[1]) % 8
    padded = np.pad(small, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    padded_hole = np.pad(small_hole, ((0, ph), (0, pw)), mode="reflect")
    filled = (fill or lama_fill)(padded, padded_hole, gpu=gpu)[:small.shape[0], :small.shape[1]]
    if k < 1:
        filled = np.asarray(Image.fromarray(filled).resize((ww, wh), Image.LANCZOS))
    # fade: 1 inside the hole, falling to 0 at the widened edge
    a = cv2.GaussianBlur(hole[t:b, l:r].astype(np.float32), (0, 0), FEATHER / 2.0)
    a = np.maximum(np.clip(a * 2.0, 0, 1), hole[t:b, l:r]) * sub_hole
    out = rgb.copy()
    blend = sub.astype(np.float32) * (1 - a[..., None]) + filled.astype(np.float32) * a[..., None]
    out[t:b, l:r] = np.clip(np.round(blend), 0, 255).astype(np.uint8)
    return out


def remove_image(img, stroke, gpu=True):
    """The upright PIL picture with the stroke's things removed; its mode and see-through parts kept."""
    mode = img.mode
    if mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
        mode = img.mode
    alpha = img.getchannel("A") if mode in ("RGBA", "LA") else None
    rgb = np.asarray(img.convert("RGB"))
    out = Image.fromarray(inpaint_rgb(rgb, stroke, gpu=gpu), "RGB")
    if mode in ("L", "LA"):
        out = out.convert("L")
    elif mode not in ("RGB", "RGBA"):
        out = out.convert(mode) if mode in ("CMYK",) else out
    if alpha is not None:
        out.putalpha(alpha)
    return out


def _stroke_from_png(stroke_png, size):
    """PNG bytes or a data URL -> bool (H, W) at `size` (w, h)."""
    import base64
    if isinstance(stroke_png, str):
        stroke_png = base64.b64decode(stroke_png.split(",", 1)[1] if "," in stroke_png else stroke_png)
    with Image.open(io.BytesIO(stroke_png)) as s:
        s.load()
        s = s.getchannel("A") if s.mode in ("RGBA", "LA") else s.convert("L")
        if s.size != size:
            s = s.resize(size, Image.NEAREST)
        return np.asarray(s) > 127


def _write_mask_png(mask):
    buf = io.BytesIO()
    Image.fromarray(np.asarray(mask, np.uint8), "L").save(buf, "PNG")
    return buf.getvalue()


def _apply_wm_aftermath(pic, hole, size, before, after):
    """After a Remove: if this picture has a watermark sidecar, restore the training mask only in the watermark
    pixels that were filled, shrink or finish the sidecar, and record those files in before/after for undo.
    Other mask edits are left as they are."""
    pair = wm_regions.read(pic)
    if pair is None or not hole.any():
        return {"restored": False, "finished": False}
    cover, prior = wm_regions.resize_pair(pair[0], pair[1], size)
    touched = cover & hole
    if not touched.any():
        return {"restored": False, "finished": False}
    sidecar = wm_regions.active_path(pic)
    before[sidecar] = _read(sidecar)
    mask_path = mask_for(pic)
    new_mask = None
    if mask_path:
        before[mask_path] = _read(mask_path)
        mask = np.asarray(read_mask(mask_path))
        if mask.shape[1] != size[0] or mask.shape[0] != size[1]:
            mask = np.asarray(Image.fromarray(mask, "L").resize(size, Image.BILINEAR))
        new_mask = wm_regions.restore_mask_array(mask, touched, prior)
        if wm_regions.mask_is_all_white(new_mask):
            after[mask_path] = None
        else:
            out = mask_path
            if not out.lower().endswith(".png"):
                out = os.path.join(os.path.dirname(pic),
                                   os.path.splitext(os.path.basename(pic))[0] + MASK_SUFFIX + ".png")
                if os.path.normcase(out) != os.path.normcase(mask_path):
                    after[mask_path] = None
            after[out] = _write_mask_png(new_mask)
    left = wm_regions.remaining_cover(cover, hole)
    if left.any():
        after[sidecar] = _write_la(wm_regions.encode(left, prior))
    else:
        after[sidecar] = None
        done = wm_regions.done_path(pic)
        after[done] = before[sidecar]     # the finished sidecar, for undo
        before.setdefault(done, None)
    return {"restored": new_mask is not None, "finished": not left.any(),
            "mask_removed": bool(new_mask is not None and wm_regions.mask_is_all_white(new_mask))}


def _write_la(im):
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def _commit_remove(pic, img, stroke, backup=False, gpu=True):
    """Fill `stroke` on the upright picture, apply watermark-mask aftermath, write the files, remember the step.
    The fill comes first, so when LaMa can't run (LamaUnavailable) nothing has been written or backed up."""
    if not stroke.any():
        raise ValueError("Paint over what to remove first.")
    before = {pic: _read(pic)}
    filled = remove_image(img, stroke, gpu=gpu)
    if backup:
        wm_regions.backup_originals(pic)
    after = {pic: encode_like(filled, before[pic])}
    hole = prepare_hole(stroke)
    info = _apply_wm_aftermath(pic, hole, img.size, before, after)
    _put(after)
    return {"ok": True, "step": _remember(before, after), "watermark": info}


def remove(folder, name, stroke_png, backup=False, gpu=True):
    """Remove what the stroke (PNG bytes or a data URL; painted = remove, any size: stretched to the picture) covers.
    gpu=False keeps LaMa on the CPU (a run is training)."""
    pic = _picture(folder, name)
    img = open_upright(_read(pic))
    stroke = _stroke_from_png(stroke_png, img.size)
    return _commit_remove(pic, img, stroke, backup=backup, gpu=gpu)


def watermark_stroke_png(folder, name, pad=wm_regions.DEFAULT_PAD):
    """The picture's active watermark region, grown by `pad` px, as a grayscale PNG (white = remove), or None."""
    pic = _picture(folder, name)
    pair = wm_regions.read(pic)
    if pair is None:
        return None
    img = open_upright(_read(pic))
    cover, _prior = wm_regions.resize_pair(pair[0], pair[1], img.size)
    cover = wm_regions.pad_cover(cover, pad)
    buf = io.BytesIO()
    Image.fromarray(cover.astype(np.uint8) * 255, "L").save(buf, "PNG")
    return buf.getvalue()


def remove_watermark(folder, name, pad=wm_regions.DEFAULT_PAD, backup=True, gpu=True):
    """Remove the picture's watermark sidecar (padded), restore the training mask there, and copy the original
    into removed_watermarks/ the first time. Raises ValueError when there is no watermark sidecar."""
    pic = _picture(folder, name)
    pair = wm_regions.read(pic)
    if pair is None:
        raise ValueError("This picture has no watermark mask to remove.")
    img = open_upright(_read(pic))
    cover, _prior = wm_regions.resize_pair(pair[0], pair[1], img.size)
    stroke = wm_regions.pad_cover(cover, pad)
    return _commit_remove(pic, img, stroke, backup=backup, gpu=gpu)



def restore_original(folder, name):
    """Put the picture (and its training mask and watermark sidecar, if they were copied) back from
    <folder>/removed_watermarks/. The backup itself is kept, so a later Restore can use it again.
    If no sidecar was backed up, a finished one is moved back to active when it still matches the
    picture's size. Records an undo step. Raises ValueError when there is no backup."""
    pic = _picture(folder, name)
    src = wm_regions.backup_path(pic)
    if not os.path.isfile(src):
        raise ValueError("This picture has no original to restore (there's no copy in removed_watermarks).")
    bak_pic = _read(src)
    before, after = {pic: _read(pic)}, {pic: bak_pic}
    size = open_upright(bak_pic).size

    bak_mask = wm_regions.backup_mask_path(pic)
    current_masks = mask_files(pic)
    if bak_mask:
        bak_name = os.path.basename(bak_mask)
        dest = next((m for m in current_masks if os.path.basename(m).lower() == bak_name.lower()),
                    os.path.join(os.path.dirname(pic), bak_name))
        before[dest] = _read(dest) if os.path.isfile(dest) else None
        after[dest] = _read(bak_mask)
        for m in current_masks:
            if os.path.normcase(m) != os.path.normcase(dest):
                before.setdefault(m, _read(m))
                after[m] = None
    else:
        for m in current_masks:
            data = _read(m)
            with Image.open(io.BytesIO(data)) as im:
                msize = im.size
            if msize != size:
                before[m] = data
                after[m] = None

    active, done = wm_regions.active_path(pic), wm_regions.done_path(pic)
    for s in (active, done):
        if os.path.isfile(s):
            before[s] = _read(s)

    def sidecar_size(data):
        with Image.open(io.BytesIO(data)) as im:
            return im.size

    bak_side = wm_regions.backup_sidecar_path(pic)
    if os.path.isfile(bak_side):
        after[active] = _read(bak_side)
        before.setdefault(active, before.get(active))
        if os.path.isfile(done) or done in before:
            before.setdefault(done, _read(done) if os.path.isfile(done) else None)
            after[done] = None
    else:
        done_ok = done in before and sidecar_size(before[done]) == size
        active_ok = active in before and sidecar_size(before[active]) == size
        if done_ok:
            after[done] = None
            after[active] = before[done]
            before.setdefault(active, None)
        else:
            if done in before:
                after[done] = None
            if active in before and not active_ok:
                after[active] = None

    changed = any(before.get(p) != after.get(p, before.get(p)) for p in set(before) | set(after))
    if not changed:
        return {"ok": True, "changed": False, "mask": bool(mask_for(pic)),
                "watermark": wm_regions.has_active(pic)}
    _put(after)
    return {"ok": True, "changed": True, "step": _remember(before, after),
            "mask": bool(mask_for(pic)), "watermark": wm_regions.has_active(pic)}


def save_watermark_edit(folder, name, add_png="", erase_png="", pad=wm_regions.DEFAULT_PAD):
    """Keep the user's red edits in Remove: `add_png` is what was painted on and `erase_png` what was rubbed out
    since the red area was loaded (PNG bytes or data URLs, white or opaque = changed, any size: stretched to the
    picture). The sidecar is kept unpadded: added pixels join the watermark, and everything within `pad` px of an
    erased pixel leaves it, so the next load (padded again) doesn't grow back into what was erased and saving the same
    area twice changes nothing. Erasing it all drops the sidecar. Not an undo step; allowed while training (training
    never reads watermarks/). Returns {"watermark": the picture still has a watermark mask}."""
    pic = _picture(folder, name)
    with Image.open(pic) as im:
        size = upright_size(im)
    add = _stroke_from_png(add_png, size) if add_png else np.zeros((size[1], size[0]), bool)
    erase = _stroke_from_png(erase_png, size) if erase_png else np.zeros((size[1], size[0]), bool)
    pair = wm_regions.read(pic)
    if pair is not None:
        cover, prior = wm_regions.resize_pair(pair[0], pair[1], size)
    else:
        cover, prior = np.zeros((size[1], size[0]), bool), None
    if not add.any() and not erase.any():
        return {"ok": True, "watermark": pair is not None and bool(cover.any()), "changed": False}
    new = cover | add
    if erase.any():
        new &= ~wm_regions.pad_cover(erase, pad)
    # what the training mask was before the watermark was hidden: kept where it was known; on added pixels the
    # training mask as it is now (nothing was hidden there for a watermark), so a later Remove leaves it alone
    mask_path = mask_for(pic)
    if mask_path:
        now = np.asarray(read_mask(mask_path))
        if now.shape[1] != size[0] or now.shape[0] != size[1]:
            now = np.asarray(Image.fromarray(now, "L").resize(size, Image.BILINEAR))
    else:
        now = np.full((size[1], size[0]), 255, np.uint8)
    out_prior = np.array(now, np.uint8, copy=True)
    if prior is not None:
        out_prior[cover] = np.asarray(prior, np.uint8)[cover]
    if new.any():
        wm_regions.write_sidecar(pic, new, out_prior)
    else:
        path = wm_regions.active_path(pic)
        if os.path.isfile(path):
            os.remove(path)
            wm_regions._rmdir_empty(os.path.dirname(path))
    return {"ok": True, "watermark": bool(new.any()), "changed": True}
