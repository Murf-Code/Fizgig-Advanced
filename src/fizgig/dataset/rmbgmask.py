"""Soft character masks for masked training from Bria's RMBG-3.0 (https://huggingface.co/briaai/RMBG-3.0): background
removal done as a Fibo 2.0 edit. The picture goes in as the image to edit, a fixed instruction asks for an alpha matte,
and 4 steps later the model has painted a grayscale matte - white the subject, black the background, grey where fur,
hair, glass or motion blur is partly see-through. It keeps tufts and whiskers that segmentation models (RMBG-2.0,
anime-seg) shave off, which is what a furry character's mask needs.

It runs on Fizgig's own Fibo 2.0 driver (fibo2/driver.py): the same DiT code, the same Qwen3-VL text encoder (it sees
the picture, as an edit's text encoder does) and the same FLUX.2 autoencoder - nothing from ComfyUI or diffusers.
The transformer comes as one of:
- "fp8": 1038lab's single file for ComfyUI (RMBG-3.0-fp8.safetensors, 7.6 GB, float8 with a scale per tensor);
- "bf16": 1038lab's bf16 single file (15.1 GB);
- "diffusers": Bria's own sharded repo (briaai/RMBG-3.0 transformer/, gated: accept its form on Hugging Face and log
  in first).
Both single files are decoded to bf16 as they load, then Fizgig's INT8 (or NF4) and block swap fit them on the card.
1038lab's NVFP4 file is a Blackwell-only packed format the loader can't read. The text encoder and the autoencoder are
Fibo 2.0's (Comfy-Org/Fibo-2.0, not gated), the same files the Fibo 2.0 family downloads, so they're shared.

Licence: the weights are CC BY-NC 4.0 (Bria's non-commercial terms). Masks made with them are for non-commercial work
unless you have a licence from Bria.

Masks are written beside their pictures as photo-masklabel.png (dataset/masks.py), never over an existing one unless
asked. The matte is kept as it is by default - masked training reads grey as partial weight - or made black and white
at a threshold; Grow widens it (so fur edges stay inside) and Feather softens the outer edge. The raw mattes are kept
in the folder's rmbg3_mattes/ sub-folder, so the edge settings can be changed later on the CPU without the model."""
import gc
import math
import os
import threading
import time

import numpy as np
from PIL import Image

from fizgig.dataset.masks import MASK_SUFFIX, MIN_COVER, mask_for
from fizgig.utils.image import flatten
from fizgig.utils.upright import upright

# The fixed instruction (RMBG-3.0's model card): the model was trained on exactly this text and nothing else - any
# other gives a coloured picture instead of a matte
INSTRUCTION = ("Generate a detailed grayscale alpha matte. Map the opaque foreground to white and the background to "
               "black. Produce soft, anti-aliased grayscale gradients at the edges of the subject to represent fine "
               "details and transparency.")
STEPS = 4                       # distilled to 4 steps, no CFG
SEED = 0
MEGAPIXELS = 1.0                # Fibo 2.0's own size (the pipeline's default area); 0.5 is quicker and lighter
FOLDER = "rmbg-3.0"             # in the models folder
MATTES_DIR = "rmbg3_mattes"     # the raw mattes, beside the pictures: <folder>/rmbg3_mattes/<stem>.png
CHUNK = 32                      # pictures read by the text encoder before the transformer takes over (RAM per chunk)

VARIANTS = {
    "fp8": {"repo": "1038lab/RMBG-3.0", "file": "RMBG-3.0-fp8.safetensors", "gb": 7.57,
            "label": "FP8 (7.6 GB) - RTX 30, 40 and 50 cards"},
    "bf16": {"repo": "1038lab/RMBG-3.0", "file": "RMBG-3.0-bf16.safetensors", "gb": 15.14,
             "label": "BF16 (15.1 GB) - the full-precision single file"},
    "diffusers": {"repo": "briaai/RMBG-3.0", "file": "transformer", "gb": 16.4, "gated": True,
                  "label": "Bria's own repo (16.4 GB, gated: accept its form on Hugging Face and log in)"},
}
VARIANT_DEFAULT = "fp8"
UNSUPPORTED = {"nvfp4": "1038lab's NVFP4 file is a packed 4-bit format for Blackwell cards in ComfyUI; Fizgig can't "
                        "read it. Use the FP8 file - Fizgig's own INT8 or NF4 then fits it on the card."}
# Fibo 2.0's text encoder and autoencoder, as the Fibo 2.0 family fetches them (families/fibo2.py: the same names in
# the models folder, so one download serves both)
TEXT_ENCODER = {"repo": "Comfy-Org/Fibo-2.0", "file": "text_encoders/qwen3vl_4b_bf16.safetensors",
                "local": "fibo2_text_encoder.safetensors", "gb": 8.88}
VAE = {"repo": "Comfy-Org/Fibo-2.0", "file": "vae/flux2-vae.safetensors", "local": "flux2-vae.safetensors",
       "gb": 0.34}
KINDS = {"subject": "The subject (the character, fur and all)",
         "background": "Just the background, not the subject"}
# the DiT's size on the card at each precision (8.2B parameters; the I/O layers stay bf16), and what a 1 MP render
# needs on top (activations, the autoencoder, CUDA's own)
DIT_GB = {"bf16": 16.4, "int8": 8.6, "nf4": 4.9}
HEADROOM_GB = 2.0
HEADROOM_PER_MP = 1.5
N_BLOCKS = 44

_lock = threading.Lock()
_use = threading.Lock()                 # one preview or mask run on the models at a time
_fetch = {"state": "", "error": "", "at": 0.0, "variant": ""}


# ---- the files ---------------------------------------------------------------------------------------------------
def model_dir(models_dir):
    return os.path.join(models_dir, FOLDER)


def find_model(models_dir, variant=VARIANT_DEFAULT):
    """The transformer of a variant in the models folder ("" before the download): a file, or the diffusers repo's
    transformer/ dir."""
    v = VARIANTS.get(variant)
    if v is None:
        return ""
    if variant == "diffusers":
        d = os.path.join(model_dir(models_dir), "briaai-RMBG-3.0", "transformer")
        return d if os.path.isfile(os.path.join(d, "diffusion_pytorch_model.safetensors.index.json")) else ""
    p = os.path.join(model_dir(models_dir), v["file"])
    return p if os.path.isfile(p) else ""


def find_part(models_dir, part):
    """Fibo 2.0's text encoder (TEXT_ENCODER) or autoencoder (VAE) in the models folder, or ""."""
    p = os.path.join(models_dir, part["local"])
    return p if os.path.isfile(p) else ""


def missing(models_dir, variant=VARIANT_DEFAULT):
    """[(what, GB)] still to download for a variant."""
    out = []
    if not find_model(models_dir, variant):
        out.append((f"RMBG-3.0 {variant}", VARIANTS[variant]["gb"]))
    if not find_part(models_dir, TEXT_ENCODER):
        out.append(("Fibo 2.0 text encoder", TEXT_ENCODER["gb"]))
    if not find_part(models_dir, VAE):
        out.append(("Fibo 2.0 autoencoder", VAE["gb"]))
    return out


def download(models_dir, variant=VARIANT_DEFAULT):
    """Fetch what a variant still needs (the transformer, the text encoder, the autoencoder) on a worker thread;
    fetch_state() follows it."""
    if variant not in VARIANTS:
        raise ValueError(UNSUPPORTED.get(variant) or f"unknown RMBG-3.0 variant {variant!r}")
    with _lock:
        if _fetch["state"] == "running":
            return
        _fetch.update(state="running", error="", at=time.time(), variant=variant)

    def work():
        from fizgig.utils import hf_public
        try:
            if not find_model(models_dir, variant):
                v = VARIANTS[variant]
                if variant == "diffusers":
                    hf_public.snapshot(repo_id=v["repo"], allow_patterns=["transformer/*"],
                                       local_dir=os.path.join(model_dir(models_dir), "briaai-RMBG-3.0"))
                else:
                    hf_public.download(repo_id=v["repo"], filename=v["file"], local_dir=model_dir(models_dir))
            for part in (VAE, TEXT_ENCODER):
                if not find_part(models_dir, part):
                    got = hf_public.download(repo_id=part["repo"], filename=part["file"], local_dir=models_dir)
                    dest = os.path.join(models_dir, part["local"])
                    if os.path.normcase(os.path.abspath(got)) != os.path.normcase(os.path.abspath(dest)):
                        os.replace(got, dest)   # from the repo's sub-folder to the name the Fibo 2.0 family uses
            with _lock:
                _fetch.update(state="done", at=time.time())
        except Exception as e:
            with _lock:
                _fetch.update(state="failed", error=hf_public.friendly_error(e), at=time.time())
    threading.Thread(target=work, daemon=True).start()


def fetch_state(models_dir):
    """{"state", "error", "gb", "of", "variant"}: how the download is going (GB of the part files so far)."""
    with _lock:
        st = dict(_fetch)
    gb = 0.0
    if st["state"] == "running":
        for root in (model_dir(models_dir), models_dir):
            d = os.path.join(root, ".cache", "huggingface", "download")
            for base, _dirs, files in os.walk(d) if os.path.isdir(d) else ():
                for n in files:
                    if n.endswith(".incomplete"):
                        try:
                            gb += os.path.getsize(os.path.join(base, n)) / 2 ** 30
                        except OSError:
                            pass
    of = sum(g for _w, g in missing(models_dir, st["variant"])) if st["variant"] in VARIANTS else 0.0
    return {"state": st["state"], "error": st["error"], "gb": round(gb, 2), "of": round(of, 2),
            "variant": st["variant"]}


def cuda_here():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def info(models_dir):
    """The masks card's RMBG-3.0 line."""
    return {"variants": {k: {"label": v["label"], "gb": v["gb"], "path": find_model(models_dir, k),
                             "gated": bool(v.get("gated")),
                             "missing": [[w, g] for w, g in missing(models_dir, k)]} for k, v in VARIANTS.items()},
            "default": VARIANT_DEFAULT, "kinds": KINDS, "fetch": fetch_state(models_dir), "gpu": cuda_here(),
            "unsupported": UNSUPPORTED}


# ---- the picture's crop, as the pipeline makes it, and back -----------------------------------------------------
def work_size(width, height, megapixels=MEGAPIXELS, multiple=16):
    """The (w, h) Fibo 2.0 renders a picture of this shape at (diffusers BriaFibo2ImageProcessor.get_image_size): of
    the sizes divisible by `multiple`, at least 4 of it a side and within 10% of the area, the one that crops the
    least from the picture, then the one closest to the area."""
    area = int(round(float(megapixels) * 1024 * 1024))
    aspect = float(width) / float(height)
    lo_side = 4 * multiple
    best, best_key = None, None
    for w in range(lo_side, 2 * area // lo_side, multiple):
        h_lo = max(lo_side, int(math.ceil(0.9 * area / w / multiple)) * multiple)
        for h in range(h_lo, 2 * area // w, multiple):
            if w * h > 1.1 * area:
                break
            if w * h < 0.9 * area:
                continue
            r = w / h
            key = (round(1 - min(r / aspect, aspect / r), 9), abs(w * h - area), -w)
            if best_key is None or key < best_key:
                best, best_key = (w, h), key
    return best


def cover_crop(img, width, height):
    """The picture scaled to cover width x height (aspect kept) and its centre cut out, as Fibo 2.0's training and
    pipeline cut it: (the crop, (scaled w, scaled h, left, top)) for uncrop."""
    sw, sh = img.size
    s = max(height / sh, width / sw)
    rw, rh = int(math.ceil(sw * s)), int(math.ceil(sh * s))
    left, top = (rw - width) // 2, (rh - height) // 2
    big = img.convert("RGB").resize((rw, rh), Image.LANCZOS)
    return big.crop((left, top, left + width, top + height)), (rw, rh, left, top)


def uncrop(matte, geom, size):
    """A matte of the crop (h, w) float in [0, 1] -> the whole picture's (H, W): the slivers the crop cut off take
    the matte's nearest edge, then it's scaled back to the picture's own size."""
    import cv2
    rw, rh, left, top = geom
    h, w = matte.shape
    full = cv2.copyMakeBorder(np.ascontiguousarray(matte, np.float32), top, max(0, rh - h - top), left,
                              max(0, rw - w - left), cv2.BORDER_REPLICATE)
    W, H = size
    shrink = W < full.shape[1] or H < full.shape[0]
    out = cv2.resize(full, (int(W), int(H)), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---- the edge settings -------------------------------------------------------------------------------------------
def finish(matte, kind="subject", threshold=None, grow=0, feather=0):
    """The mask from a matte (H, W) float in [0, 1]. threshold: None keeps the matte's greys (masked training reads
    them as partial weight), a number makes it black and white there (0.5 = half see-through counts). grow: pixels
    the white widens by, the greys moving out with it (fur edges stay inside). feather: pixels of soft edge added
    outside the white, never eating into it. kind "background": the opposite, after the rest (Grow shrinks the
    background)."""
    import cv2
    m = np.clip(np.asarray(matte, np.float32), 0.0, 1.0)
    if threshold is not None:
        m = (m >= float(threshold)).astype(np.float32)
    g = int(round(float(grow or 0)))
    if g > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
    f = float(feather or 0)
    if f > 0:
        m = np.maximum(m, cv2.GaussianBlur(m, (0, 0), f))
    if kind == "background":
        m = 1.0 - m
    elif kind != "subject":
        raise ValueError(f"unknown RMBG-3.0 mask kind {kind!r}")
    return np.clip(m, 0.0, 1.0).astype(np.float32)


def settings(kind="subject", threshold=None, grow=0, feather=0):
    """The edge settings, checked: {"kind", "threshold" (None = soft), "grow", "feather"}."""
    if kind not in KINDS:
        raise ValueError(f"unknown RMBG-3.0 mask kind {kind!r}")
    t = None if threshold in (None, "", "soft") else float(threshold)
    if t is not None and not 0.0 < t < 1.0:
        raise ValueError("the threshold is between 0 and 1 (0.5: half see-through counts)")
    g, f = int(round(float(grow or 0))), float(feather or 0)
    if not 0 <= g <= 64 or not 0 <= f <= 32:
        raise ValueError("Grow is 0 to 64 pixels, Feather 0 to 32")
    return {"kind": kind, "threshold": t, "grow": g, "feather": f}


def describe(s):
    edge = "soft edges" if s["threshold"] is None else f"black and white at {s['threshold']:.2f}"
    return (f"{KINDS[s['kind']].lower()}, {edge}" + (f", grown {s['grow']} px" if s["grow"] else "")
            + (f", feathered {s['feather']:g} px" if s["feather"] else ""))


# ---- the model ---------------------------------------------------------------------------------------------------
def plan(free_gb, precision="auto", blocks_to_swap=-1, megapixels=MEGAPIXELS):
    """(precision, blocks to swap) for the DiT on a card with free_gb free: Auto takes bf16 when it fits whole, else
    INT8 with the fewest blocks streamed that fit; NF4 never streams (its weights can't swap)."""
    need = HEADROOM_GB + HEADROOM_PER_MP * float(megapixels)
    if precision == "auto":
        precision = "bf16" if free_gb >= DIT_GB["bf16"] + need else "int8"
    if precision not in DIT_GB:
        raise ValueError(f"unknown precision {precision!r} (auto, bf16, int8 or nf4)")
    if precision == "nf4":
        return precision, 0
    if blocks_to_swap is not None and int(blocks_to_swap) >= 0:
        return precision, min(int(blocks_to_swap), N_BLOCKS - 2)
    short = DIT_GB[precision] + need - float(free_gb)
    if short <= 0:
        return precision, 0
    per = DIT_GB[precision] * 0.97 / N_BLOCKS        # the 44 main layers are nearly all of it
    return precision, min(N_BLOCKS - 2, int(math.ceil(short / per)))


class Matter:
    """RMBG-3.0 on Fizgig's Fibo 2.0 driver: mattes(pictures) -> (path, matte (H, W) float32 or None, note) per
    picture. Pictures go a chunk at a time: the text encoder (8.9 GB) reads them all, is freed, then the DiT renders
    them - the two never share the card."""

    def __init__(self, dit_path, te_path, vae_path, *, device=None, precision="auto", blocks_to_swap=-1,
                 megapixels=MEGAPIXELS, steps=STEPS, seed=SEED, chunk=CHUNK, log=print, driver=None):
        import torch
        if driver is None:
            from fizgig.families import registry
            driver = registry.get("fibo2").load_driver()
        self.driver = driver
        self.paths = (dit_path, te_path, vae_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.precision, self.blocks = precision, blocks_to_swap
        self.megapixels, self.steps, self.seed = float(megapixels), int(steps), int(seed)
        self.chunk = max(1, int(chunk))
        self.log = log
        import json
        self.prompt = json.dumps({"edit_instruction": INSTRUCTION})

    def _free(self):
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_dit(self):
        from fizgig.families import quant
        free = quant.free_vram_gb() if self.device.type == "cuda" else 0.0
        precision, swap = plan(free, self.precision, self.blocks, self.megapixels) if self.device.type == "cuda" \
            else ("bf16" if self.precision == "auto" else self.precision, 0)
        self.log(f"Loading RMBG-3.0 ({precision}" + (f", {swap} of {N_BLOCKS} blocks streamed from RAM" if swap else "")
                 + f"; {free:.1f} GB free on the card) ...\n")
        dit, swapped = quant.load_base(self.driver, self.paths[0], self.device, precision, swap,
                                       supports_backward=False)
        if swapped:
            self.driver.block_swap_mode(dit, inference=True)
        return dit

    def mattes(self, pictures):
        import torch
        pictures = list(pictures)
        vae = None
        try:
            for at in range(0, len(pictures), self.chunk):
                part = pictures[at:at + self.chunk]
                items = []
                for p in part:
                    try:
                        with Image.open(p) as im:
                            img = flatten(upright(im))
                    except Exception as e:
                        yield p, None, f"can't be read ({e})"
                        continue
                    w, h = work_size(*img.size, megapixels=self.megapixels)
                    crop, geom = cover_crop(img, w, h)
                    items.append({"path": p, "size": img.size, "wh": (w, h), "crop": crop, "geom": geom})
                if not items:
                    continue
                with torch.no_grad():
                    self.log("Reading the pictures with the text encoder ...\n")
                    te = self.driver.load_text_encoder(self.paths[1], self.device)
                    try:
                        for it in items:
                            it["cond"] = self.driver.encode_text_with_references(te, [self.prompt], [[it["crop"]]])[0]
                    finally:
                        self.driver.unload_text_encoder(te)
                        del te
                        self._free()
                    if vae is None:
                        vae = self.driver.load_vae(self.paths[2], self.device)
                    for it in items:
                        it["ref"] = self.driver.encode_images(vae, [np.asarray(it.pop("crop"))])[0][None].cpu()
                    dit = self._load_dit()
                    try:
                        for it in items:
                            w, h = it["wh"]
                            x = self.driver.generate(dit, it.pop("cond"), w, h, steps=self.steps, seed=self.seed,
                                                     cfg=1.0, refs=[it.pop("ref")])
                            img = self.driver.decode(vae, x, w, h).convert("L")
                            if img.size != (w, h):
                                img = img.resize((w, h), Image.BILINEAR)
                            m = np.asarray(img, np.float32) / 255.0
                            yield it["path"], uncrop(m, it["geom"], it["size"]), ""
                    finally:
                        del dit
                        self._free()
        finally:
            vae = None
            self._free()


class CharacterFallback:
    """The fallback when RMBG-3.0 can't run here (no NVIDIA card, its files not downloaded): the masks card's own
    character finder (charmask.py, SkyTNT anime-seg, 176 MB, CPU or GPU) - a segmentation matte, harder at the fur
    than RMBG-3.0's, through the same edge settings."""

    def __init__(self, models_dir, log=print):
        from fizgig.dataset import charmask
        path = charmask.find_model(models_dir)
        if not path:
            raise ValueError("the character finder isn't downloaded either: press its Download on the masks card")
        self.finder = charmask.CharacterFinder(path)
        self.log = log

    def mattes(self, pictures):
        try:
            for p in pictures:
                try:
                    with Image.open(p) as im:
                        img = flatten(upright(im))
                except Exception as e:
                    yield p, None, f"can't be read ({e})"
                    continue
                yield p, self.finder.mask(img), ""
        finally:
            self.finder.close()


def why_not(models_dir, variant=VARIANT_DEFAULT):
    """Why RMBG-3.0 can't run here (a message), or ""."""
    if variant in UNSUPPORTED:
        return UNSUPPORTED[variant]
    if variant not in VARIANTS:
        return f"unknown RMBG-3.0 variant {variant!r}"
    if not cuda_here():
        return "RMBG-3.0 needs an NVIDIA card (an 8B model: minutes a picture on the CPU)."
    gone = missing(models_dir, variant)
    if gone:
        return "Download RMBG-3.0 first: " + ", ".join(f"{w} ({g:g} GB)" for w, g in gone) + " still to fetch."
    return ""


def matter_for(models_dir, variant=VARIANT_DEFAULT, *, fallback=False, dit=None, te=None, vae=None, log=print,
               **kw):
    """The Matter for a variant (explicit dit / te / vae paths win over the models folder), or with fallback=True the
    character finder when RMBG-3.0 can't run here. Returns (matter, what it is)."""
    dit = dit or find_model(models_dir, variant)
    te = te or find_part(models_dir, TEXT_ENCODER)
    vae = vae or find_part(models_dir, VAE)
    why = "" if (dit and te and vae and cuda_here()) else (why_not(models_dir, variant) or
                                                           "RMBG-3.0's files weren't found")
    if why:
        if not fallback:
            raise ValueError(why)
        log(f"{why}\nUsing the character finder (anime-seg) instead.\n")
        return CharacterFallback(models_dir, log), "the character finder (fallback)"
    return Matter(dit, te, vae, log=log, **kw), f"RMBG-3.0 ({variant})"


def matte_path(folder, picture):
    return os.path.join(folder, MATTES_DIR, os.path.splitext(os.path.basename(picture))[0] + ".png")


def write_mask(picture, m, overwrite=True):
    """Save a mask beside its picture (photo-masklabel.png), replacing the picture's old one. Returns its path."""
    folder = os.path.dirname(picture)
    out = os.path.join(folder, os.path.splitext(os.path.basename(picture))[0] + MASK_SUFFIX + ".png")
    old = mask_for(picture)
    if old and os.path.normcase(old) != os.path.normcase(out):
        os.remove(old)                               # replaced: one mask per picture
    Image.fromarray((np.clip(m, 0, 1) * 255).round().astype(np.uint8), "L").save(out)
    return out


def make_folder(folder, models_dir, *, variant=VARIANT_DEFAULT, kind="subject", threshold=None, grow=0, feather=0,
                overwrite=False, keep_mattes=True, fallback=False, matter=None, log=print, **kw):
    """Write a mask for every picture in a folder, as automask.make_folder does. matter: a ready Matter (tests, the
    CLI's own paths); else matter_for(). keep_mattes: the raw mattes go to rmbg3_mattes/ for refinish_folder.
    Returns {"made", "kept", "skipped", "no_person"} (+ "stopped")."""
    from fizgig.dataset.automask import pictures
    from fizgig.dataset.masks import STOP
    s = settings(kind, threshold, grow, feather)
    counts = {"made": 0, "kept": 0, "skipped": 0, "no_person": 0}
    pics = pictures(folder)
    todo = []
    for p in pics:
        if mask_for(p) and not overwrite:
            counts["kept"] += 1
        else:
            todo.append(p)
    if not todo:
        return counts
    with _use:
        if matter is None:
            matter, what = matter_for(models_dir, variant, fallback=fallback, log=log, **kw)
        else:
            what = "RMBG-3.0"
        log(f"{len(todo)} picture(s) with {what}: {describe(s)}\n")
        done = 0
        gen = matter.mattes(todo)
        try:
            for p, matte, note in gen:
                done += 1
                name = os.path.basename(p)
                if matte is None:
                    log(f"  {name}: {note} - skipped\n")
                    counts["skipped"] += 1
                else:
                    if keep_mattes:
                        mp = matte_path(folder, p)
                        os.makedirs(os.path.dirname(mp), exist_ok=True)
                        Image.fromarray((matte * 255).round().astype(np.uint8), "L").save(mp)
                    m = finish(matte, **s)
                    if float(m.mean()) < MIN_COVER:
                        log(f"  {name}: nothing found - no mask written (it trains as normal)\n")
                        counts["no_person"] += 1
                    else:
                        write_mask(p, m)
                        counts["made"] += 1
                        log(f"  {name} -> {os.path.splitext(name)[0] + MASK_SUFFIX}.png "
                            f"({float(m.mean()) * 100:.0f}% learned)\n")
                if done % 20 == 0:
                    log(f"{done} of {len(todo)} pictures\n")
                if STOP.is_set():
                    counts["stopped"] = (done + counts["kept"], len(pics))
                    break
        finally:
            gen.close()
    return counts


def refinish_folder(folder, *, kind="subject", threshold=None, grow=0, feather=0, overwrite=True, log=print):
    """New masks from the mattes already in rmbg3_mattes/ with other edge settings: the CPU only, no model. A picture
    with no saved matte is left alone. Returns make_folder's counts."""
    from fizgig.dataset.automask import pictures
    from fizgig.dataset.masks import stop_here
    from fizgig.utils.upright import upright_size
    s = settings(kind, threshold, grow, feather)
    counts = {"made": 0, "kept": 0, "skipped": 0, "no_person": 0}
    pics = pictures(folder)
    log(f"New masks from the saved RMBG-3.0 mattes: {describe(s)}\n")
    for i, p in enumerate(pics, 1):
        if stop_here(i, len(pics), counts):
            break
        name, mp = os.path.basename(p), matte_path(folder, p)
        if not os.path.isfile(mp):
            counts["skipped"] += 1
            continue
        if mask_for(p) and not overwrite:
            counts["kept"] += 1
            continue
        try:
            with Image.open(p) as im:
                size = upright_size(im)
            with Image.open(mp) as im:
                matte = im.convert("L")
                if matte.size != tuple(size):
                    matte = matte.resize(tuple(size), Image.BILINEAR)
                matte = np.asarray(matte, np.float32) / 255.0
        except Exception as e:
            log(f"  {name}: can't be read ({e}) - skipped\n")
            counts["skipped"] += 1
            continue
        m = finish(matte, **s)
        if float(m.mean()) < MIN_COVER:
            log(f"  {name}: nothing left - no mask written (it trains as normal)\n")
            counts["no_person"] += 1
            continue
        write_mask(p, m)
        counts["made"] += 1
    log(f"{counts['made']} mask(s) remade" + (f", {counts['skipped']} picture(s) with no saved matte"
                                               if counts["skipped"] else "") + "\n")
    return counts


def preview(folder, models_dir, *, variant=VARIANT_DEFAULT, shown=(), n=2, side=320, megapixels=MEGAPIXELS,
            matter=None, log=lambda s: None, **edges):
    """n random pictures not previewed yet with their RMBG-3.0 masks over them, nothing written: ([(name, PIL picture,
    share learned, "character", note)], the next round's names or None). Loads all three models: a minute or two."""
    from fizgig.dataset import sammask
    s = settings(**edges)
    if not _use.acquire(blocking=False):
        raise ValueError("Masks are being made now. Wait for them to finish.")
    try:
        pics, fresh = sammask.preview_pictures(folder, shown, n)
        if matter is None:
            matter, _what = matter_for(models_dir, variant, log=log, megapixels=megapixels)
        out = []
        for p, matte, note in matter.mattes(pics):
            with Image.open(p) as im:
                img = flatten(upright(im))
            m = finish(matte, **s) if matte is not None else np.zeros(img.size[::-1], np.float32)
            out.append((*sammask.overlay(os.path.basename(p), img, m, side), "character", note))
        return out, fresh
    finally:
        _use.release()
