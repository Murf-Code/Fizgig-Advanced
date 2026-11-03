"""Image Prep: resize / convert to PNG / face close-ups, and the Look Consistency Filter's maths - shared by the
desktop app's Image Prep tab and the web UI's, so the two can't drift apart.

Everything happens IN the training folder. Prep never upscales and sizes by AREA (megapixels), floored to the
training bucket grid, so a prepped image is exactly what trains (see resize_image). Resized originals are moved to an
'originals' subfolder unless the user chose to replace them; crops are added beside their pictures, which stay.

The Look Filter scores every face against 3 baselines (the mean similarity - one photo would bake its own angle and
light into every score) with the face embedder Image Prep's face settings pick (ArcFace for people, DINOv2 for furry /
anime faces: face_utils), and saves <folder>/fizgig_look_scores.json, which training's "Warm up look outliers" reads:
keep its format.
"""
import glob
import json
import math
import os
import shutil

from PIL import Image
from fizgig.utils.upright import open_upright, upright_size

from fizgig.dataset.masks import is_mask_file

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
LOOK_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
MODES = ("Auto Prep (Face Crops)", "Resize Only", "Face Crop Only")
MEGAPIXELS = ["0.25", "0.37", "0.5", "0.75", "1.0", "1.5", "2.0", "2.4", "3.0", "4.2"]
FACE_TARGETS = ["Largest Face", "Largest Male Face", "Largest Female Face"]
OTHER_TARGET = "Other…"           # the Face dropdown's Other…: crop to what the Find this box names (SAM 3.1)
FIND_SCORE = 0.5                  # SAM 3.1's cutoff for a hit: the same as Make masks uses
FIND_MIN_SIDE = 16                # a hit smaller than this (pixels, either side) is too small to crop
FACE_CROP_PREFIX = "FaceCrop_"     # face close-ups: FaceCrop_001.png
HIT_MAX_SHARE = 0.60              # a single SAM 3.1 find covering more of the picture than this is ignored
FIT_ALL_MAX_SHARE = 0.85          # Fit all words in one crop: a crop with SAM 3.1 boxes over this share is skipped
DETECTOR_KEEP_SHARE = 0.95        # ... one from the detectors alone is kept while it still trims this much off


class _NothingToAdd(Exception):
    """A crop that would add nothing (Fit all words in one crop covering nearly the whole picture)."""


RESIZED_DIR, CROPS_DIR, SKIPPED_DIR = "resized", "crops", "skipped"   # Image Prep's outputs
CROP_PREFIX = "Crop_"             # Find this close-ups: Crop_paw_001.png (faces keep FaceCrop_)
CLASS_TARGET = "YOLO class: "     # an older build's Crop to choice for a YOLO class ("YOLO class: penis"): now Other…


def bucket_step():
    """The resolution grid training buckets on (RESOLUTION_STEPS), so prep and bucketing can never drift apart."""
    try:
        from fizgig.dataset.image_dataset import RESOLUTION_STEPS
        return int(RESOLUTION_STEPS)
    except Exception:
        return 16


def target_area(mp):
    """Training's real target AREA for a megapixel setting, derived as the dataset TOML writer derives `resolution`:
    floor the square side to the bucket grid (1.0 MP -> 992x992). Prep then lands at or just under what training asks
    for, which keeps training's no-upscale path: the cache step resamples and crops nothing."""
    step = bucket_step()
    side = max(step, int(math.sqrt(max(0.0, mp) * 1_000_000)) // step * step)
    return side * side


def output_size(size, area):
    """The (w, h) resize_image would produce for `size` - same maths, no pixels touched."""
    width, height = size
    if width * height <= area:
        return width, height
    step = bucket_step()
    scale = math.sqrt(area / (width * height))
    return (max(step, int(width * scale) // step * step), max(step, int(height * scale) // step * step))


def resize_image(img, area):
    """Resize to ~`area` pixels, aspect kept, never upscaling; (img, resized). Area, not longest edge: training picks
    its resolution by area, and a longest-edge cap left non-square images below the target for good (issue #44).
    Both sides floored to the bucket step, so the saved file is exactly what trains."""
    width, height = img.size
    if width * height <= area:
        return img, False                      # never upscale - it would only invent detail
    new = output_size((width, height), area)
    if new == (width, height):
        return img, False
    return img.resize(new, Image.LANCZOS), True


def face_mode_of(label):
    """The face target dropdown's label as the detector's mode."""
    if "Male" in (label or ""):
        return "largest_male"
    if "Female" in (label or ""):
        return "largest_female"
    return "largest_face"


def is_other(label):
    """Whether the Face dropdown is on Other… (crop to typed words, not faces)."""
    return str(label or "").strip().lower().startswith("other")


def class_of(label):
    """The YOLO class an older build's Crop to choice named ("YOLO class: penis" -> "penis"), else ""."""
    t = str(label or "")
    return t[len(CLASS_TARGET):].strip() if t.startswith(CLASS_TARGET) else ""


def migrate_target(face_label, find=""):
    """(Crop to, Find this) with an older build's "YOLO class: <name>" turned into Other… finding <name> (the YOLO
    model still finds it, through Other…). Anything else comes back as it was."""
    cls = class_of(face_label)
    return (OTHER_TARGET, cls) if cls else (face_label, find)


def split_or(phrase):
    """'ass OR butt' -> ['ass', 'butt']: the choices for one word, first found wins (OR or | between them)."""
    import re
    return [p.strip() for p in re.split(r"\s+or\s+|\|", str(phrase or ""), flags=re.I) if p.strip()]


def word_slug(word):
    """'the left paw' -> 'the_left_paw': a phrase as a file-name part (letters, digits, _ and -; at most 24)."""
    import re
    s = re.sub(r"[^0-9A-Za-z_-]+", "_", str(word).strip()).strip("_")[:24].strip("_")
    return s or "thing"


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class ThingFinder:
    """Finds what the Find this box names with SAM 3.1 (or anything with SAM3Segmenter's masks()), as boxes the face
    crop code takes: FaceInfo with the phrase in `gender` and source "sam3". Each comma-separated phrase is searched
    on its own; hits under FIND_SCORE or under FIND_MIN_SIDE pixels a side are dropped; boxes are clamped to the
    picture.

    yolo / yolo_words: a phrase that names one of the YOLO face model's classes (Fdetailer's penis, pawpads, ...) is
    found by that model instead (face_utils.YoloOnnxBackend.find_classes, its own score cutoff), source "yolo"; seg
    may then be None when no phrase is left for SAM 3.1."""

    def __init__(self, seg, words, threshold=FIND_SCORE, min_side=FIND_MIN_SIDE, yolo=None, yolo_words=None,
                 fit_all=False):
        self.fit_all = fit_all            # Fit all words in one crop: one close-up around every word found
        self.separate = False             # Separate crop for each word (as well, when Fit all is on too)
        self.groups = [g for g in (split_or(w) for w in words if w) if g]   # 'ass OR pussy': one group, in order
        self.seg, self.words = seg, list(dict.fromkeys(a for g in self.groups for a in g))
        self.threshold, self.min_side = threshold, min_side
        self.yolo = yolo
        self.yolo_words = {w: int(c) for w, c in (yolo_words or {}).items() if w in self.words} if yolo else {}
        self.sam_words = [w for w in self.words if w not in self.yolo_words]

    def _yolo_label(self):
        return getattr(self.yolo, "label", "") or "YOLO model"

    def describe(self):
        parts = []
        if self.yolo_words:
            parts.append(f"{self._yolo_label()} (YOLO), finding {', '.join(self.yolo_words)}")
        if self.sam_words:
            parts.append(f"SAM 3.1, finding {', '.join(self.sam_words)}")
        return "; ".join(parts)

    def _clamped(self, box, w, h):
        x1, y1, x2, y2 = box
        x1, y1 = max(0, min(w, int(round(x1)))), max(0, min(h, int(round(y1))))
        x2, y2 = max(0, min(w, int(round(x2)))), max(0, min(h, int(round(y2))))
        if x2 - x1 < self.min_side or y2 - y1 < self.min_side:
            return None
        return x1, y1, x2, y2

    def find(self, img):
        """[FaceInfo, ...] for every hit, strongest first per phrase, phrases in the order typed."""
        import numpy as np
        from fizgig.face_utils import _face_info
        arr = np.asarray(img.convert("RGB"))
        h, w = arr.shape[:2]
        out = []
        if self.sam_words and self.seg is not None:
            for f in self.seg.masks(arr, self.sam_words, threshold=self.threshold, refine=0):
                if f.get("score", 1.0) < self.threshold:
                    continue
                b = self._clamped(f["box"], w, h)
                if b:
                    out.append(_face_info(*b, gender=f["prompt"], score=f.get("score", 1.0), source="sam3"))
        if self.yolo_words:
            by_id = {}
            for word, cid in self.yolo_words.items():
                by_id.setdefault(cid, []).append(word)
            for x1, y1, x2, y2, score, cid in self.yolo.find_classes(arr, sorted(by_id)):
                b = self._clamped((x1, y1, x2, y2), w, h)
                if b:
                    for word in by_id.get(int(cid), []):
                        out.append(_face_info(*b, gender=word, score=float(score), source="yolo"))
        if self.yolo_words:                            # both finders' hits: phrases in the order typed
            order = {wd: i for i, wd in enumerate(self.words)}
            out.sort(key=lambda f: (order.get(f.gender, len(order)), -f.score))
        return out

    def pick_or(self, hits):
        """Each group's hits from its first choice that was found ('ass OR pussy': pussy only when no ass)."""
        found = {f.gender for f in hits}
        keep = set()
        for g in getattr(self, "groups", None) or [[w] for w in self.words]:
            for alt in g:
                if alt in found:
                    keep.add(alt)
                    break
        return [f for f in hits if f.gender in keep]

    def group_count(self):
        return len(getattr(self, "groups", None) or self.words)

    @staticmethod
    def largest(hits):
        return max(hits, key=lambda f: f.area) if hits else None

    @staticmethod
    def per_word(hits, overlap=0.8):
        """The largest hit of each phrase, in the order typed; a hit that is mostly the same box as one already
        picked (another phrase finding the same thing) is left out."""
        picked = []
        for word in dict.fromkeys(f.gender for f in hits):
            best = max((f for f in hits if f.gender == word), key=lambda f: f.area)
            if all(_iou(best.bbox, p.bbox) < overlap for p in picked):
                picked.append(best)
        return picked


def image_files(folder):
    """The folder's images (top level), sorted."""
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
            and not is_mask_file(f)]


def load_image(path):
    img = open_upright(path)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        return img.convert('RGBA')
    return img.convert('RGB')


def _rotated(path):
    from fizgig.dataset.masks import exif_rotated
    return exif_rotated(path)


def atomic_png_save(img, output_path):
    """Write to a temp file then os.replace: in-place mode saves over the original, and a crash mid-write used to
    truncate the source photo beyond recovery."""
    tmp = output_path + ".fizgig-tmp"
    try:
        img.save(tmp, "PNG")
        os.replace(tmp, output_path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def source_stats(folder, max_sample=40):
    """(image count, median area, median size) of the folder's top-level images, from headers only (cheap)."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    try:
        files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts
                 and not is_mask_file(f)]
    except OSError:
        return 0, None, None
    sizes = []
    for p in files[:max_sample]:
        try:
            with Image.open(p) as im:
                sizes.append(upright_size(im))
        except Exception:
            pass
    if not sizes:
        return len(files), None, None
    sizes.sort(key=lambda wh: wh[0] * wh[1])
    med = sizes[len(sizes) // 2]
    return len(files), med[0] * med[1], med


def summary(folder, mode, replace, prep_mp, train_mp=None, find=""):
    """The 'What will happen' card: {lines, warn (prep below training's megapixels), count}. find: the Find this
    box's words when the Face dropdown is on Other… (else "")."""
    area = target_area(prep_mp)
    n, median_area, median_size = source_stats(folder)
    example = ""
    if median_size:
        w, h = output_size(median_size, area)
        if (w, h) != tuple(median_size):
            example = f" (your typical {median_size[0]}×{median_size[1]} → {w}×{h})"
    count = f"your {n} images" if n else "your images"
    sized = f"resized to about {prep_mp:g} MP{example} and saved as PNG"
    if find and mode == "Auto Prep (Face Crops)":
        what = f"{count} → {sized}, plus a close-up of each {find} found"
    elif find and mode == "Face Crop Only":
        what = (f"{count} → a close-up of the {find} in each picture (the biggest one found), added as PNG "
                f"named {CROP_PREFIX}{word_slug(find.replace('\n', ',').split(',')[0].strip() or find)}_001.png and on, as the close-ups "
                f"of Resize + close-up crops are")
    elif mode == "Auto Prep (Face Crops)":
        what = f"{count} → {sized}, plus one face close-up each{f' (≈{n * 2} files)' if n else ''}"
    elif mode == "Face Crop Only":
        what = (f"{count} → a face close-up from each photo, added as PNG named {FACE_CROP_PREFIX}001.png and "
                f"on, as the close-ups of Resize + close-up crops are")
    else:
        what = f"{count} → {sized}"
    if mode == "Face Crop Only":
        originals = ("Your pictures and captions stay exactly as they are. The close-ups, with their captions, go "
                     "in the 'crops' sub-folder; pictures with nothing to crop are copied to 'skipped' (with the "
                     "reason in skipped.txt) and aren't trained twice.")
    elif mode == "Auto Prep (Face Crops)":
        originals = ("Your pictures and captions stay exactly as they are. Resized copies, with their captions, go "
                     "in the 'resized' sub-folder and close-ups in 'crops'; pictures the cropping skipped are "
                     "copied to 'skipped' with the reason. Training uses the resized copies in place of the "
                     "full-size ones. To undo, delete those folders.")
    else:
        originals = ("Your pictures and captions stay exactly as they are. Resized copies, with their captions, go "
                     "in the 'resized' sub-folder, and training uses them in place of the full-size ones. To undo, "
                     "delete that folder.")
    lines = [f"{what}. {originals}"]
    if mode != "Resize Only" and median_area is not None and median_area <= area:
        what_cut = "close-ups" if find else "face close-ups"
        lines.append(f"⚠ Your images are already at or below {prep_mp:g} MP — {what_cut} cut from them will be "
                     f"soft. If you have higher-res versions, prep from those instead.")
    if mode != "Resize Only" and find:
        lines.append("Pictures where nothing is found are left whole. Next: eyeball the close-ups on the Captions "
                     "tab and Remove any you don't want before captioning.")
    elif mode != "Resize Only":
        lines.append("Next: eyeball the face close-ups on the Captions tab and Remove any blurry ones before "
                     "captioning.")
    warn = ""
    if train_mp is not None and prep_mp < train_mp:
        warn = (f"⚠ Training is set to {train_mp:g} MP but prep is set to {prep_mp:g} MP — your images would be "
                f"shrunk below what training asks for, and training cannot get that detail back. Match them, or "
                f"prep higher.")
    return {"lines": lines, "warn": warn, "count": n}


class Prep:
    """One prep run over a folder. log(text) receives the Output Log's lines; detector is a face_utils.FaceDetector
    (needed by the face modes). finder: a ThingFinder instead, when the Face dropdown is on Other…. captions: a
    crop_captions.CropCaptions - crops get their picture's caption (None: captions aren't touched, as before)."""

    def __init__(self, log, detector=None, finder=None, captions=None):
        self.log = log
        self.detector = detector
        self.finder = finder
        self.captions = captions
        self._originals = {}                     # folder -> its originals folder, picked once per run (run() resets)

    def _caption_crop(self, filepath, crop_path, crop_img):
        if self.captions is not None:
            try:
                self.captions.to_crop(filepath, crop_path, self.log, crop_img)
            except Exception as e:
                self.log(f"  caption not copied: {e}\n")

    def _caption_in_place(self, filepath, output_path, folder, replace, cropped):
        """cropped: the picture became a crop (its caption may be trimmed against the saved crop)."""
        if self.captions is None:
            return
        try:
            crop_img = None
            if cropped and getattr(self.captions, "tagger", None) is not None:
                crop_img = load_image(output_path)
            self.captions.in_place(filepath, output_path, None if replace else self._originals_for(folder),
                                   self.log, crop_img, cropped=bool(cropped))
        except Exception as e:
            self.log(f"  caption not updated: {e}\n")

    # ---- originals ----
    def _originals_dir(self, folder):
        """originals, else originals_2, _3... - the first that holds no images."""
        n = 1
        while True:
            cand = os.path.join(folder, "originals" if n == 1 else f"originals_{n}")
            if not os.path.isdir(cand) or not any(
                    os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
                    for f in os.listdir(cand) if os.path.isfile(os.path.join(cand, f))):
                return cand
            n += 1

    def _originals_for(self, folder):
        if folder not in self._originals:
            self._originals[folder] = self._originals_dir(folder)
        return self._originals[folder]

    def safe_output_path(self, filepath, output_path):
        """Never overwrite a DIFFERENT file: photo.jpg's output must not destroy an unrelated photo.png."""
        if not os.path.exists(output_path):
            return output_path
        try:
            if os.path.samefile(filepath, output_path):
                return output_path            # in-place re-save of itself
        except OSError:
            pass
        stem, ext = os.path.splitext(output_path)
        n = 2
        cand = f"{stem}_{n}{ext}"
        while os.path.exists(cand):
            n += 1
            cand = f"{stem}_{n}{ext}"
        self.log(f"Name collision: {os.path.basename(output_path)} already exists — "
                 f"writing {os.path.basename(cand)} instead\n")
        return cand

    def stash_if_inplace(self, filepath, output_path, folder, replace):
        """Before saving: keep-safe + in-place PNG means the save overwrites the original (issue #43), so copy it to
        originals/ first (a copy: PIL may still hold the source open, and moving an open file fails on Windows)."""
        if replace or filepath != output_path or not os.path.exists(filepath):
            return
        d = self._originals_for(folder)
        os.makedirs(d, exist_ok=True)
        shutil.copy2(filepath, os.path.join(d, os.path.basename(filepath)))

    def handle_original(self, filepath, output_path, folder, replace):
        """After saving: delete the original (replace) or move it to originals/ (keep safe)."""
        if filepath == output_path:
            return
        if replace:
            os.remove(filepath)
        else:
            d = self._originals_for(folder)
            os.makedirs(d, exist_ok=True)
            shutil.move(filepath, os.path.join(d, os.path.basename(filepath)))

    def keep_mask(self, filepath, output_path, folder, replace, crop=None):
        """Masked training: a picture's mask (photo-masklabel.png) follows it. A cropped picture's mask gets the same
        crop (crop: the mask -> its cropped version) and the output's name, its original going to originals/ with the
        picture (unless replacing); otherwise it only takes the output's name if that changed. Called after the
        picture is saved and its original handled."""
        from fizgig.dataset.masks import MASK_SUFFIX, mask_for, read_mask
        m = mask_for(filepath)
        if not m:
            return
        out_mask = os.path.join(os.path.dirname(output_path),
                                os.path.splitext(os.path.basename(output_path))[0] + MASK_SUFFIX + ".png")
        if crop is None:
            if os.path.normcase(m) != os.path.normcase(out_mask) and not os.path.exists(out_mask):
                os.replace(m, out_mask)                  # photo.jpg -> photo_2.png: its mask takes the new name
            return
        cropped = crop(read_mask(m))
        if not replace:
            d = self._originals_for(folder)
            os.makedirs(d, exist_ok=True)
            shutil.copy2(m, os.path.join(d, os.path.basename(m)))
        if os.path.normcase(m) != os.path.normcase(out_mask):
            os.remove(m)
        atomic_png_save(cropped, out_mask)
        self.log(f"  its mask: cropped the same way -> {os.path.basename(out_mask)}\n")

    def crop_mask_for(self, filepath, size, face, padding, crop_path):
        """Masked training: a face close-up made from a picture with a mask gets that mask, cropped the same way."""
        from fizgig.dataset.masks import MASK_SUFFIX, mask_for, read_mask
        from fizgig.face_utils import crop_to_face
        m = mask_for(filepath)
        if not m:
            return
        mk = read_mask(m)
        mk = crop_to_face(mk.resize(size) if mk.size != size else mk, face, padding)
        atomic_png_save(mk, os.path.splitext(crop_path)[0] + MASK_SUFFIX + ".png")

    # ---- faces ----
    def select_face(self, faces, face_mode):
        """(FaceInfo or None, note) by the face target."""
        if not faces:
            return None, ""
        for want in ("male", "female"):
            if face_mode == f"largest_{want}":
                sel = self.detector.get_largest_by_gender(faces, want, fallback_to_any=True)
                if not sel or sel.gender == want:
                    return sel, ""
                if sel.gender == "unknown":      # a YOLO face (furry / anime): no gender to go by
                    return sel, "  Note: the face finder can't tell gender on this face, using largest face\n"
                return sel, f"  Note: No {want} face, using largest face\n"
        return self.detector.get_largest(faces), ""

    @staticmethod
    def is_crop(filename):
        """A crop Fizgig made (FaceCrop_ / Crop_): never cropped again on a re-run."""
        return filename.startswith(FACE_CROP_PREFIX) or filename.startswith(CROP_PREFIX)

    @staticmethod
    def _next_index(folder, stem, first_part=False):
        """The next free number after stem (FaceCrop_ / Crop_paw_): every file of that name counts - a picture, its
        caption or mask, or a caption left behind by a crop that was deleted (Crop_sheath_045.txt), so a new crop
        never takes an old one's caption. first_part: FaceCrop_001_x counts as 1, as it always has."""
        from fizgig.dataset.masks import MASK_SUFFIX
        best = 0
        try:
            names = os.listdir(folder)
        except OSError:
            return 1
        for f in names:
            name = os.path.splitext(f)[0]
            if name.endswith(MASK_SUFFIX):
                name = name[:-len(MASK_SUFFIX)]
            if not name.startswith(stem):
                continue
            tail = name[len(stem):]
            if first_part:
                tail = tail.split("_")[0]
            if tail.isdigit():
                best = max(best, int(tail))
        return best + 1

    @classmethod
    def next_facecrop_index(cls, folder):
        """The next number for FaceCrop_NNN.png."""
        return cls._next_index(folder, FACE_CROP_PREFIX, first_part=True)

    @classmethod
    def next_crop_index(cls, folder, word):
        """The next number for Crop_<word>_NNN.png."""
        return cls._next_index(folder, f"{CROP_PREFIX}{word_slug(word)}_")

    def _place_crop(self, filepath, img, crop_path, folder, replace, crop_mask=None):
        """Close-up crops only: the crop is saved under its own name (crop_path: FaceCrop_NNN / Crop_<word>_NNN, as
        the Resize + close-up crops mode names them) and gets the picture's caption the way those crops do (copied,
        trimmed, close-up, the target kept). The picture, its caption and its mask stay where they are, untouched -
        the crop is added beside them (replace doesn't apply: nothing is overwritten). A mask gets the same crop."""
        from fizgig.dataset.crop_captions import CropCaptions, find_caption
        atomic_png_save(img, crop_path)
        caps = self.captions if self.captions is not None else CropCaptions()     # None: the caption copied as is
        if find_caption(filepath, caps.ext):
            try:
                caps.to_crop(filepath, crop_path, self.log, img)
            except Exception as e:
                self.log(f"  caption not copied: {e}\n")
        if crop_mask is not None:
            from fizgig.dataset.masks import MASK_SUFFIX, mask_for, read_mask
            m = mask_for(filepath)
            if m:
                out_mask = os.path.splitext(crop_path)[0] + MASK_SUFFIX + ".png"
                atomic_png_save(crop_mask(read_mask(m)), out_mask)
                self.log(f"  its mask: cropped the same way -> {os.path.basename(out_mask)}\n")

    # ---- the three modes: the training folder itself is never changed ----
    def run(self, mode, folder, prep_mp, face_label="Largest Face", face_padding=20.0, replace=False):
        """Prep never touches the training folder's own files: resized copies go to resized/, close-ups to crops/,
        and pictures the cropping skipped (nothing found / an error) are copied to skipped/ with skipped.txt saying
        why. A re-run empties the folders it writes first; undo = delete the folder. replace is no longer used."""
        self._originals = {}
        area = target_area(prep_mp)
        face_mode = face_mode_of(face_label)
        if self.captions is not None:                 # what the crops are of stays in their captions (trim never drops it)
            self.captions.targets = list(self.finder.words) if self.finder is not None else ["face"]
        if self.detector is not None and mode != "Resize Only" and getattr(self.detector, "backend_name",
                                                                             "insightface") != "insightface":
            self.log(f"Face finder: {self.detector.describe()}\n")      # only when it isn't the usual one
        if self.finder is not None and mode != "Resize Only":
            if getattr(self.finder, "yolo_words", None):      # some (or all) of it found by the YOLO model
                self.log(f"Finding with {self.finder.describe()}\n")
            else:
                self.log(f"Finding: {', '.join(self.finder.words)} (SAM 3.1)\n")
        resize_on, crop_on = mode != "Face Crop Only", mode != "Resize Only"
        self.log(f"Mode: {mode}\n")
        dirs = {}
        if resize_on:
            dirs["resized"] = os.path.join(folder, RESIZED_DIR)
        if crop_on:
            dirs["crops"] = os.path.join(folder, CROPS_DIR)
            dirs["skipped"] = os.path.join(folder, SKIPPED_DIR)
        for d in dirs.values():                       # a re-run replaces what the last one made
            if os.path.isdir(d):
                shutil.rmtree(d)
        self._padding = face_padding
        self._sel_pad = {}                            # a crop cut with less padding than asked: {(box, word): padding}
        self._go(folder, dirs, area, face_mode, face_padding, single=(mode == "Face Crop Only"))

    def _picks(self, img, face_mode, single):
        """[(sel, crop name stem)] of what to cut from img: the face, or each word found (only the biggest in
        Close-up crops only)."""
        if self.finder is not None:
            found = self._sane(self.finder.find(img), img.size)
            if hasattr(self.finder, "pick_or"):
                found = self.finder.pick_or(found)
            groups = self.finder.group_count() if hasattr(self.finder, "group_count") else len(self.finder.words)
            fit = getattr(self.finder, "fit_all", False) and groups > 1
            sep = getattr(self.finder, "separate", False)
            if fit:
                one = self._fit_all(img, found, raise_skip=not sep)
                if not sep:
                    return one
                each = list(self.finder.per_word(found) or [])
                return (one if one and one[0].source == "union" else []) + each
            if single and not sep:
                sel = self.finder.largest(found)
                return [sel] if sel else []
            return list(self.finder.per_word(found) or [])
        faces = self.detector.detect_from_pil(img)
        sel, note = self.select_face(faces, face_mode)
        if note:
            self.log(note)
        return [sel] if sel else []

    def _sane(self, hits, size):
        """Hits worth cropping: a word a detector found keeps only the detector's boxes (SAM's are dropped), and a SAM
        3.1 box over HIT_MAX_SHARE of the picture is ignored (logged) - SAM's big boxes are usually wrong. A detector's
        box is trusted, however big. Raises _NothingToAdd when every hit was too big."""
        by_yolo = {f.gender for f in hits if getattr(f, "source", "") == "yolo"}
        hits = [f for f in hits if getattr(f, "source", "") == "yolo" or f.gender not in by_yolo]
        area = float(size[0] * size[1]) or 1.0
        kept, big = [], []
        for f in hits:
            x1, y1, x2, y2 = f.bbox
            share = (x2 - x1) * (y2 - y1) / area
            if share > HIT_MAX_SHARE and getattr(f, "source", "") != "yolo":
                big.append(f"{f.gender} ({100 * share:.0f}%)")
                self.log(f"  {f.gender}: SAM 3.1's box is too big ({100 * share:.0f}%), ignored\n")
            else:
                kept.append(f)
        if big and not kept:
            raise _NothingToAdd("only boxes covering most of the picture were found: " + ", ".join(big))
        return kept

    def _fit_all(self, img, found, raise_skip=True):
        """Fit all words in one crop: the biggest hit of each word, one box around them all. One word found: just
        that one (the missing ones logged).

        Too big: with padding, a crop over FIT_ALL_MAX_SHARE of the picture. When every box came from a detector
        (trusted), the padding is left off instead, and the crop is kept while it still trims some background (under
        DETECTOR_KEEP_SHARE); only a crop that is the whole picture anyway is skipped. With a SAM 3.1 box in it, an
        over-big crop is skipped as before."""
        from fizgig.face_utils import _face_info, crop_to_face
        picked = list(self.finder.per_word(found) or [])
        if len(picked) <= 1:
            if picked:
                groups = getattr(self.finder, "groups", None) or [[w] for w in self.finder.words]
                missing = [" or ".join(g) for g in groups
                           if word_slug(picked[0].gender).lower() not in {word_slug(a).lower() for a in g}]
                self.log(f"  Only {picked[0].gender} found (not {', '.join(missing)}): cropping just that\n")
            return picked
        boxes = [p.bbox for p in picked]
        x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        sel = _face_info(x1, y1, x2, y2, gender="_".join(word_slug(p.gender) for p in picked),
                         score=min(getattr(p, "score", 1.0) for p in picked), source="union")
        whole = float(img.size[0] * img.size[1]) or 1.0
        w, h = crop_to_face(img, sel, self._padding).size
        if w * h <= FIT_ALL_MAX_SHARE * whole:
            return [sel]
        share = w * h / whole
        if all(getattr(p, "source", "") == "yolo" for p in picked):
            tw, th = crop_to_face(img, sel, 0).size
            if tw * th < DETECTOR_KEEP_SHARE * whole:
                self.__dict__.setdefault("_sel_pad", {})[(sel.bbox, sel.gender)] = 0
                self.log(f"  Big crop ({100 * share:.0f}% with padding): padding left off, so it still trims some "
                         f"background ({100 * tw * th / whole:.0f}%)\n")
                return [sel]
            share = tw * th / whole
        if not raise_skip:
            return []
        raise _NothingToAdd(f"the crop around {', '.join(p.gender for p in picked)} would cover nearly the "
                            f"whole picture ({100 * share:.0f}%), so it adds nothing")

    def _copy_with_sidecars(self, filepath, dest_dir, new_stem=None):
        """Copy a picture's caption and mask (not the picture) into dest_dir, under new_stem when given."""
        from fizgig.dataset.crop_captions import find_caption
        from fizgig.dataset.masks import MASK_SUFFIX, mask_for
        stem = new_stem or os.path.splitext(os.path.basename(filepath))[0]
        cap = find_caption(filepath, getattr(self.captions, "ext", None) or ".txt")
        if cap:
            shutil.copy2(cap, os.path.join(dest_dir, stem + os.path.splitext(cap)[1]))
        m = mask_for(filepath)
        return m, os.path.join(dest_dir, stem + MASK_SUFFIX + ".png")

    def _skip(self, filepath, d, reason, skipped):
        """Copy a picture the cropping skipped (bytes as they are) with its caption and mask into skipped/."""
        os.makedirs(d, exist_ok=True)
        shutil.copy2(filepath, os.path.join(d, os.path.basename(filepath)))
        m, out_mask = self._copy_with_sidecars(filepath, d)
        if m:
            shutil.copy2(m, out_mask)
        skipped.append(f"{os.path.basename(filepath)}: {reason}")
        with open(os.path.join(d, "skipped.txt"), "w", encoding="utf-8") as f:
            f.write("Pictures the cropping skipped (copies; the originals are untouched in the training folder):\n"
                    + "\n".join(skipped) + "\n")

    def _go(self, source, dirs, area, face_mode, padding, single):
        from fizgig.dataset.crop_captions import CropCaptions, find_caption
        from fizgig.dataset.masks import read_mask
        from fizgig.face_utils import crop_to_face
        rdir, cdir, sdir = dirs.get("resized"), dirs.get("crops"), dirs.get("skipped")
        resized_n = crops_n = errors = 0
        skipped = []
        index = {}
        for filepath in image_files(source):
            filename = os.path.basename(filepath)
            base, ext = os.path.splitext(filename)
            if self.is_crop(filename):
                self.log(f"Skipped (derivative): {filename}\n")
                continue
            try:
                original_img = load_image(filepath)
                original = original_img.size
                if cdir:
                    try:
                        picks = self._picks(original_img, face_mode, single)
                        if not picks:
                            self.log(f"Nothing found in {filename}, not cropped\n")
                            self._skip(filepath, sdir, "nothing found to crop", skipped)
                        for sel in picks:
                            pad = getattr(self, "_sel_pad", {}).get((sel.bbox, sel.gender), padding)
                            word = FACE_CROP_PREFIX if self.finder is None else f"{CROP_PREFIX}{word_slug(sel.gender)}_"
                            os.makedirs(cdir, exist_ok=True)
                            if word not in index:
                                index[word] = (self.next_facecrop_index(cdir) if self.finder is None else
                                               self.next_crop_index(cdir, sel.gender))
                            crop_path = os.path.join(cdir, f"{word}{index[word]:03d}.png")
                            index[word] += 1
                            crop, _ = resize_image(crop_to_face(original_img, sel, pad), area)
                            atomic_png_save(crop, crop_path)
                            self.crop_mask_for(filepath, original, sel, pad, crop_path)
                            # Close-up crops only always copies the caption (as before); Auto Prep only with settings
                            caps = self.captions if self.captions is not None else (CropCaptions() if single else None)
                            if caps is not None and find_caption(filepath, caps.ext):
                                try:
                                    caps.to_crop(filepath, crop_path, self.log, crop)
                                except Exception as e:
                                    self.log(f"  caption not copied: {e}\n")
                            self.log(f"Crop: {os.path.basename(crop_path)} ({crop.size[0]}x{crop.size[1]}) from "
                                     f"{filename} " + (f"[face: {sel.gender}]" if single and self.finder is None else f"[{sel.gender}]")
                                     + "\n")
                            crops_n += 1
                            crop.close()
                    except _NothingToAdd as e:
                        self.log(f"Not cropped: {filename} - {e}\n")
                        self._skip(filepath, sdir, str(e), skipped)
                    except Exception as e:
                        self.log(f"Find error ({filename}): {e}\n")
                        self._skip(filepath, sdir, f"error while finding: {e}", skipped)
                if rdir:
                    os.makedirs(rdir, exist_ok=True)
                    img, resized = resize_image(original_img, area)
                    out = os.path.join(rdir, base + ".png")
                    if ext.lower() == ".png" and not resized and not _rotated(filepath):
                        shutil.copy2(filepath, out)          # already a PNG at the right size: copied as is
                        self.log(f"Copied as is (already the right size): {filename}\n")
                    else:
                        atomic_png_save(img, out)
                        self.log(f"Resized: {filename} [{original[0]}x{original[1]} -> {img.size[0]}x{img.size[1]}]\n")
                    m, out_mask = self._copy_with_sidecars(filepath, rdir)
                    if m:
                        mk = read_mask(m)
                        atomic_png_save(mk.resize(img.size) if mk.size != img.size else mk, out_mask)
                    resized_n += 1
                    img.close()
                original_img.close()
            except Exception as e:
                self.log(f"Error ({filename}): {e}\n")
                errors += 1
                if sdir:
                    self._skip(filepath, sdir, f"error: {e}", skipped)
        self.log("\n--- Summary ---\n")
        self.log("Your training folder's own pictures and captions were not changed.\n")
        if rdir:
            self.log(f"Resized copies: {resized_n}, with their captions, in {rdir}\n")
        if cdir:
            self.log(f"Close-ups made: {crops_n}, with their captions, in {cdir}\n")
            self.log(f"Skipped by the cropping: {len(skipped)}" +
                     (f" - copies and the reasons (skipped.txt) in {sdir}\n" if skipped else "\n"))
        if errors:
            self.log(f"Errors: {errors}\n")
        self.log("Training uses the resized copies and crops in place of the full-size pictures. "
                 "To undo, delete those folders.\n")


# ---- Look Consistency Filter ---------------------------------------------------------------------------------
ARCFACE = {"match": 0.45, "borderline": 0.30, "floor": 0.25}     # face_utils.IDENTITY_THRESHOLDS["arcface"]


def verdict(sim, thresholds=None):
    """(label, colour, blurb) for a similarity under the embedder's thresholds (face_utils.IDENTITY_THRESHOLDS; the
    default is ArcFace's - the same person across varied photos usually lands 0.30-0.70, a different person rarely
    clears 0.25). Scores from different embedders are on different scales."""
    t = thresholds or ARCFACE
    if sim is None:
        return ("no face", "#7F8C8D", "No face detected — can't be scored. Back shots are fine; judge by eye.")
    if sim >= t["match"]:
        return ("match", "#70AD47", "Solid match to the baseline look.")
    if sim >= t["borderline"]:
        return ("borderline", "#E67E22", "Same subject territory, but drifting — worth an eyeball.")
    return ("drift", "#E74C3C", "Weak match — likely off-look (synthetic drift or a different subject).")


def look_files(folder):
    """The folder's images to score, normpath'd (baselines are compared by path) - not the masks beside them
    (photo-masklabel.png, #247)."""
    return sorted(os.path.normpath(os.path.join(folder, f)) for f in os.listdir(folder)
                  if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1].lower() in LOOK_EXTS
                  and not is_mask_file(f))


def cutoff(scores, floor=0.25):
    """The auto-suggest fence: median - 1.5 x IQR of the scored faces, floored at the embedder's floor (ArcFace 0.25);
    None under 4 faces. (median, cutoff)."""
    ss = sorted(s for s in scores.values() if s is not None)
    if len(ss) < 4:
        return None, None
    n = len(ss)
    med, q1, q3 = ss[n // 2], ss[n // 4], ss[(3 * n) // 4]
    return med, max(med - 1.5 * (q3 - q1), floor)


def suggest(scores, baselines, floor=0.25):
    """The drifters auto-suggest marks: below the fence; never a baseline, never a no-face."""
    _, cut = cutoff(scores, floor)
    if cut is None:
        return set()
    return {p for p, s in scores.items() if s is not None and s < cut and p not in baselines}


def score(files, baselines, embed, progress=None):
    """{path: mean similarity to the 3 baselines, or None (no face)}; embed(path) -> normed embedding or None.
    Raises ValueError naming any baseline with no face."""
    import numpy as np
    base = [embed(b) for b in baselines]
    missing = [os.path.basename(b) for b, e in zip(baselines, base) if e is None]
    if missing:
        raise ValueError("No face found in baseline(s): " + ", ".join(missing) + " — pick images with a clear face.")
    out = {}
    for i, p in enumerate(files, 1):
        emb = embed(p)
        # the mean of the 3 similarities == similarity to the baselines' centroid: no one photo's framing dominates
        out[p] = None if emb is None else float(np.mean([np.dot(be, emb) for be in base]))
        if progress:
            progress(i, len(files))
    return out


def save_scores(folder, baselines, scores, embedder=None):
    """<folder>/fizgig_look_scores.json for training's "Warm up look outliers": baselines, the auto-suggest cutoff,
    each image's score by file stem, and the embedder that made them (face_utils.FaceEmbedder: its cache_id and
    label - a file without one is ArcFace). Keep the format."""
    floor = embedder.thresholds["floor"] if embedder is not None else ARCFACE["floor"]
    payload = {"baselines": [os.path.basename(b) for b in baselines], "cutoff": cutoff(scores, floor)[1],
               "scores": {os.path.splitext(os.path.basename(p))[0]: s for p, s in scores.items()}}
    if embedder is not None:
        payload["embedder"], payload["embedder_label"] = embedder.cache_id, embedder.label
    with open(os.path.join(folder, "fizgig_look_scores.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def move_out(folder, paths, caption_exts=(".txt",)):
    """Move the images (and their captions) to <folder>/excluded_by_look/ - never deletes. (moved paths, errors).
    A failed move that left a copy behind removes the copy, so the image isn't in and out of the dataset."""
    dest = os.path.join(folder, "excluded_by_look")
    os.makedirs(dest, exist_ok=True)
    moved, errors = [], []
    for p in paths:
        if not os.path.exists(p):
            continue
        target = None
        try:
            base = os.path.basename(p)
            target = os.path.join(dest, base)
            n = 2
            while os.path.exists(target):
                stem, ext = os.path.splitext(base)
                target = os.path.join(dest, f"{stem}_{n}{ext}")
                n += 1
            shutil.move(p, target)
            for cext in dict.fromkeys(caption_exts):
                cap = os.path.splitext(p)[0] + cext
                if os.path.exists(cap):
                    shutil.move(cap, os.path.splitext(target)[0] + cext)
            from fizgig.dataset.masks import MASK_SUFFIX, mask_for
            mk = mask_for(p)                             # its mask goes with it
            if mk:
                shutil.move(mk, os.path.splitext(target)[0] + MASK_SUFFIX + os.path.splitext(mk)[1])
            moved.append(p)
        except Exception as e:
            try:
                if target and os.path.exists(p) and os.path.exists(target):
                    os.remove(target)
            except Exception:
                pass
            errors.append(f"Could not move {os.path.basename(p)}: {e}")
    return moved, errors
