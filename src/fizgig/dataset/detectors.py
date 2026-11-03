"""Extra body-part and watermark detectors for Image Prep, run with onnxruntime only (no ultralytics in Fizgig).

Other…'s Find this words go down a chain, each word to the first model that has a class for it (or a synonym, e.g.
butt for ass): the YOLO face model (Fdetailer) first, then booru_yolo, then Anime NSFW All-in-One, then assDetailer,
then SAM 3.1 last. A detector's boxes are trusted however big; only SAM 3.1's are dropped when they cover more than
prep.HIT_MAX_SHARE of the picture. A model that finds nothing for a word hands it on to the next one. The log names the
model behind each box. A model whose file isn't there is skipped (said once).

Weights are the user's: booru_yolo has a Download button (Hugging Face, deepghs/booru_yolo; the weights' own licence
is Ultralytics' AGPL-3.0, so Fizgig never ships them). The Civitai models are .pt files the user downloads and turns
into .onnx once with tools/convert_detector_to_onnx.py (docs/FACE_BACKENDS.md).

Watermark masks (Make masks' Mask watermarks, never during cropping): both watermark models' boxes (score >= WM_SCORE,
padded WM_PAD px) are painted black on the picture's mask beside it (photo-masklabel.png), merged with the mask it
has; masked training then leaves them out. A box over WM_MAX_SHARE of the picture is dropped: watermark_s fires on big
blank areas at scores 0.3-0.7 (measured on real sets), while real watermarks and signatures are small."""
import json
import os
import threading

import numpy as np

BODY = [
    {"key": "booru", "label": "booru_yolo", "file": "booru_yolov8s_aa11.onnx",
     "hf": {"repo_id": "deepghs/booru_yolo", "filename": "yolov8s_aa11/model.onnx",
            "revision": "0b38996f5c4dd9c391596c5d5579f44b1b8bc083"},
     "page": "https://huggingface.co/deepghs/booru_yolo",
     "finds": "head, bust, boob, belly, butt, ass, hip, wing, feral and animal heads (anime and furry art)"},
    {"key": "aio", "label": "Anime NSFW All-in-One v5.0", "file": "anime_nsfw_aio_v5.onnx",
     "page": "https://civitai.com/models/1313556?modelVersionId=2266294",
     "finds": "nipples, pussy, anus, penis, testicles and more (anime art)"},
    {"key": "assdetailer", "label": "assDetailer v2", "file": "assdetailer_v2.onnx",
     "page": "https://civitai.com/models/1156687?modelVersionId=1309631", "finds": "butt"},
]
WATERMARK = [
    {"key": "watermark", "label": "Watermark (s)", "file": "watermark_s.onnx",
     "page": "https://civitai.com/models/311866?modelVersionId=350000", "finds": "watermarks, signatures, text"},
    {"key": "nsfw_watermark", "label": "NSFW Watermark (s)", "file": "nsfw_watermark_s.onnx",
     "page": "https://civitai.com/models/311869?modelVersionId=350003", "finds": "watermarks on NSFW art"},
]
MODELS = {m["key"]: m for m in BODY + WATERMARK}
CONF = 0.35
WM_SCORE, WM_PAD, WM_WARN, WM_MAX_SHARE = 0.3, 8, 0.15, 0.10

# a typed word -> class names that count as it (compared by face_utils.class_key: case, spaces, _ and - ignored)
SYNONYMS = {
    "ass": ["ass", "butt", "buttocks"], "butt": ["butt", "ass", "buttocks"], "buttocks": ["butt", "ass"],
    "breasts": ["breasts", "breast", "boob", "boobs"], "breast": ["breast", "breasts", "boob"],
    "boobs": ["boob", "breasts", "breast"], "boob": ["boob", "breasts", "breast"], "chest": ["bust", "chest"],
    "bust": ["bust"], "belly": ["belly", "stomach"], "stomach": ["belly"], "navel": ["navel", "belly"],
    "hips": ["hip", "hips"], "hip": ["hip"], "wings": ["wing", "wings"], "wing": ["wing"],
    "head": ["head"], "nipples": ["nipples", "nipple"], "nipple": ["nipple", "nipples"],
    "pussy": ["pussy", "vagina", "vulva"], "vagina": ["vagina", "pussy"], "anus": ["anus", "anal"],
    "penis": ["penis", "dick"], "testicles": ["testicles", "balls"], "balls": ["testicles", "balls"],
    "paws": ["pawpads", "paw"], "pawpads": ["pawpads"],
}

_said = set()
_said_lock = threading.Lock()


def _once(log, key, text):
    with _said_lock:
        if key in _said:
            return
        _said.add(key)
    log(text)


def models_dir():
    from fizgig.user_data import MODELS_DIR
    return os.path.join(MODELS_DIR, "detectors")


def _prefs():
    from fizgig import face_utils as F
    try:
        with open(F.PREFS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def custom_models(prefs=None):
    """Your own YOLO / ADetailer models added with Add model (prefs.json detector_custom: [{key, label, path, kind}])."""
    pr = prefs if prefs is not None else _prefs()
    out = []
    for m in pr.get("detector_custom") or []:
        if isinstance(m, dict) and m.get("key") and m.get("path"):
            out.append({"key": str(m["key"]), "label": str(m.get("label") or os.path.basename(m["path"])),
                        "path": str(m["path"]), "kind": "watermark" if m.get("kind") == "watermark" else "body"})
    return out


def is_on(key, prefs=None):
    """A model is used unless its toggle was switched off (prefs.json detector_off: [keys])."""
    pr = prefs if prefs is not None else _prefs()
    return key not in (pr.get("detector_off") or [])


def _chain(kind, prefs=None):
    """[(key, label)] of the switched-on models of a kind, built-in ones first, then yours in the order added."""
    pr = prefs if prefs is not None else _prefs()
    base = BODY if kind == "body" else WATERMARK
    out = [(m["key"], m["label"]) for m in base] + [(m["key"], m["label"]) for m in custom_models(pr) if m["kind"] == kind]
    return [(k, lab) for k, lab in out if is_on(k, pr)]


def resolve_model_file(p):
    """An Add model path: an .onnx file, or a folder holding one (model.onnx first, else the only .onnx in it)."""
    p = str(p or "").strip().strip('"')
    if os.path.isdir(p):
        found = sorted(f for f in os.listdir(p) if f.lower().endswith(".onnx"))
        if "model.onnx" in found:
            found = ["model.onnx"]
        if len(found) != 1:
            raise ValueError("That folder needs exactly one .onnx model in it (or a model.onnx)." if found else
                             "There's no .onnx model in that folder.")
        p = os.path.join(p, found[0])
    if p.lower().endswith(".pt"):
        raise ValueError("A .pt model needs converting to .onnx once first: tools/convert_detector_to_onnx.py "
                         "(docs/FACE_BACKENDS.md). Then add the .onnx.")
    if not p:
        raise ValueError("Pick a model file or folder first.")
    return check_path(p)


def model_path(key, prefs=None):
    """The model's .onnx: the path saved for it (prefs.json detector_<key>), else user_data/models/detectors/<file>.
    "" when it isn't there."""
    pr = prefs if prefs is not None else _prefs()
    if key not in MODELS:
        c = next((m for m in custom_models(pr) if m["key"] == key), None)
        p = c["path"] if c else ""
        return p if p.lower().endswith(".onnx") and os.path.isfile(p) else ""
    p = str(pr.get(f"detector_{key}") or "").strip().strip('"')
    p = p or os.path.join(models_dir(), MODELS[key]["file"])
    return p if p.lower().endswith(".onnx") and os.path.isfile(p) else ""


def saved_paths(prefs=None):
    pr = prefs if prefs is not None else _prefs()
    return {k: str(pr.get(f"detector_{k}") or "") for k in MODELS}


def check_path(p):
    """A path the page sends for a model: "" (the default place) or an .onnx file that is there."""
    p = str(p or "").strip().strip('"')
    if p and not (p.lower().endswith(".onnx") and os.path.isfile(p)):
        raise ValueError("Pick an .onnx model on the Fizgig computer. A Civitai .pt file needs converting once: "
                         "tools/convert_detector_to_onnx.py (docs/FACE_BACKENDS.md).")
    return p


def _backend_class():
    from fizgig.face_utils import YoloOnnxBackend

    class DetectorModel(YoloOnnxBackend):
        """A YOLO detect or segment model (yolov8 / yolo11, ultralytics export) - every class, boxes only."""
        name = "yolo"

        def _pick_classes(self):
            return None
    return DetectorModel


def load(key, path=None, conf=CONF):
    """The model, loaded; None when its file isn't there or it can't load."""
    p = path or model_path(key)
    if not p:
        return None
    b = _backend_class()(p, conf=conf)
    c = {m["key"]: m["label"] for m in custom_models()}
    b.label = MODELS[key]["label"] if key in MODELS else c.get(key) or os.path.basename(p)
    try:
        b.load()
    except Exception:
        return None
    return b if b.names else None


def classes_for(word, names):
    """The class ids of names ({id: name}) that count as word (itself or a SYNONYMS entry)."""
    from fizgig.face_utils import class_key
    k = class_key(word)
    want = {class_key(w) for w in [word] + SYNONYMS.get(k, [])}
    return [i for i, n in sorted((names or {}).items()) if class_key(n) in want]


def stages(log=print, face_backend=None, loader=load):
    """[(label, backend)] in chain order: the YOLO face model (when it is the face finder), then the body models whose
    files are there. A missing one is said once and skipped."""
    out = []
    if face_backend is not None:
        out.append((getattr(face_backend, "label", "") or "YOLO face model", face_backend))
    for key, label in _chain("body"):           # switched-off models are left out
        b = loader(key)
        if b is None:
            _once(log, ("missing", key), f"[detectors] {label} isn't installed - skipped "
                                         f"(Anime/furry/drawn characters -> Body-part detectors)\n")
            continue
        out.append((label, b))
    return out


def route(words, stage_list):
    """{word: [(stage index, class ids), ...]} - the models that can find each word, in chain order."""
    out = {}
    for w in words:
        r = []
        for i, (_label, b) in enumerate(stage_list):
            try:
                ids = classes_for(w, b.all_classes())
            except Exception:
                ids = []
            if ids:
                r.append((i, ids))
        out[w] = r
    return out


def _share_text(b, w, h):
    return f"{100 * (b[2] - b[0]) * (b[3] - b[1]) / float(w * h or 1):.0f}%"


def chain_finder(words, stage_list, seg=None, log=print, fit_all=False, max_share=None, seg_factory=None):
    """A prep.ThingFinder that sends each word down the chain (SAM 3.1 last: seg, or seg_factory() made the first time
    a word needs it - SAM 3.1 is only loaded when a model couldn't find the word)."""
    from fizgig.dataset import prep

    class ChainFinder(prep.ThingFinder):
        def __init__(self):
            super().__init__(seg, words, fit_all=fit_all)
            self.stages = stage_list
            self.routes = route(self.words, stage_list)
            self.yolo_words = {w: 0 for w, r in self.routes.items() if r}
            self.sam_words = [w for w in self.words if not self.routes[w]]
            self.max_share = prep.HIT_MAX_SHARE if max_share is None else max_share

        def describe(self):
            parts = []
            for w in self.words:
                r = self.routes[w]
                who = [self.stages[i][0] for i, _ in r] + (["SAM 3.1"] if self.seg is not None or seg_factory else [])
                parts.append(f"{w}: {' then '.join(who) if who else 'nothing installed finds it'}")
            return "; ".join(parts)

        def find(self, img):
            from fizgig.face_utils import _face_info
            arr = np.asarray(img.convert("RGB"))
            h, w = arr.shape[:2]
            cache, out = {}, []
            for word in self.words:
                got = False
                for i, ids in self.routes[word]:
                    if i not in cache:
                        need = sorted({c for r in self.routes.values() for j, cs in r if j == i for c in cs})
                        try:
                            cache[i] = self.stages[i][1].find_classes(arr, need)
                        except Exception as e:
                            _once(log, ("err", i), f"[detectors] {self.stages[i][0]} failed: {e}\n")
                            cache[i] = []
                    hits = [t for t in cache[i] if int(t[5]) in ids]
                    kept = []
                    for x1, y1, x2, y2, s, _c in hits:
                        b = self._clamped((x1, y1, x2, y2), w, h)
                        if b:                              # a detector's box is trusted, however big
                            kept.append(_face_info(*b, gender=word, score=float(s), source="yolo"))
                    if kept:
                        kept.sort(key=lambda f: -f.score)
                        log(f"  {word}: found by {self.stages[i][0]} ("
                            + ", ".join(f"score {f.score:.2f}, {_share_text(f.bbox, w, h)} of the picture"
                                        for f in kept[:3]) + ")\n")
                        out += kept
                        got = True
                        break
                if not got and self.seg is None and seg_factory is not None:
                    self.seg = seg_factory()
                if not got and self.seg is not None:
                    for f in self.seg.masks(arr, [word], threshold=self.threshold, refine=0):
                        if f.get("score", 1.0) < self.threshold:
                            continue
                        b = self._clamped(f["box"], w, h)
                        if b:
                            log(f"  {word}: found by SAM 3.1 (score {f.get('score', 1.0):.2f}, "
                                f"{_share_text(b, w, h)} of the picture)\n")
                            out.append(_face_info(*b, gender=word, score=f.get("score", 1.0), source="sam3"))
            return out

    return ChainFinder()


# ---- watermark masks ----
TRACE_PAD = 0.15        # SAM looks this share of the box's size around it (YOLO boxes often stop short of the text)
TRACE_PAD_X = 0.30      # ... and more sideways, where text runs on
TRACE_PAD_MIN = 12      # px, at least, each way
TRACE_GROW = 3          # px the cleaned shape is grown by, to catch soft edges
TRACE_CLOSE = 5         # px: the morphological close that joins letters and fills small holes
TRACE_SPECK = 0.004     # a piece under this share of the box (and under 6 px) is a speck: dropped
TRACE_MIN = 0.03        # a cleaned shape under this share of the box is not trusted: the padded box is used
TRACE_MAX = 0.85        # ... nor one covering more than this of the padded area (SAM took the picture's texture)
TRACE_PIECES = 12       # ... nor one in more pieces than this (speckled)


def padded_box(box, w, h):
    """The YOLO box grown for tracing (TRACE_PAD, more sideways), kept on the picture."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = max(TRACE_PAD_MIN, int(TRACE_PAD_X * bw)), max(TRACE_PAD_MIN, int(TRACE_PAD * bh))
    return max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)


def clean_trace(t, box_area):
    """SAM's raw shape (bool crop) -> cleaned bool, or None when it's too noisy to trust: specks dropped, letters joined
    and small holes filled (close), then judged by its pieces and how much it covers."""
    import cv2
    t = np.asarray(t, np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TRACE_CLOSE, TRACE_CLOSE))
    n, lab, st, _ = cv2.connectedComponentsWithStats(t, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= max(6, TRACE_SPECK * box_area)
    t = keep[lab].astype(np.uint8)
    t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, k)
    area = int(t.sum())
    if area < TRACE_MIN * box_area or area > TRACE_MAX * t.size:
        return None
    if cv2.connectedComponents(t, connectivity=8)[0] - 1 > TRACE_PIECES:
        return None
    if TRACE_GROW:
        t = cv2.dilate(t, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TRACE_GROW + 1, 2 * TRACE_GROW + 1)))
    return t.astype(bool)


def trace_box(arr, box, tracer):
    """bool (H, W): the watermark in `box` traced by tracer(crop uint8 (h, w, 3)) -> bool (h, w) or None, on the
    padded box (padded_box), cleaned (clean_trace). Falls back to the padded box when the trace is empty, tiny, noisy
    or fails; without a tracer, the box as it is."""
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = box
    out = np.zeros((h, w), bool)
    if tracer is None or x2 <= x1 or y2 <= y1:
        out[y1:y2, x1:x2] = True
        return out
    cx1, cy1, cx2, cy2 = padded_box(box, w, h)
    out[cy1:cy2, cx1:cx2] = True
    try:
        t = tracer(arr[cy1:cy2, cx1:cx2])
    except Exception:
        t = None
    if t is None or np.asarray(t).shape != (cy2 - cy1, cx2 - cx1):
        return out
    t = clean_trace(t, (x2 - x1) * (y2 - y1))
    if t is None:
        return out
    traced = np.zeros((h, w), bool)
    traced[cy1:cy2, cx1:cx2] = t
    return traced


MERGE_GAP = 0.5         # boxes closer than this x the smaller one's height are one watermark (merge_boxes)


def merge_boxes(boxes):
    """Stacked, overlapping or nearly touching boxes (several detectors, or one watermark found in parts) -> their
    union boxes, merged until none are left close: a gap under MERGE_GAP x the smaller box's height joins them."""
    out = [tuple(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                gap = MERGE_GAP * min(a[3] - a[1], b[3] - b[1])
                gx = max(a[0], b[0]) - min(a[2], b[2])
                gy = max(a[1], b[1]) - min(a[3], b[3])
                if gx < gap and gy < gap:
                    out[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    del out[j]
                    changed = True
                    break
            if changed:
                break
    return out


def watermark_cover(arr, boxes, tracer=None):
    """bool (H, W): what's blacked out - each box, or with a tracer each merged box's traced shape (trace_box)."""
    cover = np.zeros(arr.shape[:2], bool)
    if tracer is not None:
        boxes = merge_boxes(boxes)
    for b in boxes:
        cover |= trace_box(arr, b, tracer)
    return cover


def watermark_models(loader=None, log=print):
    loader = loader or (lambda k: load(k, conf=WM_SCORE))
    out = []
    for key, label in _chain("watermark"):
        b = loader(key)
        if b is None:
            _once(log, ("missing", key), f"[watermarks] {label} isn't installed - skipped\n")
            continue
        out.append(b)
    return out


def watermark_boxes(arr, models):
    h, w = arr.shape[:2]
    out = []
    for b in models:
        try:
            ids = sorted((b.all_classes() or {}).keys())
            for x1, y1, x2, y2, s, _c in b.find_classes(arr, ids):
                if s >= WM_SCORE and (x2 - x1) * (y2 - y1) <= WM_MAX_SHARE * w * h:
                    out.append((max(0, int(x1) - WM_PAD), max(0, int(y1) - WM_PAD),
                                min(w, int(np.ceil(x2)) + WM_PAD), min(h, int(np.ceil(y2)) + WM_PAD)))
        except Exception:
            continue
    return out


def watermark_mask(image_path, img, models, tracer=None):
    """Paint the watermarks black on the picture's mask (beside it: <stem>-masklabel.png, merged with the mask it
    already has). Returns the share of the picture covered (0 when none found: nothing written)."""
    from PIL import Image
    from fizgig.dataset.masks import MASK_SUFFIX, mask_for
    arr = np.asarray(img.convert("RGB"))
    boxes = watermark_boxes(arr, models)
    if not boxes:
        return 0.0
    h, w = arr.shape[:2]
    m = np.full((h, w), 255, np.uint8)
    old = mask_for(image_path)
    if old:
        with Image.open(old) as o:
            m = np.asarray(o.convert("L").resize((w, h))).copy()
    cover = watermark_cover(arr, boxes, tracer)
    m[cover] = 0
    stem = os.path.splitext(os.path.basename(image_path))[0]
    Image.fromarray(m, "L").save(os.path.join(os.path.dirname(image_path), stem + MASK_SUFFIX + ".png"))
    return float(cover.mean())


def watermarks_only(shape, boxes, base=None, cover=None):
    """Just hide the watermarks: the mask (h, w) uint8 - `base` (a mask already there, same size) or all white - with
    each watermark box black (existing AND not-watermark), or None when there are no boxes (nothing to change)."""
    if not boxes:
        return None
    h, w = shape[:2]
    m = np.full((h, w), 255, np.uint8) if base is None else np.array(base, np.uint8)
    if cover is not None:                       # traced shapes (watermark_cover) instead of the boxes
        m[cover] = 0
        return m
    for x1, y1, x2, y2 in boxes:
        m[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 0
    return m


def watermarks_only_folder(folder, models, overwrite=False, log=print, tracer=None):
    """Make masks' Just hide the watermarks. A picture with a mask already (e.g. from a character-mask run) gets its
    watermarks blacked out on top of it (no watermark: the mask is kept as it is). With overwrite, old masks are
    ignored: a fresh all-white mask minus the watermarks, and none (the old one removed) where none is found.
    Returns {"made", "merged", "kept", "skipped", "no_person", "removed"} (no_person: no watermark, no mask)."""
    from PIL import Image
    from fizgig.dataset import prep
    from fizgig.dataset.masks import MASK_SUFFIX, mask_for
    c = {"made": 0, "merged": 0, "kept": 0, "skipped": 0, "no_person": 0, "removed": 0}
    for p in prep.image_files(folder):
        name, old = os.path.basename(p), mask_for(p)
        try:
            img = prep.load_image(p)
            try:
                arr = np.asarray(img.convert("RGB"))
            finally:
                img.close()
            boxes = watermark_boxes(arr, models)
            base = None
            if old and not overwrite:
                with Image.open(old) as o:
                    base = np.asarray(o.convert("L").resize((arr.shape[1], arr.shape[0]))).copy()
            m = watermarks_only(arr.shape, boxes, base, watermark_cover(arr, boxes, tracer) if tracer else None)
        except Exception as e:
            log(f"  {name}: not checked ({e})\n")
            c["skipped"] += 1
            continue
        if m is None:
            if base is not None:
                c["kept"] += 1
                log(f"  {name}: no watermark found - its mask is kept as it is\n")
                continue
            gone = 0
            while old and gone < 2:            # Replace: the old mask goes (beside it, and in masks/)
                os.remove(old)
                gone += 1
                old = mask_for(p)
            c["removed" if gone else "no_person"] += 1
            log(f"  {name}: no watermark found - " + ("old mask removed" if gone else "no mask") + ", trains as normal\n")
            continue
        stem = os.path.splitext(name)[0] + MASK_SUFFIX
        where = os.path.dirname(old) if base is not None else os.path.dirname(p)
        Image.fromarray(m, "L").save(os.path.join(where, stem + ".png"))
        if base is not None and os.path.normcase(old) != os.path.normcase(os.path.join(where, stem + ".png")):
            os.remove(old)                     # a .jpg/.webp mask replaced by its merged .png
        c["merged" if base is not None else "made"] += 1
        share = float((m == 0).mean()) if base is None else \
            float(sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes)) / (arr.shape[0] * arr.shape[1])
        log(f"  {name}: watermarks {100 * share:.1f}% of it" + (", merged with its mask" if base is not None else "")
            + (" - worth a look" if share > WM_WARN else "") + "\n")
    return c


def watermark_folder(folder, models, log=print, tracer=None):
    """Make masks' Mask watermarks: every picture in the folder (top level) gets its watermarks masked. Returns
    [(name, share covered)] of the pictures with any."""
    from fizgig.dataset import prep
    found = []
    for p in prep.image_files(folder):
        try:
            img = prep.load_image(p)
            try:
                share = watermark_mask(p, img, models, tracer)
            finally:
                img.close()
        except Exception as e:
            log(f"  {os.path.basename(p)}: watermarks not checked ({e})\n")
            continue
        if share:
            found.append((os.path.basename(p), share))
    log(f"Watermarks masked: {len(found)} picture(s)\n")
    big = [f"{n} ({100 * s:.0f}%)" for n, s in found if s > WM_WARN]
    if big:
        log("Watermarks cover a lot of these - worth a look: " + ", ".join(big) + "\n")
    return found


def download_booru():
    """booru_yolo's .onnx into user_data/models/detectors (the page's Download button - the user asks for it)."""
    from fizgig.utils import hf_public
    import shutil
    m = MODELS["booru"]
    os.makedirs(models_dir(), exist_ok=True)
    p = hf_public.download(**m["hf"])
    dest = os.path.join(models_dir(), m["file"])
    shutil.copyfile(p, dest)
    return dest


def info(prefs=None):
    """For the page: each model, what it finds, where its file goes, whether it's there."""
    pr = prefs if prefs is not None else _prefs()
    return [{"key": m["key"], "label": m["label"], "finds": m["finds"], "page": m["page"],
             "kind": "watermark" if m in WATERMARK else "body", "download": "hf" in m,
             "default": os.path.join(models_dir(), m["file"]), "saved": saved_paths(pr)[m["key"]],
             "installed": bool(model_path(m["key"], pr)), "on": is_on(m["key"], pr), "custom": False}
            for m in BODY + WATERMARK] + [
            {"key": m["key"], "label": m["label"], "finds": "your own model" + (" (watermarks)" if m["kind"] == "watermark"
             else ""), "page": "", "kind": m["kind"], "download": False, "default": m["path"], "saved": m["path"],
             "installed": bool(model_path(m["key"], pr)), "on": is_on(m["key"], pr), "custom": True}
            for m in custom_models(pr)]
