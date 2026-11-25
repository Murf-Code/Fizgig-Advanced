"""Local, opt-in generated-sample nudity detection; never modifies an image.

Reuses Fizgig's existing user-installed ONNX body detectors and CPU backend.
The NSFW all-in-one model is required for a negative result; the booru model
can supplement detection of illustrated/furry anatomy. No model download or
image upload is initiated here. This is a fallible display aid, not moderation.
"""
import os
import threading
from collections import OrderedDict

_LOCK = threading.Lock()
_CACHE = OrderedDict()
_CACHE_MAX = 2048
_EXPLICIT = ("nipples", "pussy", "anus", "penis", "testicles")
_SUPPLEMENT = ("boob",)


def _model_signature(detectors):
    out = []
    for key in ("aio", "booru"):
        p = detectors.model_path(key)
        if p:
            try:
                st = os.stat(p)
                out.append((key, p, st.st_mtime_ns, st.st_size))
            except OSError:
                pass
    return tuple(out)


def _class_ids(detectors, model, words):
    names = model.all_classes()
    return sorted({i for w in words for i in detectors.classes_for(w, names)})


def _classify(path, detectors, models):
    from PIL import Image, ImageOps
    import numpy as np
    if not models:
        return {"state": "unknown", "reason": "Install Anime NSFW All-in-One in Image Prep → Body-part detectors."}
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((1024, 1024))
        rgb = np.asarray(im)
    complete = False
    for key, model_path, *_ in models:
        model = detectors.load(key, path=model_path, conf=0.30)
        if model is None:
            continue
        ids = _class_ids(detectors, model, _EXPLICIT if key == "aio" else _EXPLICIT + _SUPPLEMENT)
        if not ids:
            continue
        if model.find_classes(rgb, ids):
            return {"state": "flagged", "reason": "Possible nudity detected."}
        if key == "aio":
            # A model with only unrelated classes cannot certify a negative.
            complete = all(_class_ids(detectors, model, (word,)) for word in ("penis", "pussy", "nipples"))
    if complete:
        return {"state": "clear"}
    return {"state": "unknown", "reason": "NSFW detector missing or incomplete. Install Anime NSFW All-in-One in Image Prep."}


def scan(path):
    from fizgig.dataset import detectors
    try:
        with _LOCK:  # one CPU inference at a time; no GPU/trainer allocation
            st = os.stat(path)
            models = _model_signature(detectors)
            key = (os.path.realpath(path), st.st_mtime_ns, st.st_size, models)
            if key in _CACHE:
                _CACHE.move_to_end(key)
                return dict(_CACHE[key])
            result = _classify(path, detectors, models)
            if result["state"] != "unknown":
                _CACHE[key] = result
                while len(_CACHE) > _CACHE_MAX:
                    _CACHE.popitem(last=False)
            return dict(result)
    except Exception:
        return {"state": "unknown", "reason": "Sample scan unavailable; click to reveal manually."}
