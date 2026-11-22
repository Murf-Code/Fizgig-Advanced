"""CPU-only near-duplicate explorer and conservative tag merging for Caption Studio.

No training model, GPU, or network calls are used. Pictures are decoded one at
a time by a background thread. The threshold uses a 63-bit perceptual DCT hash
and a mean-color sanity check, so visually unrelated solid-color pictures are
not mistakenly presented as exact duplicates. All matches require human review.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import threading
from collections import defaultdict
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps

from fizgig.dataset.masks import is_mask_file

PHOTO_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif",
              ".tif", ".tiff", ".avif"}
MAX_FILES = 30000
MAX_PAIRS = 150000
MIN_SCORE = 75
_JOBS: dict[str, dict] = {}
_LOCK = threading.RLock()
_WRITE_LOCK = threading.RLock()


def _root(folder):
    folder = os.path.realpath(os.path.abspath(str(folder or "")))
    if not folder or not os.path.isdir(folder):
        raise ValueError("Choose an existing training dataset folder.")
    return folder


def checked_path(root, path):
    """Only pictures inside a dataset, never masks, backups or removed files."""
    if not isinstance(path, str) or not path:
        raise ValueError("A picture path is missing.")
    p = os.path.realpath(os.path.abspath(
        path if os.path.isabs(path) else os.path.join(root, path)))
    if os.path.commonpath((root, p)) != root or p == root:
        raise ValueError("Picture is outside this dataset.")
    parts = os.path.relpath(p, root).split(os.sep)
    if any(x.lower() in ("removed", "removed_watermarks", ".git", "__pycache__")
           or x.startswith(".") for x in parts[:-1]):
        raise ValueError("Removed/hidden folders are not training pictures.")
    if os.path.splitext(p)[1].lower() not in PHOTO_EXTS or is_mask_file(os.path.basename(p)):
        raise ValueError("Only source photos may be compared or merged.")
    if not os.path.isfile(p):
        raise ValueError(f"Picture no longer exists: {os.path.basename(p)}")
    return p


def paths_for_scan(folder, paths):
    root = _root(folder)
    if not isinstance(paths, list) or len(paths) > MAX_FILES:
        raise ValueError(f"Send at most {MAX_FILES} dataset pictures.")
    result = {}
    for path in paths:
        try:
            p = checked_path(root, path)
            st = os.stat(p)
            result[p] = (st.st_size, st.st_mtime_ns)
        except (ValueError, OSError):
            continue  # an edited, removed or still-uploading file can vanish mid-scan
    return root, sorted(result), result


@lru_cache(maxsize=1)
def _dct_basis():
    n = 32
    v = np.arange(n, dtype=np.float64)
    basis = np.cos((np.pi / (2 * n)) * np.outer(v, 2 * v + 1))
    return basis[:8, :]


def fingerprint(path):
    with Image.open(path) as src:
        src.seek(0)
        im = ImageOps.exif_transpose(src).convert("RGB")
        width, height = im.size
        rgb = np.asarray(im.resize((8, 8), Image.Resampling.BOX), dtype=np.float64)
        mean = tuple(float(v) for v in rgb.mean(axis=(0, 1)))
        gray = np.asarray(im.resize((32, 32), Image.Resampling.LANCZOS).convert("L"),
                          dtype=np.float64)
    b = _dct_basis()
    coefficients = (b @ gray @ b.T).ravel()[1:]  # ignore the DC/lightness coefficient
    median = np.median(coefficients)
    bits = sum(1 << i for i, v in enumerate(coefficients) if v > median)
    return (bits, mean, width, height)


def distance(a, b):
    return (a ^ b).bit_count()


def similarity(a, b):
    # Mean color prevents a monotone/solid red picture from appearing identical
    # to the same-shaped blue picture just because their pHashes have no detail.
    phash = 100.0 * (1.0 - distance(a[0], b[0]) / 63.0)
    color = 100.0 * (1.0 - sum(abs(x - y) for x, y in zip(a[1], b[1])) / (3 * 255.0))
    return round(max(0.0, min(phash, color)), 1)


class _BKTree:
    """Metric tree over Hamming bit distance; no quadratic all-pairs scan."""
    def __init__(self):
        self.root = None

    def insert(self, key, index):
        if self.root is None:
            self.root = [key, [index], {}]
            return
        node = self.root
        while True:
            d = distance(key, node[0])
            if not d:
                node[1].append(index)
                return
            child = node[2].get(d)
            if child is None:
                node[2][d] = [key, [index], {}]
                return
            node = child

    def matches(self, key, radius):
        if self.root is None:
            return
        stack = [self.root]
        while stack:
            node = stack.pop()
            d = distance(key, node[0])
            if d <= radius:
                yield from node[1]
            stack.extend(child for k, child in node[2].items()
                         if d - radius <= k <= d + radius)


def _compute(job):
    tree = _BKTree()
    features = {}
    files = job["paths"]
    failures = []
    for i, path in enumerate(files):
        try:
            expected = job["stamps"][path]
            st = os.stat(path)
            if (st.st_size, st.st_mtime_ns) != expected:
                raise ValueError("changed during duplicate scan")
            f = fingerprint(path)
            st = os.stat(path)
            if (st.st_size, st.st_mtime_ns) != expected:
                raise ValueError("changed while being decoded")
            features[path] = f
        except (OSError, ValueError, Image.UnidentifiedImageError) as e:
            if len(failures) < 12:
                failures.append(os.path.basename(path) + ": " + str(e))
        if i % 12 == 0 or i + 1 == len(files):
            with _LOCK:
                job["processed"] = i + 1
    with _LOCK:
        job["stage"] = "matching"
    pairs = []
    truncated = False
    records = list(features.items())
    # Threshold is independently adjustable afterwards. Precompute all edges
    # at the lowest supported threshold and filter cached scores in status().
    radius = math.ceil(63 * (100 - MIN_SCORE) / 100.0)
    for i, (_path, f) in enumerate(records):
        for j in tree.matches(f[0], radius):
            other_path, other = records[j]
            score = similarity(other, f)
            if score >= MIN_SCORE:
                pairs.append((other_path, _path, score))
                if len(pairs) >= MAX_PAIRS:
                    truncated = True
                    break
        if truncated:
            break
        tree.insert(f[0], i)
    with _LOCK:
        job.update(stage="done", files={p: {"path": p, "name": os.path.basename(p),
                                            "width": f[2], "height": f[3]}
                                        for p, f in features.items()},
                   pairs=pairs, failures=failures, truncated=truncated,
                   scanned=len(features))


def _worker(job):
    try:
        _compute(job)
    except Exception as exc:
        with _LOCK:
            job.update(stage="error", error=f"{type(exc).__name__}: {exc}")


def start(folder, paths):
    root, files, stamps = paths_for_scan(folder, paths)
    with _LOCK:
        old = _JOBS.get(root)
        if old and old["stage"] in ("scanning", "matching"):
            return status(root)
        job = {"stage": "scanning", "paths": files, "stamps": stamps,
               "total": len(files), "processed": 0, "files": {}, "pairs": [],
               "failures": [], "truncated": False, "scanned": 0}
        _JOBS[root] = job
        # Drop older folders' scans to bound retained fingerprints and matches.
        if len(_JOBS) > 3:
            for key in list(_JOBS):
                if key != root and _JOBS[key]["stage"] not in ("scanning", "matching"):
                    _JOBS.pop(key, None)
    threading.Thread(target=_worker, args=(job,), name="caption-duplicate-scan",
                     daemon=True).start()
    return status(root)


def status(folder, threshold=92, offset=0, page_size=32):
    root = _root(folder)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        raise ValueError("Similarity threshold must be a number.") from None
    if not MIN_SCORE <= threshold <= 100:
        raise ValueError(f"Similarity must be {MIN_SCORE}–100%.")
    try:
        offset = int(offset)
    except (ValueError, TypeError):
        offset = 0
    offset = max(0, offset)
    with _LOCK:
        j = _JOBS.get(root)
        if j is None:
            return {"stage": "idle", "processed": 0, "total": 0, "threshold": threshold}
        overview = {k: j[k] for k in ("stage", "processed", "total", "scanned",
                                      "failures", "truncated")}
        overview["threshold"] = threshold
        if j["stage"] == "error":
            overview["error"] = j.get("error", "Scan failed")
            return overview
        if j["stage"] != "done":
            return overview
        matches = [(a, b, score) for a, b, score in j["pairs"] if score >= threshold]
        info = j["files"]
    parent = {p: p for a, b, _ in matches for p in (a, b)}
    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p
    for a, b, _ in matches:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pb] = pa
    members = defaultdict(set)
    for p in parent:
        members[find(p)].add(p)
    scores = defaultdict(list)
    for a, _b, score in matches:
        scores[find(a)].append(score)
    groups = []
    for paths in members.values():
        sorted_paths = sorted(paths)
        groups.append({"id": hashlib.sha1(sorted_paths[0].encode("utf-8")).hexdigest()[:16],
                       "similarity": round(min(scores[find(sorted_paths[0])]), 1),
                       "count": len(paths),
                       "items": [info[p] for p in sorted_paths]})
    groups.sort(key=lambda g: (-g["similarity"], -g["count"], g["items"][0]["path"]))
    overview.update(groups=groups[offset:offset + page_size],
                    group_count=len(groups), pair_count=len(matches),
                    offset=offset, next_offset=(offset + page_size if
                        offset + page_size < len(groups) else None))
    return overview


def _caption_kind(text, delimiter):
    first = (text or "").splitlines()[0].strip() if text else ""
    if not first:
        return "tags"
    parts = [p.strip() for p in first.split(delimiter) if p.strip()]
    if re.search(r"[.!?](?:\s|$)", first) or any(len(p.split()) >= 8 for p in parts):
        return "prose"
    if len(parts) == 1 and len(parts[0].split()) >= 5:
        return "prose"
    return "tags"


def _caption_text(path, ext):
    target = os.path.splitext(path)[0] + ext
    try:
        with open(target, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return None


def merge_tags(folder, keeper, donors, *, ext=".txt", delimiter=",",
               expected=None, apply=False):
    """Non-destructively merge the first-line tags of selected near duplicates.

    Leaves the keeper's second and later lines untouched. Requires a snapshot
    of EVERY source caption, preventing stale UI data from overwriting edits.
    A missing caption is represented by null/None. Prose is never modified.
    """
    from fizgig.web import caption_review
    from fizgig.web.image_edit import write_atomic

    root = _root(folder)
    if delimiter not in (",", ";", "|"):
        raise ValueError("Unsupported tag separator.")
    if not isinstance(ext, str) or not re.fullmatch(r"\.[A-Za-z0-9]{1,10}", ext):
        raise ValueError("Invalid caption extension.")
    if not isinstance(donors, list) or not 1 <= len(donors) <= 100:
        raise ValueError("Select between 1 and 100 source images to merge.")
    keeper = checked_path(root, keeper)
    others = [checked_path(root, p) for p in donors]
    if keeper in others or len(set(others)) != len(others):
        raise ValueError("The keeper and source images must be different.")
    if not isinstance(expected, dict):
        raise ValueError("The current caption snapshots are required.")
    with _WRITE_LOCK:
        if caption_review.is_locked(os.path.dirname(keeper), os.path.basename(keeper)):
            raise ValueError("This keeper is reviewed and locked. Unlock it first.")
        paths = [keeper] + others
        captions = []
        for p in paths:
            if p not in expected or (expected[p] is not None and
                                     not isinstance(expected[p], str)):
                raise ValueError("A source caption snapshot is missing.")
            current = _caption_text(p, ext)
            if current != expected[p]:
                raise ValueError("A caption changed since the scan. Refresh before merging.")
            if _caption_kind(current, delimiter) != "tags":
                raise ValueError("A source uses prose rather than tags; merge it manually.")
            captions.append(current or "")
        original = captions[0]
        first, newline, rest = original.partition("\n")
        merged = [t.strip() for t in first.split(delimiter) if t.strip()]
        known = {x.casefold() for x in merged}
        added = []
        for text in captions[1:]:
            for tag in [t.strip() for t in text.splitlines()[0].split(delimiter)
                        if t.strip()] if text.splitlines() else []:
                if tag.casefold() not in known:
                    known.add(tag.casefold())
                    merged.append(tag)
                    added.append(tag)
        merged_text = (delimiter + " ").join(merged) + (newline + rest if newline else "")
        if apply and merged_text != original:
            # Recheck immediately before the atomic write.
            if any(_caption_text(p, ext) != expected[p] for p in paths):
                raise ValueError("Caption changed before saving; nothing was overwritten.")
            write_atomic(os.path.splitext(keeper)[0] + ext, merged_text.encode("utf-8"))
        return {"keeper": keeper, "caption": merged_text, "added": added,
                "added_count": len(added), "changed": merged_text != original,
                "applied": bool(apply and merged_text != original)}
