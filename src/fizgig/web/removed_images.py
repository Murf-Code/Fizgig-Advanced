"""Persistent, lossless Caption Studio removal history.

The dataset's removed/ directory is the source of truth. The journal remembers
original names when collision-safe moves append " (2)", so a later restore does
not guess and does not overwrite live dataset files. Older untracked files are
offered as same-name restores only.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid

_LOCK = threading.RLock()
_FILENAME = ".fizgig_caption_removals.json"
_PICTURE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif",
                 ".jxl", ".gif", ".tif", ".tiff"}


def _removed(folder):
    return os.path.join(os.path.abspath(folder), "removed")


def _journal(folder):
    return os.path.join(_removed(folder), _FILENAME)


def _clean_pairs(pairs):
    out = []
    for pair in pairs or ():
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            continue
        orig, stored = pair
        def flat(name):
            return (isinstance(name, str) and name and "/" not in name and
                    "\\" not in name and name not in (".", ".."))
        # Masks may live in <dataset>/masks, but removed/ is flat. Keep the
        # original relative directory so restoring returns them correctly.
        original_ok = flat(orig) or (isinstance(orig, str) and
            orig.startswith("masks/") and orig.count("/") == 1 and flat(orig[6:]))
        if not original_ok or not flat(stored):
            continue
        out.append([orig, stored])
    return out


def _load(folder):
    try:
        with open(_journal(folder), encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [entry for entry in data if isinstance(entry, dict) and
            isinstance(entry.get("id"), str) and isinstance(entry.get("moved"), list)]


def _save(folder, entries):
    target = _journal(folder)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    temp = target + ".tmp"
    with open(temp, "w", encoding="utf-8") as stream:
        json.dump(entries, stream, indent=2, ensure_ascii=False)
    os.replace(temp, target)


def record(folder, moved):
    pairs = _clean_pairs(moved)
    if not pairs:
        return ""
    with _LOCK:
        rows = _load(folder)
        ident = uuid.uuid4().hex
        rows.append({"id": ident, "name": pairs[0][0], "moved": pairs,
                     "time": time.time()})
        _save(folder, rows)
        return ident


def listing(folder, caption_ext=".txt"):
    if not os.path.isdir(_removed(folder)):
        return []
    with _LOCK:
        rows = _load(folder)
        covered = set()
        items = []
        for entry in rows:
            pairs = [pair for pair in _clean_pairs(entry.get("moved"))
                     if os.path.isfile(os.path.join(_removed(folder), pair[1]))]
            if not pairs:
                continue
            covered.update(pair[1] for pair in pairs)
            items.append({"id": entry["id"], "name": entry.get("name") or pairs[0][0],
                          "moved": pairs, "time": entry.get("time", 0), "legacy": False})
        # Older builds moved files without journaling them. They can be safely
        # restored under exactly their stored filename (never guess original
        # names for collision-renamed files).
        names = sorted(os.listdir(_removed(folder)))
        for name in names:
            stem, ext = os.path.splitext(name)
            if (ext.lower() not in _PICTURE_EXTS or name in covered or
                    not os.path.isfile(os.path.join(_removed(folder), name))):
                continue
            pairs = [[name, name]]
            for suffix in {".txt", caption_ext}:
                candidate = stem + suffix
                if candidate not in covered and os.path.isfile(os.path.join(_removed(folder), candidate)):
                    pairs.append([candidate, candidate])
                    covered.add(candidate)
            covered.add(name)
            items.append({"id": "legacy:" + name, "name": name, "moved": pairs,
                          "time": 0, "legacy": True})
        return sorted(items, key=lambda it: (-float(it["time"] or 0), it["name"].lower()))


def lookup(folder, ident, caption_ext=".txt"):
    return next((row for row in listing(folder, caption_ext)
                 if row["id"] == ident), None)


def reconcile(folder, moved, restored):
    """Remove restored pairs from durable records, leaving partial conflicts visible."""
    done = set(restored)
    if not done:
        return
    pairs = {tuple(p) for p in _clean_pairs(moved) if p[0] in done}
    with _LOCK:
        entries = _load(folder)
        for entry in entries:
            entry["moved"] = [pair for pair in _clean_pairs(entry["moved"])
                              if tuple(pair) not in pairs]
        _save(folder, [entry for entry in entries if entry["moved"]])
