"""Persistent Caption Studio review locks.

A reviewed picture is protected from direct Caption Studio save/removal routes.
Other external tools and direct filesystem edits are not restricted by this ledger.
"""
import json
import os
import threading

_lock = threading.RLock()
_FILENAME = ".fizgig_caption_reviews.json"


def _path(folder):
    folder = os.path.realpath(str(folder or ""))
    if not os.path.isdir(folder):
        raise ValueError("The dataset folder does not exist.")
    return os.path.join(folder, _FILENAME)


def _load(folder):
    path = _path(folder)
    try:
        with open(path, encoding="utf-8") as f:
            values = json.load(f)
    except FileNotFoundError:
        return set()
    except (OSError, ValueError) as exc:
        raise ValueError("Could not read caption review status; nothing was overwritten.") from exc
    if not isinstance(values, list) or not all(isinstance(x, str) for x in values):
        raise ValueError("Invalid caption review status; nothing was overwritten.")
    return set(values)


def reviewed(folder):
    with _lock:
        return sorted(_load(folder))


def is_locked(folder, name):
    with _lock:
        return str(name) in _load(folder)


def mark(folder, name, value):
    name = str(name or "")
    if not name or os.path.basename(name) != name or name in (".", ".."):
        raise ValueError("Choose a file in the dataset folder.")
    with _lock:
        values = _load(folder)
        if value:
            values.add(name)
        else:
            values.discard(name)
        path = _path(folder)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(sorted(values), f, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.isfile(tmp):
                os.remove(tmp)
    return sorted(values)
