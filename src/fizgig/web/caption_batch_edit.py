"""Atomic, optimistic caption edits for Caption Studio group undo/redo.

Each edit names the media file, its caption extension, the expected on-disk text
(or None when missing), and the replacement text. All preconditions, including
review locks, are checked before the first file is modified.
"""
import os
import re
import threading

from fizgig.web import caption_review, captions
from fizgig.web.image_edit import write_atomic

_LOCK = threading.RLock()


def apply(edits, ext=".txt"):
    if not isinstance(edits, list) or not 1 <= len(edits) <= 1000:
        raise ValueError("Choose 1–1000 caption changes.")
    if not isinstance(ext, str) or not re.fullmatch(r"\.[A-Za-z0-9_]{1,12}", ext):
        raise ValueError("Invalid caption extension.")
    planned, seen = [], set()
    with _LOCK, caption_review._lock:
        for entry in edits:
            if not isinstance(entry, dict):
                raise ValueError("Invalid caption change.")
            folder, name = entry.get("folder"), entry.get("name")
            if not isinstance(folder, str) or not os.path.isdir(folder):
                raise ValueError("Caption dataset folder was not found.")
            if not isinstance(name, str) or not name or os.path.basename(name) != name or name in (".", ".."):
                raise ValueError("Invalid media filename.")
            if not os.path.isfile(os.path.join(folder, name)):
                raise ValueError("The original media file is missing: " + name)
            before, after = entry.get("expected"), entry.get("text")
            if before is not None and not isinstance(before, str):
                raise ValueError("Invalid expected caption text.")
            if after is not None and not isinstance(after, str):
                raise ValueError("Invalid replacement caption text.")
            if caption_review.is_locked(folder, name) or caption_review.is_caption_locked(
                os.path.join(folder, os.path.splitext(name)[0] + ext)
            ):
                raise ValueError("Reviewed caption is locked: " + name)
            target = os.path.join(folder, os.path.splitext(name)[0] + ext)
            key = os.path.normcase(os.path.realpath(target))
            if key in seen:
                raise ValueError("Two changes target the same caption: " + name)
            seen.add(key)
            actual = captions.read_text(target) if os.path.isfile(target) else None
            if actual != before:
                raise ValueError("Caption changed outside Studio: " + name + ". Refresh before retrying.")
            planned.append((folder, name, target, actual, after))

        written = []
        try:
            for folder, name, target, before, after in planned:
                if before == after:
                    continue
                if after is None:
                    os.remove(target)
                else:
                    write_atomic(target, after.encode("utf-8"))
                written.append((target, before))
        except Exception as exc:
            rollback_errors = []
            for target, before in reversed(written):
                try:
                    if before is None:
                        if os.path.isfile(target):
                            os.remove(target)
                    else:
                        write_atomic(target, before.encode("utf-8"))
                except OSError as rollback_error:
                    rollback_errors.append(str(rollback_error))
            if rollback_errors:
                raise RuntimeError("Batch failed and rollback needs attention: " + "; ".join(rollback_errors)) from exc
            raise
    return [{"folder": folder, "name": name, "before": before, "after": after}
            for folder, name, _, before, after in planned if before != after]
