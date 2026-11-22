"""Read the newest matching run's per-image learning verdicts for Caption Workbench.

The UI consumes this small snapshot, not the full per-epoch loss chart. No
training or GPU work is performed. Only the loss watch decides a verdict.
"""
import os

from fizgig.web import image_loss


TRACKING_KEYS = ("KREA2_LOSS_WATCH", "KREA2_PER_IMAGE_LR",
                 "KREA2_WARMUP_LOOK", "KREA2_AUTO_RECAPTION")
ENABLED_VALUES = (True, "true", "True", "1", 1)


def tracking_enabled(run):
    """The same four controls used by the Training tab's Per-image loss button."""
    values = run.get("values") or {}
    return any(values.get(key) in ENABLED_VALUES for key in TRACKING_KEYS)


def snapshot(folder, runs, family_of, active_run_id=None):
    empty = {"run": None, "images": [], "auto_enable": False}
    if not folder or not os.path.isdir(folder):
        return empty
    root = os.path.normcase(os.path.realpath(folder))
    owners = image_loss.owner_runs(runs)
    for run in runs:  # history.runs() returns newest first
        other = str(run.get("image_folder") or "")
        if not other or os.path.normcase(os.path.realpath(other)) != root:
            continue
        family = family_of(run.get("family") or "")
        if not getattr(family, "loss_watch", False):
            continue
        active = (active_run_id is not None and run.get("id") == active_run_id
                  and run.get("result") == "running" and tracking_enabled(run))
        # A current watched run should enable its colors even before it writes
        # the first per-image loss record. Never borrow a previous run's log.
        owned = bool(owners.get(run.get("id")))
        if not owned and not active:
            continue
        data = image_loss.residuals(run.get("output_dir")) if owned else None
        if not data:
            if active:
                return {"run": {"id": run.get("id"), "name": run.get("lora"),
                                "epoch": None}, "images": [], "auto_enable": True}
            continue
        paths = image_loss.image_paths(run, [im["key"] for im in data["images"]])
        images = []
        for im in data["images"]:
            path = paths.get(im["key"])
            if not path:  # deleted or renamed pictures cannot get a ring
                continue
            # A verdict is a measured training status, never inferred from raw
            # image size, tag frequency or from a missing loss history.
            images.append({
                "path": path,
                "verdict": im.get("verdict"),
                "recent": im.get("recent"),
                "multiplier": im.get("multiplier"),
            })
        return {"run": {"id": run.get("id"), "name": run.get("lora"),
                        "epoch": (data.get("report") or {}).get("epoch")},
                "images": images, "auto_enable": active}
    return empty
