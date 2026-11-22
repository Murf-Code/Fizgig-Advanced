"""Read the newest matching run's per-image learning verdicts for Caption Workbench.

The UI consumes this small snapshot, not the full per-epoch loss chart. No
training or GPU work is performed. Only the loss watch decides a verdict.
"""
import os

from fizgig.web import image_loss


def snapshot(folder, runs, family_of):
    empty = {"run": None, "images": []}
    if not folder or not os.path.isdir(folder):
        return empty
    root = os.path.normcase(os.path.realpath(folder))
    owners = image_loss.owner_runs(runs)
    for run in runs:  # history.runs() returns newest first
        other = str(run.get("image_folder") or "")
        if not other or os.path.normcase(os.path.realpath(other)) != root:
            continue
        if not owners.get(run.get("id")):
            continue  # never attach an older run to a newer run's overwritten log
        family = family_of(run.get("family") or "")
        if not getattr(family, "loss_watch", False):
            continue
        data = image_loss.residuals(run.get("output_dir"))
        if not data:
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
                "images": images}
    return empty
