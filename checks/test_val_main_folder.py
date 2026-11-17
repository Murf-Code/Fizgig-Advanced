"""Automatic validation pick uses only the main training folder, not subfolders (crops and the like)."""
from fizgig.training import holdout
from fizgig.training.val_loss import auto_pick_keys, pick_keys


def _cands(main=10, crops=20, poses=20, resized=0):
    out = []
    if main:
        out.append(("", [f"/d/p{i}.png" for i in range(main)]))
    if resized:
        out.append(("resized/", [f"/d/resized/r{i}.png" for i in range(resized)]))
    if crops:
        out.append(("crops/", [f"/d/crops/c{i}.png" for i in range(crops)]))
    if poses:
        out.append(("poses/", [f"/d/poses/s{i}.png" for i in range(poses)]))
    return out


def test_auto_pick_uses_only_the_main_folder():
    paths, names, how, why = holdout.choose(_cands(), 4)
    assert how == "auto" and why == ""
    assert paths and all(p.startswith("/d/p") for p in paths)
    assert all(not n.startswith(("crops/", "poses/")) for n in names)
    # a small main folder: pick fewer from it, still no subfolders
    paths, names, how, why = holdout.choose(_cands(main=3, crops=30, poses=30), 4)
    assert how == "auto" and len(paths) == 1 and paths[0].startswith("/d/p")
    assert names == [n for n in names if not n.startswith(("crops/", "poses/"))]


def test_hand_picked_subfolder_image_is_kept():
    paths, names, how, why = holdout.choose(_cands(), 4, ["c0.png", "s1.png"])
    assert how == "picked" and why == ""
    assert names == ["crops/c0.png", "poses/s1.png"]


def test_fallback_when_main_folder_is_empty():
    # Image Prep: the main folder's pictures train from resized/, crops sit beside them
    paths, names, how, why = holdout.choose(_cands(main=0, resized=10, crops=20, poses=0), 4)
    assert how == "auto"
    assert "taken from resized/" in why
    assert paths and all("/resized/" in p for p in paths)
    assert all(n.startswith("resized/") for n in names)
    # no main, no resized: other training folders, but not crops
    paths, names, how, why = holdout.choose(_cands(main=0, resized=0, crops=20, poses=10), 4)
    assert how == "auto" and "crops left out" in why
    assert paths and all("/poses/" in p for p in paths)
    assert all(n.startswith("poses/") for n in names)
    # only crops: nothing automatic is held out
    paths, names, how, why = holdout.choose(_cands(main=0, resized=0, crops=20, poses=0), 4)
    assert paths == [] and how == "" and "no pictures in the main folder" in why


def test_under_five_pictures_still_holds_none():
    paths, names, how, why = holdout.choose(_cands(main=4, crops=0, poses=0), 4)
    assert paths == [] and "at least 5" in why


def test_fixed_validation_auto_prefers_the_main_folder():
    keys = [f"p{i}" for i in range(10)] + [f"crops/c{i}" for i in range(20)] + [f"poses/s{i}" for i in range(20)]
    wanted = auto_pick_keys(keys, 4)
    assert wanted and all("/" not in k for k in wanted)
    # a hold-out list (checking every key it was given) is left as it is, even from resized/
    held = [f"resized/r{i}" for i in range(2)]
    assert auto_pick_keys(held, 2) == pick_keys(held, 2)
