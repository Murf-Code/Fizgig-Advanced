"""Validation hold-out: a few pictures set aside for Image Validation for the whole run, so its line measures pictures
the LoRA never trains on (on by default).

How it works - nothing is moved. Start writes the run's hold-out list (<dataset TOML>_holdout.json: the pictures' full
paths) and names it in the dataset TOML (holdout_file). Every dataset built from that TOML leaves those pictures out:
the cache pass never caches them as training pictures (a training cache left from an earlier run is cleared as stale)
and training never draws them. The cache pass caches them on their own, plain caption only, in the cache folder's
_fizgig_holdout/ folder, which only validation reads. Start at a smaller size's dataset leaves them out too.

Giving them back: when the run ends - finished, stopped, paused with Save & stop, or failed - the list is emptied
(the pictures it held are kept in it as "returned"), so the next run or cache pass sees the whole folder again. The
pictures and captions never leave their folder, so a crash can't lose one; a list a crash left behind is rewritten by
the next Start before anything reads it. A resume holds out the same pictures as before: each restore point keeps the
list (training_state.json), and Resume reads it from there.

Which pictures: the ones picked in Image Validation (Your images to check), or - none picked - about a tenth of the
pictures (at least 1, at most Images to check), spread over the run's folders in proportion and, inside each, evenly
through its pictures sorted by shape, so every aspect is represented. Never every picture; a set under MIN_POOL
pictures holds none. The same dataset always gives the same pick. No torch here (the web form and Start use it)."""
import json
import os

HOLDOUT_CACHE = "_fizgig_holdout"   # the cache folder's sub-folder of the set-aside pictures' cache
SHARE = 0.10                        # about this much of the pictures when none are picked
MIN_POOL = 5                        # fewer pictures than this: none held out


def path_for(dataset_config):
    return os.path.splitext(str(dataset_config))[0] + "_holdout.json"


def _norm(p):
    return os.path.normcase(os.path.abspath(str(p)))


def read(path):
    """The hold-out file -> {"paths": [...], "names": [...], "how": ..., "returned": [...]}; {} when none/unreadable."""
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def content(paths=(), names=(), how="", why="", returned=()):
    """The hold-out file's text (written by Start through its files list, and by give_back)."""
    return json.dumps({"version": 1, "paths": [os.path.abspath(p) for p in paths], "names": list(names), "how": how,
                       "why": why, "returned": list(returned)}, indent=1) + "\n"


def write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def give_back(path):
    """The run is over: the list emptied, so every picture is in the dataset again. -> the names given back."""
    data = read(path)
    if not data.get("paths"):
        return []
    write(path, content(how=data.get("how", ""), returned=data.get("names") or data["paths"]))
    return list(data.get("names") or data["paths"])


def held_stems(holdout_file, image_directory):
    """The file stems (cache keys) of the held-out pictures in this dataset folder; a detail crop folder leaves out the
    crops of its training folder's held-out pictures (a crop is named as its picture)."""
    paths = read(holdout_file).get("paths") or []
    if not paths or not image_directory:
        return set()
    here = _norm(image_directory)
    from fizgig.dataset import detail_crops
    if detail_crops.is_crop_dir(image_directory):
        here = _norm(detail_crops.root_of(image_directory))
    return {os.path.splitext(os.path.basename(p))[0] for p in paths if _norm(os.path.dirname(p)) == here}


def _aspect(p):
    try:
        from PIL import Image
        with Image.open(p) as im:
            w, h = im.size
        return round(w / max(1, h), 2)
    except Exception:
        return 1.0


def candidates(toml_text):
    """[(item prefix, [picture paths, sorted by shape then name])] of the dataset TOML's blocks that can lend a
    picture: not a regularisation set, an edit's or slider's pairs, or a detail crop folder."""
    import toml
    from fizgig.dataset import detail_crops
    from fizgig.dataset.image_dataset import glob_images
    try:
        cfg = toml.loads(toml_text or "")
    except toml.TomlDecodeError:
        return []
    gen = cfg.get("general") or {}
    out = []
    for ds in cfg.get("datasets") or []:
        d = ds.get("image_directory")
        if not d or not os.path.isdir(d) or ds.get("is_reg") or ds.get("control_directory") \
                or detail_crops.is_crop_dir(d):
            continue
        pics = glob_images(d, caption_extension=ds.get("caption_extension") or gen.get("caption_extension"))
        if pics:
            out.append((ds.get("item_prefix") or "",
                        sorted(pics, key=lambda p: (_aspect(p), os.path.basename(p).lower()))))
    return out


def choose(cands, cap=4, picked=()):
    """(paths, names, how, why): the pictures to hold out. how: "picked" / "auto" / "" (none, why says so)."""
    from fizgig.training.val_pick import picked_list, resolve_picked
    flat = [(prefix + os.path.basename(p), p) for prefix, pics in cands for p in pics]
    n = len(flat)
    if picked_list(picked):
        keys = {k: p for k, p in flat}
        found, missing = resolve_picked(picked, keys)
        if found and len(found) >= n:
            return [], [], "", "you picked every picture - at least one has to train, so nothing is held out"
        if found:
            why = f"; not found: {', '.join(missing)}" if missing else ""
            return [keys[k] for k in found], found, "picked", why
    if n < MIN_POOL:
        return [], [], "", f"only {n} picture(s) - a hold-out needs at least {MIN_POOL}"
    k = min(max(1, int(cap or 1)), max(1, round(n * SHARE)), n - 1)
    sizes = [len(pics) for _p, pics in cands]
    quota = [k * s // n for s in sizes]                  # each folder its share, the rest to the largest remainders
    for i in sorted(range(len(sizes)), key=lambda i: (-(k * sizes[i] % n), i))[:k - sum(quota)]:
        quota[i] += 1
    paths, names = [], []
    for (prefix, pics), q in zip(cands, quota):
        for j in range(q):
            p = pics[int((j + 0.5) * len(pics) / q)]     # evenly through the folder's shapes
            paths.append(p)
            names.append(prefix + os.path.basename(p))
    return paths, names, "auto", ""


def from_state(state_dir):
    """The hold-out a restore point was saved with ({"paths", "names"}), or None (none, or a run from before)."""
    try:
        with open(os.path.join(str(state_dir), "training_state.json"), encoding="utf-8") as f:
            v = json.load(f).get("validation_holdout")
        return v if isinstance(v, dict) else None
    except (OSError, ValueError, AttributeError):
        return None
