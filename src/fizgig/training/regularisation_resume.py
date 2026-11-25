"""Resume compatibility for capped LoRA regularisation buckets."""
from collections import Counter
import hashlib
import json
import random

MODES = ("fixed-v1", "fractional-v1", "fractional-v2")
DEFAULT = MODES[-1]


def capped(group, per_epoch):
    return [ds for ds in getattr(group, "datasets", []) if getattr(ds, "is_reg", False)
            and getattr(ds, "batch_manager", None) is not None
            and 0 < int(per_epoch or 0) < sum(len(b) for b in ds.batch_manager.buckets.values())]


def signature(group, per_epoch):
    """Content-independent identity of the regularisation sampling plan."""
    rows = []
    for ds in capped(group, per_epoch):
        bm = ds.batch_manager
        rows.append([int(per_epoch), bm.batch_size,
                     [[list(r), sorted(str(it.item_key) for it in bm.buckets[r])]
                      for r in bm.bucket_resos]])
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def restore_seeds(group, seeds):
    datasets = list(getattr(group, "datasets", []))
    if isinstance(seeds, list) and len(seeds) == len(datasets) and all(type(s) is int for s in seeds):
        for ds, seed in zip(datasets, seeds):
            ds.seed = seed
        return True
    return False


def snapshot(group, mode, per_epoch, low_group=None):
    return {"mode": mode, "signature": signature(group, per_epoch),
            "dataset_seeds": [getattr(ds, "seed", None) for ds in group.datasets],
            "low_dataset_seeds": [getattr(ds, "seed", None) for ds in low_group.datasets] if low_group else []}


def epoch_keys(group, per_epoch, mode, epoch):
    """Replay only the capped regularisation managers, without loading any tensors or touching the global RNG."""
    from fizgig.dataset.image_dataset import BucketBatchManager
    keys = Counter()
    for ds in capped(group, per_epoch):
        original = ds.batch_manager
        bm = BucketBatchManager({r: list(b) for r, b in original.buckets.items()},
                                original.batch_size, original.num_timestep_buckets)
        bm.limit(int(per_epoch), quota_mode=mode)
        for e in range(1, epoch + 1):
            bm.shuffle(random.Random(ds.seed + e))
        for r, i in bm.bucket_batch_indices:
            start = i * bm.bucket_batch(r)
            end = min(start + bm.bucket_batch(r), bm.quota[r])
            keys[tuple(str(it.item_key) for it in bm.buckets[r][start:end])] += 1
    return keys


def prepare(group, per_epoch, meta, batch_size, *, staged=False, log=lambda msg: None):
    """Choose the checkpoint's schedule before limit() mutates the initial buckets.

    Old mid-epoch states predate versioned quotas; distinguish the original fixed
    schedule from the briefly shipped fractional-v1 using the recorded batch keys.
    Never silently proceed with known mismatches in the capped regularisation set.
    """
    datasets = capped(group, per_epoch)
    saved = meta.get("regularisation_schedule") or {}
    mid = meta.get("mid_epoch") or {}
    same_batch = int(meta.get("batch_size", 1)) == int(batch_size)
    if mid and same_batch and saved.get("signature") and saved["signature"] != signature(group, per_epoch):
        raise RuntimeError("[resume] regularisation buckets or per-epoch limit changed since Save & stop; "
                           "restore the original settings or resume from an epoch-boundary checkpoint")
    if not datasets:
        return DEFAULT
    mode = saved.get("mode")
    if mode is not None and mode not in MODES:
        raise RuntimeError(f"[resume] unknown regularisation schedule {mode!r}")
    if mid and same_batch:
        if not restore_seeds(group, mid.get("dataset_seeds") or saved.get("dataset_seeds")):
            raise RuntimeError("[resume] capped regularisation needs the saved dataset seeds for a mid-epoch restore; "
                               "use an epoch-boundary checkpoint or restore the original dataset layout")
        if not staged:
            reg_keys = {str(it.item_key) for ds in datasets for b in ds.batch_manager.buckets.values() for it in b}
            done = Counter(tuple(str(k) for k in ks) for ks, _ in mid.get("batches") or []
                           if any(str(k) in reg_keys for k in ks))
            epoch = int(mid.get("epoch") or (int(meta.get("epoch", 0)) + 1))
            candidates = (mode,) if mode else ("fixed-v1", "fractional-v1")
            matches = [m for m in candidates if not (done - epoch_keys(group, per_epoch, m, epoch))]
            if not matches:
                raise RuntimeError("[resume] saved regularisation batches do not match this epoch; "
                                   "restore the original dataset or use an epoch-boundary checkpoint")
            if not mode:
                mode = matches[0]
                if len(matches) > 1:
                    log("[resume] unversioned regularisation checkpoint: using original fixed quota schedule "
                        "(both old schedules match the saved regularisation batches)")
        elif not mode:
            mode = "fixed-v1"
            log("[resume] unversioned staged regularisation: preserving the original fixed quota schedule")
    elif saved:
        if not restore_seeds(group, saved.get("dataset_seeds")):
            log("[resume] dataset layout changed; regularisation shuffle seeds could not be restored")
    mode = mode or DEFAULT
    if mid or saved:
        log(f"[resume] regularisation quota schedule: {mode}")
    return mode
