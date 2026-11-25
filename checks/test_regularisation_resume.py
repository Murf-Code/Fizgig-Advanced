"""Capped regularisation schedules survive epoch and Save & stop restore points, including prefetch."""
from collections import Counter
import json
import random
from types import SimpleNamespace

import pytest
import torch

from fizgig.dataset.image_dataset import BucketBatchManager, DatasetGroup, ImageDataset
from fizgig.training import regularisation_resume as reg
from fizgig.training.prefetch import Prefetch


def _dataset(rows, bs, seed, is_reg, shared, reverse=False):
    ds = object.__new__(ImageDataset)
    ds.is_reg = is_reg
    ds.seed = seed
    ds.current_epoch = 0
    ds.shared_epoch = shared
    items = [(r, [SimpleNamespace(item_key=f"{'reg:' if is_reg else 'subject:'}{r[0]}_{i}")
                  for i in range(n)]) for r, n in rows]
    if reverse:
        items = [(r, list(reversed(b))) for r, b in reversed(items)]
    ds.batch_manager = BucketBatchManager(dict(items), bs)
    ds.num_train_items = sum(n for _, n in rows)
    return ds


def _group(bs, seed=17, reverse=False, rows=(((512, 512), 9), ((448, 576), 2), ((576, 448), 2))):
    shared = SimpleNamespace(value=0)
    subject = _dataset([((768, 768), 12)], bs, seed, False, shared)
    regular = _dataset(rows, bs, seed + 1, True, shared, reverse)
    return DatasetGroup([subject, regular]), shared


def _limit(group, cap, mode):
    group.datasets[1].batch_manager.limit(cap, quota_mode=mode)
    group.cumulative_sizes = group.cumsum(group.datasets)


def _batches(group, shared, epoch, enabled, order_seed):
    shared.value = epoch
    loader = torch.utils.data.DataLoader(group, batch_size=1, shuffle=True, num_workers=0,
                                        generator=torch.Generator().manual_seed(order_seed),
                                        collate_fn=lambda b: b[0])
    return list(Prefetch(loader, enabled=enabled))


@pytest.fixture(autouse=True)
def no_tensor_reads(monkeypatch):
    def keys(bm, idx):
        r, i = bm.bucket_batch_indices[idx]
        start = i * bm.bucket_batch(r)
        end = min(start + bm.bucket_batch(r), len(bm.buckets[r]) if bm.quota is None else bm.quota[r])
        return tuple(it.item_key for it in bm.buckets[r][start:end])
    monkeypatch.setattr(BucketBatchManager, "__getitem__", keys)


@pytest.mark.parametrize("bs,cap", [(1, 5), (4, 5)])
@pytest.mark.parametrize("enabled", [False, True])
def test_pause_resume_reconstructs_same_epoch_and_step_count(bs, cap, enabled):
    group, shared = _group(bs)
    mode = reg.prepare(group, cap, {}, bs)
    _limit(group, cap, mode)
    n = len(group)
    for epoch in (1, 2, 3):
        _batches(group, shared, epoch, enabled, epoch)
    current = _batches(group, shared, 4, enabled, 44)
    done = current[:len(current) // 2]
    checkpoint = json.loads(json.dumps({
        "epoch": 3, "batch_size": bs, "global_step": 3 * n + len(done),
        "regularisation_schedule": reg.snapshot(group, mode, cap),
        "mid_epoch": {"epoch": 4, "steps_per_epoch": n, "steps": len(done),
                      "batches": [[list(b), 0.5] for b in done],
                      "dataset_seeds": [ds.seed for ds in group.datasets]}}))
    restarted, rs = _group(bs, seed=999, reverse=True)
    resumed_mode = reg.prepare(restarted, cap, checkpoint, bs)
    _limit(restarted, cap, resumed_mode)
    actual = _batches(restarted, rs, 4, not enabled, 999)
    assert resumed_mode == mode and len(restarted) == len(actual) == n
    assert Counter(actual) == Counter(current)
    pending = Counter(done)
    trained = []
    for batch in actual:
        if pending[batch]:
            pending[batch] -= 1
        else:
            trained.append(batch)
    assert not +pending
    assert Counter(done + trained) == Counter(current)
    assert checkpoint["global_step"] + len(trained) == 4 * n
    expected_next = _batches(group, shared, 5, enabled, 45)
    next_state = {"epoch": 4, "batch_size": bs,
                  "regularisation_schedule": reg.snapshot(group, mode, cap)}
    boundary, bs_shared = _group(bs, seed=555, reverse=True)
    _limit(boundary, cap, reg.prepare(boundary, cap, next_state, bs))
    assert Counter(_batches(boundary, bs_shared, 5, not enabled, 123)) == Counter(expected_next)


def test_old_mid_epoch_checkpoint_preserves_fixed_schedule():
    group, shared = _group(1)
    _limit(group, 5, "fixed-v1")
    current = _batches(group, shared, 7, False, 17)
    meta = {"epoch": 6, "mid_epoch": {"epoch": 7, "dataset_seeds": [ds.seed for ds in group.datasets],
                                    "batches": [[list(b), 0.0] for b in current]}}
    restarted, rs = _group(1, seed=999)
    mode = reg.prepare(restarted, 5, meta, 1)
    assert mode == "fixed-v1"
    _limit(restarted, 5, mode)
    assert Counter(_batches(restarted, rs, 7, True, 999)) == Counter(current)


def test_unversioned_fractional_checkpoint_is_recognised():
    rows = (((512, 512), 1), ((448, 576), 1), ((576, 448), 1))
    group, _ = _group(1, rows=rows)
    for epoch in range(1, 50):
        fixed = reg.epoch_keys(group, 1, "fixed-v1", epoch)
        fractional = reg.epoch_keys(group, 1, "fractional-v1", epoch)
        if fixed != fractional:
            break
    else:
        pytest.fail("fractional schedule never differed from fixed schedule")
    meta = {"epoch": epoch - 1, "mid_epoch": {"epoch": epoch, "dataset_seeds": [ds.seed for ds in group.datasets],
                                           "batches": [[list(k), 0.5] for k in fractional]}}
    assert reg.prepare(group, 1, meta, 1) == "fractional-v1"


def test_changed_cap_rejects_mid_epoch_resume():
    group, _ = _group(1)
    _limit(group, 5, reg.DEFAULT)
    meta = {"regularisation_schedule": reg.snapshot(group, reg.DEFAULT, 5),
            "mid_epoch": {"epoch": 1, "dataset_seeds": [ds.seed for ds in group.datasets], "batches": []}}
    fresh, _ = _group(1)
    for changed in (4, 0):
        with pytest.raises(RuntimeError, match="regularisation buckets or per-epoch limit changed"):
            reg.prepare(fresh, changed, meta, 1)


def test_large_batches_never_pad_or_drop_regularisation_images():
    group, shared = _group(4)
    _limit(group, 5, reg.DEFAULT)
    n = len(group)
    for epoch in range(1, 41):
        batches = _batches(group, shared, epoch, False, epoch)
        keys = [key for b in batches for key in b if key.startswith("reg:")]
        assert len(batches) == n
        assert len(keys) == len(set(keys)) == 5
