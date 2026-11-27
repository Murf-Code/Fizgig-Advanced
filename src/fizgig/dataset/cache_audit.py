"""Opt-in text cache numeric audit. No changes to production tensor values."""
import json
import logging
from pathlib import Path

_path = None
_totals = {}
logger = logging.getLogger(__name__)


def active():
    return _path is not None


def enable(path):
    global _path
    _path = Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)
    with _path.open("x", encoding="utf-8"):
        pass
    _totals.clear()


def _write(row):
    with _path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def record(phase, kind, label, sd):
    if not active():
        return
    import torch
    for name, tensor in sd.items():
        if not name.endswith("hidden_states") or not tensor.is_floating_point():
            continue
        x = tensor.detach().cpu()
        mask = sd.get(name[:-len("hidden_states")] + "attention_mask")
        if mask is None or mask.ndim != 1 or x.ndim < 2 or x.shape[0] != mask.shape[0]:
            raise ValueError(f"Cache audit cannot identify real tokens: {label} / {name}")
        m = mask.detach().cpu().bool()
        valid = x[m]
        rows = valid.flatten(1)
        n = int(valid.numel())
        row = {"phase": phase, "kind": kind, "source": str(label), "tensor": name,
               "valid_tokens": int(m.sum()), "valid_values": n,
               "valid_nan": int(torch.isnan(valid).sum()), "valid_inf": int(torch.isinf(valid).sum()),
               "valid_zero": int((valid == 0).sum()), "all_zero_rows": int((rows == 0).all(dim=1).sum()),
               "total_nan": int(torch.isnan(x).sum()), "total_inf": int(torch.isinf(x).sum()),
               "empty_mask": int(n == 0), "all_zero_embedding": int(n > 0 and not bool(torch.count_nonzero(valid)))}
        _write(row)
        totals = _totals.setdefault(phase + "/" + kind, {"embeddings": 0})
        totals["embeddings"] += 1
        for key in ("valid_tokens", "valid_values", "valid_nan", "valid_inf", "valid_zero",
                    "all_zero_rows", "total_nan", "total_inf", "empty_mask", "all_zero_embedding"):
            totals[key] = totals.get(key, 0) + row[key]
        if row["valid_nan"] or row["valid_inf"] or row["all_zero_rows"] or row["empty_mask"]:
            logger.warning("[cache-audit] suspicious %s %s: %s", phase, kind, row)


def finish():
    if not active():
        return
    _write({"summary": _totals})
    logger.info("[cache-audit] %s", json.dumps(_totals))
    if not _totals.get("readback/variant", {}).get("embeddings"):
        logger.warning("[cache-audit] No shuffled variant embeddings were saved; shuffle path was not exercised.")
    logger.info("[cache-audit] report: %s", _path)
