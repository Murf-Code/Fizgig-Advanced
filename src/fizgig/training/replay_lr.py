"""Provisional AdamW-reference LR conversion for the isolated 7G.1 test.

Not a universal calibration. No production optimizer defaults are changed.
"""
from decimal import Decimal, InvalidOperation

SCALES = {"adamw": Decimal("1"), "adamw8bit": Decimal("1"),
          "pagedadamw8bit": Decimal("1"), "muon": Decimal("1.8")}


def reference_rate(optimizer, requested):
    """Return (reference LR, native group LR, factor), with exact decimal scaling."""
    key = str(optimizer).strip().lower()
    if key not in SCALES:
        raise ValueError("No experimental AdamW-reference scale for " + key)
    try:
        rate = Decimal(str(requested))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("Invalid reference learning rate") from exc
    if not rate.is_finite() or rate <= 0:
        raise ValueError("Invalid reference learning rate")
    native = rate * SCALES[key]
    import math
    result = (float(rate), float(native), float(SCALES[key]))
    if not all(math.isfinite(x) and x > 0 for x in result):
        raise ValueError("Reference learning rate outside representable range")
    return result
