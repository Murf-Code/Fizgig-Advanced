"""Optional mid-epoch best LoRA checkpoint selection.

Pure score bookkeeping. The trainer alone owns LoRA writes, after a completed optimizer
step or a fixed-validation pass. An unsuccessful file write MUST NOT advance the best score.

Training scores use a complete rolling window, not one unusually easy image/noise
sample. Validation scores come from repeated fixed images/timesteps. Both have a
relative improvement requirement and a minimum gap between checkpoint writes.
"""
from collections import deque
import json
import math
import os


MODES = ("off", "train", "validation")
DEFAULT_WINDOW = 32
DEFAULT_COOLDOWN = 100
DEFAULT_MIN_DELTA = 0.005  # 0.5% relative improvement


class RollingBest:
    def __init__(self, mode, *, window=DEFAULT_WINDOW, cooldown=DEFAULT_COOLDOWN,
                 min_delta=DEFAULT_MIN_DELTA):
        if mode not in MODES:
            raise ValueError(f"Unknown best-checkpoint mode: {mode}")
        self.mode = mode
        self.window = max(2, min(8192, int(window)))
        self.cooldown = max(1, min(1_000_000, int(cooldown)))
        self.min_delta = max(0.0, min(1.0, float(min_delta)))
        self.values = deque(maxlen=self.window)
        self.total = 0.0
        self.best = None
        self.step = None
        self.epoch = None
        self.seg = None

    def _candidate(self, score, step, seg=None):
        score, step = float(score), int(step)
        if not math.isfinite(score) or score < 0:
            return None
        if self.mode == "validation" and self.best is not None and seg != self.seg:
            # Different fixed images, captions or noise parameters make scores incomparable.
            self.best, self.step, self.epoch = None, None, None
        if self.best is not None:
            if score >= self.best * (1.0 - self.min_delta):
                return None
            if step - self.step < self.cooldown:
                return None
        return score

    def training(self, loss, step, *, ready=True):
        """One completed batch; return a savable moving-average loss at optimizer boundaries."""
        if self.mode != "train":
            return None
        try:
            loss = float(loss)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(loss) or loss < 0:
            return None
        if len(self.values) == self.window:
            self.total -= self.values[0]
        self.values.append(loss)
        self.total += loss
        if not ready or len(self.values) != self.window:
            return None
        return self._candidate(self.total / len(self.values), step)

    def validation(self, loss, step, *, seg=0):
        if self.mode != "validation":
            return None
        return self._candidate(loss, step, seg=seg)

    def saved(self, score, step, epoch, *, seg=None):
        """Call only after the safetensors write and atomic rename both succeed."""
        self.best = float(score)
        self.step = int(step)
        self.epoch = int(epoch)
        self.seg = seg

    def snapshot(self):
        return {"version": 1, "mode": self.mode, "score": self.best, "step": self.step,
                "epoch": self.epoch, "seg": self.seg, "window": self.window,
                "cooldown": self.cooldown, "min_delta": self.min_delta}

    def restore(self, path, *, checkpoint_exists):
        """A resume can pick up its previous best without discarding the already saved LoRA."""
        if not checkpoint_exists:
            return False
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("version") != 1 or d.get("mode") != self.mode:
                return False
            if (self.mode == "train" and
                    (int(d.get("window")) != self.window or int(d.get("cooldown")) != self.cooldown)):
                return False
            score, step, epoch = float(d["score"]), int(d["step"]), int(d["epoch"])
            if not math.isfinite(score) or score < 0 or step < 0 or epoch < 0:
                return False
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            return False
        self.saved(score, step, epoch, seg=d.get("seg"))
        return True


def save_snapshot(path, data):
    """Write metadata only after the checkpoint is safely installed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
