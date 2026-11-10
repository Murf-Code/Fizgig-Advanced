"""Slow-step warning: a training step far slower than this run's own validation passes say it should be.

A validation pass is one forward over the same base, block swap and picture size as a training step, so it measures
what this card does when everything fits. A training step is about three of them (the forward, the checkpointed
blocks' recompute, the backward). When steady steps take many times that, the usual cause on Windows is that the
step ran out of VRAM and the driver moved part of it into shared GPU memory (system RAM over PCIe) - measured on a
12 GB RTX 4070 Ti, Krea 2 at 0.25 MP, Nov 2026: 1.07 s a validation pass, 3.9 s a step at swap 20, 22-66 s at swap
17. One plain warning per run, never a stop.
"""
from statistics import median

BACKWARD_FACTOR = 3.0      # a training step against one forward pass, when it fits
SLOW_FACTOR = 4.0          # warn past this many times that
SKIP_STEPS = 3             # the first clean steps: kernels planning, caches filling
WINDOW = 5                 # steady steps the median is taken over


class SlowStepWatch:
    def __init__(self, blocks_to_swap=0, max_swap=0):
        self.pass_s = None
        self.swap = int(blocks_to_swap or 0)
        self.max_swap = int(max_swap or 0)
        self.seen = 0
        self.dts = []
        self.told = False

    def set_pass(self, seconds, passes):
        """A validation check's time and its number of forward passes (images x noise levels)."""
        if passes and seconds and seconds > 0:
            self.pass_s = float(seconds) / int(passes)

    def observe(self, step_s):
        """A clean training step's seconds -> the warning text, once, or None."""
        self.seen += 1
        if self.told or self.pass_s is None or self.seen <= SKIP_STEPS:
            return None
        self.dts.append(float(step_s))
        if len(self.dts) < WINDOW:
            return None
        now = median(self.dts[-WINDOW:])
        expected = self.pass_s * BACKWARD_FACTOR
        if now <= SLOW_FACTOR * expected:
            return None
        self.told = True
        more = min(self.swap + 3, self.max_swap) if self.max_swap else self.swap + 3
        swap = (f"raise Blocks Swap (this run swaps {self.swap}; try {more})" if more > self.swap
                else "lower the resolution or turn off EMA or a context LoRA (Blocks Swap is already at its most)")
        return (f"[vram] Training steps take {now:.1f} s, about {now / expected:.0f}x longer than this card's "
                f"validation passes ({self.pass_s:.2f} s each) say a step should take (about {expected:.1f} s). That "
                f"usually means training ran out of VRAM and Windows moved part of it into shared GPU memory. Check "
                f"Task Manager > Performance > GPU: if Shared GPU memory is above a few hundred MB while it trains, "
                f"stop and {swap}.")


def warmup_line(now_s, start_s):
    """The warm-up note's progress line: 'Still speeding up' only when the steps really got faster."""
    if now_s < 0.9 * start_s:
        return f"[warm-up] Still speeding up: {now_s:.2f} s/step now, from {start_s:.1f} s at the start."
    return f"[warm-up] Not faster yet: {now_s:.2f} s/step now, {start_s:.1f} s at the start."
