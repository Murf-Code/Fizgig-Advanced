"""Frozen 7E/7F/7G benchmark helpers; no per-update geometry."""
import hashlib
import math

import torch

MODES = ("full", "existing", "balanced", "compensated")
BASE_LR = .00045


def active_blocks(n, update, start=0):
    if n != 28 or update < 1 or start not in (0,1):
        raise ValueError("final schedule requires 28 blocks and a valid global update")
    return set(range((update-1+start)%2,n,2))


def multiplier(mode, active, total=28):
    if mode not in MODES or not 1 <= active <= total:
        raise ValueError("invalid final allocation")
    return math.sqrt(total/active) if mode == "compensated" else 1.


def gradient_counts(pairs):
    result = {}
    for pair in pairs:
        row = result.setdefault(pair.block, [0,0])
        for p in (pair.A,pair.B):
            row[0] += int(p.grad is not None)
            row[1] += 1
    return result


def checked_counts(pairs, active):
    counts = gradient_counts(pairs)
    if set(counts) != set(range(28)) or any(
            total != 16 or count != (total if b in active else 0)
            for b,(count,total) in counts.items()):
        raise RuntimeError("final benchmark factor-gradient mask mismatch")
    return counts


def digest_tensor(p):
    cpu = p.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(cpu.view(torch.uint8).numpy()).cast("B")).hexdigest()


class FirstUpdateAudit:
    """First update only: compare native directions and momentum across C/D."""
    def __init__(self, pairs, optimizer, scale):
        self.names = {p: pair.name+"."+suffix for pair in pairs
                      for suffix,p in (("A",pair.A),("B",pair.B))}
        self.opt, self.scale = optimizer, scale
        self.directions, self.alphas = {}, {}

    def observe(self, ps, updates, alpha, foreach):
        for p,u in zip(ps,updates):
            key = self.names[p]
            if key in self.directions:
                raise RuntimeError("duplicate proposal")
            self.directions[key] = digest_tensor(u)
            self.alphas[key] = float(alpha)

    def result(self):
        active = {self.names[p] for p in self.names if p.grad is not None}
        if set(self.directions) != active:
            raise RuntimeError("incomplete first-step proposal audit")
        momentum = {self.names[p]:digest_tensor(self.opt.state[p]["momentum_buffer"])
                    for p in self.names if p.grad is not None}
        return {"directions":self.directions,"applied_alphas":self.alphas,"momentum":momentum,
                "scale":self.scale}
