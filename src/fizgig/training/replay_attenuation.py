"""7C/7D development-only live attenuation; actual Muon proposal, joint BA movement."""
import math
import time

import numpy as np
import torch

from .muon import BatchedMuon, Muon
from .replay_forks import Fork, MovementCurve

MODES = ("full", "rotate", "uniform", "compensated")
RTOL = .001
CAP = 1.5


def rotation(blocks, step):
    """Complementary parity groups, not a learned importance ranking."""
    if step < 1 or not blocks or len(blocks) % 2:
        raise ValueError("rotation needs an even block count and positive step")
    return {b: .5 if i % 2 == (step-1) % 2 else 1.
            for i,b in enumerate(sorted(blocks))}


def solve(cores, weights, target, cap):
    # MovementCurve's domain is [0,1]. Scale its linear/quadratic terms
    # to parameterize c in [0,cap] without changing the existing solver.
    curve = MovementCurve([(b,C,cap*L,cap*cap*Q) for b,C,L,Q in cores], weights)
    lo, hi = curve.increasing_interval(max_initial_dip=.01)
    low, high = curve.norm(lo), curve.norm(hi)
    tol = max(1e-12, RTOL*target)
    if target < low-tol or target > high+tol:
        raise RuntimeError("joint effective movement target unreachable within multiplier cap")
    clipped = min(high,max(low,target))
    return cap*curve.solve(clipped,hi,lo)


@torch.no_grad()
def allocated_step(pairs, optimizer, mode, step):
    """Advance Muon momentum exactly once, then commit the chosen native add.
    No model forward, extra gradient, optimizer replay or independent block scoring.
    Full proposal and actual endpoints use the same pre-step factors and decay.
    Diagnostic copies are CPU-only and released at return. Failure aborts training
    before scheduler/EMA/save; this research path is not a production fast path.
    """
    if mode not in MODES or type(optimizer) not in (BatchedMuon, Muon):
        raise ValueError("unsupported research allocation or optimizer")
    pairs = tuple(pairs)
    started = time.perf_counter()
    fork = Fork(pairs, optimizer, None, compact=True)
    blocks = set(fork.block.values())
    weights = rotation(blocks,step)
    full = dict.fromkeys(blocks,1.)
    try:
        optimizer.step(proposal_sink=fork.observe)
        if set(fork.direction) != set(fork.parent):
            raise RuntimeError("incomplete Muon proposal")
        native, native_shares = fork.measured()
        if not math.isfinite(native) or native <= 1e-12:
            raise RuntimeError("noninformative native movement")
        # Compare both controls to the same *local* parent and native proposal,
        # not to another run's diverged parameter trajectory.
        c = 1.
        if mode == "full":
            actual, shares = native, native_shares
            multipliers, target = full, native
        elif mode == "rotate":
            multipliers = weights
            actual, shares = fork.achieved(multipliers)
            target = actual
        else:
            cores = fork.cores()
            if mode == "uniform":
                target, _ = fork.achieved(weights)
                relative, cap = full, 1.
            else:
                target, relative, cap = native, weights, CAP
            if not math.isfinite(target) or target <= 1e-12:
                raise RuntimeError("noninformative allocation target")
            c = solve(cores, relative, target, cap)
            multipliers = {b:c*q for b,q in relative.items()}
            actual, shares = fork.achieved(multipliers,max_multiplier=cap)
            # Rounding may differ from the analytic FP64 curve. Accept only the
            # measured committed endpoint, not the predicted norm.
            if abs(actual-target) > max(1e-12,RTOL*target):
                raise RuntimeError("committed joint movement outside tolerance")
        if not math.isfinite(actual) or actual <= 1e-12:
            raise RuntimeError("nonfinite or zero committed movement")
        return {"mode":mode, "step":step, "native_movement":native,
                "target_movement":target, "actual_movement":actual,
                "relative_gap":abs(actual-target)/target,
                "movement_ratio":actual/native, "common_scale":c,
                "multipliers":multipliers, "block_squared_movement":shares,
                "native_block_squared_movement":native_shares,
                "attenuated_blocks":[b for b,q in sorted(weights.items()) if q < 1.],
                "seconds":time.perf_counter()-started,
                "momentum_policy":"preserve", "gradient_blocks":sorted(blocks)}
    finally:
        fork.live = False
