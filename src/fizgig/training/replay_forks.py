"""Research-only one-step Muon forks. No trainer or queue integration.

CPU snapshots, native-device commits, compact FP64 effective-weight geometry.
The context restores the parent on exit; it never advances EMA or the selector.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import copy
import math
import random

import numpy as np
import torch


def _cpu(t):
    return t.detach().to('cpu', copy=True)


def _clone_cpu(x):
    if isinstance(x, torch.Tensor):
        return _cpu(x)
    if isinstance(x, dict):
        return {k: _clone_cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clone_cpu(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_clone_cpu(v) for v in x)
    return copy.deepcopy(x)


class RNGSnapshot:
    """Explicit generators are not covered by torch.get_rng_state()."""
    def __init__(self, generators=()):
        self.python = random.getstate()
        self.numpy = copy.deepcopy(np.random.get_state())
        self.cpu = torch.get_rng_state().clone()
        self.cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        self.generators = [(g, g.get_state().clone()) for g in generators]

    def restore(self):
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state_all(self.cuda)
        for g, state in self.generators:
            g.set_state(state)


@dataclass(frozen=True)
class Pair:
    name: str
    block: int
    A: torch.Tensor
    B: torch.Tensor
    scale: float


def network_pairs(net):
    """Standard Krea 2 LoRA only; do not infer scale from rank."""
    from fizgig.families.lora import TRAINABLE
    out = []
    for name, wrapped in sorted(net.wrapped.items()):
        if TRAINABLE not in wrapped.adapters:
            continue
        a, b = wrapped.adapters[TRAINABLE]
        out.append(Pair(name, int(name.split('.')[1]), a.weight, b.weight,
                        float(wrapped.scales[TRAINABLE])))
    return out


def _bits_equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


class Fork:
    """Common native proposal with the parent's weights and post-step optimizer buffers live.

    Only use inside capture(). Scoring must be forward-only: no optimizer,
    gradient, EMA, routing or model-state mutations. No concurrent training.
    """
    def __init__(self, pairs, optimizer, rng):
        self.pairs = tuple(pairs)
        self.optimizer = optimizer
        self.rng = rng
        ps = [p for pair in self.pairs for p in (pair.A, pair.B)]
        opt_ps = [p for g in optimizer.param_groups for p in g['params']]
        if not ps or len({id(p) for p in ps}) != len(ps) or set(ps) != set(opt_ps) or len(opt_ps) != len(ps):
            raise ValueError('one-step fork requires exactly the paired LoRA optimizer parameters')
        if len({pair.name for pair in self.pairs}) != len(self.pairs):
            raise ValueError('duplicate pair names')
        for pair in self.pairs:
            if (pair.A.ndim != 2 or pair.B.ndim != 2 or pair.B.shape[1] != pair.A.shape[0]
                    or not math.isfinite(pair.scale) or pair.A.grad is None or pair.B.grad is None
                    or not pair.A.requires_grad or not pair.B.requires_grad):
                raise ValueError('full-probe, finite-scale, standard LoRA gradients required')
        self.parent = {p: _cpu(p) for p in ps}
        self.block = {p: pair.block for pair in self.pairs for p in (pair.A, pair.B)}
        self.decayed = {}
        self.direction = {}
        self.alpha = {}
        self.groups = []
        self.live = True

    def observe(self, ps, updates, alpha, foreach):
        ps, updates = list(ps), list(updates)
        if len(ps) != len(updates) or any(p not in self.parent or p in self.direction for p in ps):
            raise RuntimeError('invalid native proposal capture')
        if not math.isfinite(alpha):
            raise RuntimeError('nonfinite native step scale')
        self.groups.append((ps, float(alpha), foreach))
        for p, update in zip(ps, updates):
            self.decayed[p] = _cpu(p)
            self.direction[p] = _cpu(update)
            self.alpha[p] = float(alpha)

    @torch.no_grad()
    def reset_weights(self):
        for p, old in self.parent.items():
            p.copy_(old)

    @torch.no_grad()
    def apply(self, multipliers):
        if not self.live or set(multipliers) != set(self.block.values()):
            raise ValueError('fork closed or missing/extraneous block multiplier')
        if any(not math.isfinite(m) or not 0 <= m <= 1 for m in multipliers.values()):
            raise ValueError('multipliers must be finite and in [0, 1]')
        # Preserve native add/alpha and decay rounding, rather than subtracting
        # the rounded native endpoint to reconstruct a learning delta.
        for ps, alpha, foreach in self.groups:
            bins = {}
            for p in ps:
                p.copy_(self.decayed[p])
                bins.setdefault(float(multipliers[self.block[p]]), []).append(p)
            for m, params in bins.items():
                updates = [self.direction[p].to(device=p.device, dtype=p.dtype) for p in params]
                if foreach:
                    torch._foreach_add_(params, updates, alpha=alpha*m)
                else:
                    for p, u in zip(params, updates):
                        p.add_(u, alpha=alpha*m)

    def cores(self):
        out = []
        for pair in self.pairs:
            A, B = self.parent[pair.A].double(), self.parent[pair.B].double()
            dA = self.decayed[pair.A].double() - A
            dB = self.decayed[pair.B].double() - B
            a = self.direction[pair.A].double() * self.alpha[pair.A]
            b = self.direction[pair.B].double() * self.alpha[pair.B]
            # QR compresses column/row spaces; only small cores are multiplied.
            # Express decay, linear and quadratic terms without subtracting BA.
            RB = torch.linalg.qr(torch.cat((B, dB, b), dim=1), mode='r')[1]
            RA = torch.linalg.qr(torch.cat((A, dA, a), dim=0).T, mode='r')[1].T
            B0, db, bb = RB.split(A.shape[0], dim=1)
            A0, da, aa = RA.split(A.shape[0], dim=0)
            C = B0 @ da + db @ (A0 + da)
            L = (B0 + db) @ aa + bb @ (A0 + da)
            Q = bb @ aa
            out.append((pair.block, *(x.numpy()*pair.scale for x in (C, L, Q))))
        return out

    def achieved(self, multipliers):
        self.apply(multipliers)
        shares = {b: 0.0 for b in multipliers}
        for pair in self.pairs:
            A, B = self.parent[pair.A].double(), self.parent[pair.B].double()
            da, db = _cpu(pair.A).double()-A, _cpu(pair.B).double()-B
            R = torch.linalg.qr(torch.cat((B, db), dim=1), mode='r')[1]
            S = torch.linalg.qr(torch.cat((da, A+da), dim=0).T, mode='r')[1]
            norm2 = float((R @ S.T).square().sum()) * pair.scale**2
            shares[pair.block] += norm2
        return math.sqrt(sum(shares.values())), shares


@contextmanager
def capture(pairs, optimizer, *, generators=()):
    """Native step exactly once, audit replay parity, roll back on every exit.

    Optimizer state is common to all candidate endpoints. This is NOT disk
    resume parity or a full multi-step trainer fork. Use only after clipping.
    """
    from fizgig.training.muon import BatchedMuon, Muon
    if type(optimizer) not in (BatchedMuon, Muon):
        raise TypeError('only Fizgig native Muon implementations are supported')
    rng = RNGSnapshot(generators)
    fork = Fork(pairs, optimizer, rng)
    incoming = _clone_cpu(optimizer.state_dict())
    try:
        optimizer.step(proposal_sink=fork.observe)
        if set(fork.direction) != set(fork.parent):
            raise RuntimeError('incomplete full-block native proposal')
        native = {p: _cpu(p) for p in fork.parent}
        fork.apply({b: 1.0 for b in set(fork.block.values())})
        if any(not _bits_equal(_cpu(p), old) for p, old in native.items()):
            raise RuntimeError('native commit replay is not bitwise identical')
        del native
        fork.reset_weights()
        rng.restore()
        yield fork
    finally:
        fork.live = False
        try:
            fork.reset_weights()
            optimizer.state.clear()
            optimizer.load_state_dict(incoming)
        finally:
            rng.restore()


class MovementCurve:
    """D(c) = C + c*q*L + c²*q²*Q; evaluate norms via compact cores."""
    def __init__(self, cores, weights):
        if set(weights) != {b for b, *_ in cores} or any(not 0 < q <= 1 for q in weights.values()):
            raise ValueError('relative weights must cover every block, in (0, 1]')
        self.parts = [(C, weights[b]*L, weights[b]**2*Q) for b,C,L,Q in cores]
        self.coeff = np.zeros(5, dtype=np.float64)
        for C,L,Q in self.parts:
            self.coeff += [np.vdot(C,C), 2*np.vdot(C,L), np.vdot(L,L)+2*np.vdot(C,Q),
                           2*np.vdot(L,Q), np.vdot(Q,Q)]
        if not np.isfinite(self.coeff).all():
            raise ValueError('nonfinite movement curve')

    def norm(self, c):
        return math.sqrt(sum(float(np.square(C+c*L+c*c*Q).sum()) for C,L,Q in self.parts))

    def increasing_interval(self, max_initial_dip=0.):
        """First increasing interval, optionally after a bounded initial dip in c.

        The caller must still require a target above norm(0).
        max_initial_dip is a multiplier bound, not a displacement tolerance.
        """
        if not math.isfinite(max_initial_dip) or not 0 <= max_initial_dip <= .01:
            raise ValueError('initial dip bound must be in [0, 0.01]')
        d = self.coeff[1:] * np.arange(1,5)
        scale = np.max(np.abs(d))
        if scale == 0:
            raise ValueError('flat movement curve')
        d = d/scale
        roots = np.polynomial.polynomial.polyroots(np.trim_zeros(d, 'b'))
        cuts = sorted({0.,1.,*(float(r.real) for r in roots if abs(r.imag) <= 1e-9 and 0 < r.real < 1)})
        start = None
        for lo,hi in zip(cuts,cuts[1:]):
            sign = np.polynomial.polynomial.polyval((lo+hi)/2,d)
            if sign > 0:
                if start is None:
                    if lo > max_initial_dip:
                        raise ValueError('movement initially decreases beyond permitted multiplier')
                    start = lo
            elif start is not None:
                return start, lo
            elif max_initial_dip == 0.:
                raise ValueError('movement initially decreases or is unresolved')
        if start is None:
            raise ValueError('no increasing movement interval')
        return start, 1.

    def first_increasing_end(self):
        return self.increasing_interval()[1]

    def solve(self, target, end, start=0.):
        lo,hi = start,end
        if not self.norm(lo) <= target <= self.norm(hi):
            raise ValueError('unreachable movement budget')
        for _ in range(60):
            mid = (lo+hi)/2
            if self.norm(mid) < target:
                lo = mid
            else:
                hi = mid
        return (lo+hi)/2


def match(fork, allocations, *, minimum, atol, rtol, max_initial_dip=0.):
    """Lock candidates before evaluation; return explicit unmatched reason.

    Bounded precision repair searches movement only, never evaluation losses.
    Callers must report all scheduled forks, including rejected ones.
    """
    if len(allocations) < 2 or any(not math.isfinite(x) or x < 0 for x in (minimum,atol,rtol)):
        raise ValueError('need >=2 allocations and explicit finite nonnegative tolerances')
    try:
        cores = fork.cores()
        curves = {n: MovementCurve(cores,w) for n,w in allocations.items()}
        intervals = {n: f.increasing_interval(max_initial_dip) for n,f in curves.items()}
        ends = {n: interval[1] for n,interval in intervals.items()}
        upper = {n: f.norm(ends[n]) for n,f in curves.items()}
        lower = max(f.norm(0) for f in curves.values())
        limiter = min(upper,key=upper.get)
        target = upper[limiter]
        if target <= max(lower,minimum):
            return {'matched':False,'reason':'no informative common budget','target':target,'limiter':limiter}
        rows = {}
        def measure(n,c):
            multipliers = {b: c*q for b,q in allocations[n].items()}
            actual, shares = fork.achieved(multipliers)
            if not math.isfinite(actual):
                raise ValueError('nonfinite committed movement')
            return {'scale':c,'multipliers':multipliers,'achieved':actual,'block_squared_movement':shares}
        for n,f in curves.items():
            rows[n] = measure(n,f.solve(target,ends[n],intervals[n][0]))
        def gap():
            vals = [r['achieved'] for r in rows.values()]
            return max(vals)-min(vals)
        tolerance = atol+rtol*target
        if gap() > tolerance:
            for n,row in list(rows.items()):
                c = row['scale']
                for offset in (-1e-6,1e-6,-1e-5,1e-5,-1e-4,1e-4,-1e-3,1e-3):
                    candidate = measure(n,min(ends[n],max(intervals[n][0],c+offset)))
                    if abs(candidate['achieved']-target) < abs(rows[n]['achieved']-target):
                        rows[n] = candidate
        ok = gap() <= tolerance and min(r['achieved'] for r in rows.values()) > minimum
        return {'matched':ok,'reason':None if ok else 'committed movement mismatch or below floor',
                'target':target,'limiter':limiter,'gap':gap(),'tolerance':tolerance,'arms':rows,
                'increasing_intervals':intervals,'max_initial_dip':max_initial_dip}
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {'matched':False,'reason':str(exc)}
    finally:
        fork.reset_weights()
        fork.rng.restore()
