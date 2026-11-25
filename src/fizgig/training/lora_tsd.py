"""LoRA-TSD - tangent-space spectral descent for LoRA pairs.

Adapted from https://github.com/brain-lab-research/LoRA-TSD (src/optimizers/lora_tsd.py, commit e6ef0e16; a fork
of dimativator/LoRA-TSD), MIT License, Copyright (c) 2026 Dmitrii Andriianov - the licence is in
docs/THIRD_PARTY_NOTICES.md.

Each LoRA's down (A, r x n) and up (B, m x r) train together: the factors' gradients rebuild the weight gradient's
projection onto the tangent space of rank-r matrices at W = B A, a few rounds of matrix-sign (Newton-Schulz on a
small QR core) and re-projection turn it into a steepest-descent step under the spectral norm, and that step is
split back into dA and dB. The scaled, stored adapter change is checked against max_delta_norm (Frobenius) per pair per step.

Changes from the reference: trimmed to the update itself (its diagnostics, logging and lr_A / lr_B schedulers
removed); one learning rate per pair in the group's "lr" (the reference's lr_A), so Fizgig's scheduler and
Adaptive LR move it as for any optimizer; the reference config's defaults (momentum 0.95, 5 ball iterations,
5 Newton-Schulz steps, max_delta_norm 0.1, norm balancing on); and a start for a zero factor (Fizgig's addition):
a LoRA's up starts at zero, where the tangent space is undefined, so until both factors are non-zero the zero one
steps alone by a rank-controlled polar update. A small-adapter warm phase avoids an early singular factor split.

magnitude_scale (Fizgig's addition, as PoLoRA's): multiplies every step and its cap alike, so a bigger magnitude
isn't swallowed by max_delta_norm. 1.0 is the reference's step. Adaptive LR and per-image LR move the group's "lr",
so the step is lr x magnitude_scale and the effective size is magnitude_scale x image x adaptive, as with PoLoRA.
Every log_every steps one log line gives the magnitude, the average weight change a pair and how many hit the cap.

Experimental tangent controls adapted from Grok Round 10 (credit: independent audit and CPU tests).
The warm branch retains the group LR; tangent LR scale, per-pair decay, optional rank-truncated SVD polar,
and factor-space signal budget act only in the tangent branch. Defaults preserve the old update.
The budget is not gauge invariant; use unbalanced matched arms for the initial experiment.
"""

from __future__ import annotations

import logging
import math

import torch
from torch.optim import Optimizer



@torch.no_grad()
def _ns_sq(T: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz orthogonalisation of a small square matrix."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = T / (T.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.t()
        X = a * X + (b * A + c * A @ A) @ X
    return X


@torch.no_grad()
def _svd_polar(T, rank_tau):
    """FP64 core polar with a numerical-rank cutoff; no dense weight matrix."""
    U, S, Vh = torch.linalg.svd(T.double())
    tol = float(rank_tau) * S.max().clamp_min(1e-300)
    return ((U * (S > tol).to(U.dtype)) @ Vh).to(T.dtype)


_TANGENT_DEFAULTS = dict(tangent_lr_scale=1.0, polar="ns", polar_rank_tau=1e-3, tangent_decay="none",
                         tangent_horizon=0, tangent_floor=0.0, budget="none", budget_min=0.05, budget_beta=0.99)


def _check_tangent_options(o):
    if not (math.isfinite(o["tangent_lr_scale"]) and o["tangent_lr_scale"] >= 0):
        raise ValueError("tangent_lr_scale must be finite and >= 0")
    if o["polar"] not in ("ns", "svd") or o["tangent_decay"] not in ("none", "linear", "rsqrt") or o["budget"] not in ("none", "snr", "cos"):
        raise ValueError("invalid TSD polar, tangent decay or budget")
    if not (0 < o["polar_rank_tau"] < 1):
        raise ValueError("polar_rank_tau must be in (0, 1)")
    if o["tangent_decay"] != "none" and not o["tangent_horizon"] > 0:
        raise ValueError("a tangent decay needs tangent_horizon > 0")
    if not (0 <= o["tangent_floor"] <= 1 and 0 < o["budget_min"] <= 1 and 0 < o["budget_beta"] < 1):
        raise ValueError("invalid tangent floor, budget minimum or budget beta")


def _decay_factor(kind, k, horizon, floor):
    if kind == "linear":
        f = 1.0 - k / horizon
    elif kind == "rsqrt":
        f = 1.0 / math.sqrt(1.0 + k / horizon)
    else:
        return 1.0
    return max(float(floor), f)


@torch.no_grad()
def _proj_T_factored(L, R, U_B, V_A):
    """The tangent projector applied to L @ R, kept in factored form."""
    C = U_B.t() @ L
    D = R @ V_A
    L_new = torch.cat([U_B, L @ D], dim=1)
    R_new = torch.cat([C @ R - (C @ D) @ V_A.t(), V_A.t()], dim=0)
    return L_new, R_new


@torch.no_grad()
def _spd_inv(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Inverse of a small symmetric PSD matrix (a factor's Gram), as a pseudo-inverse when it is rank-deficient.

    A factor can be short of full rank - a layer fed a single vector (a modulation layer) has a rank-1 weight
    gradient, so its up leaves the zero start rank 1 - and a plain inverse then fails. Directions whose eigenvalue
    is below eps times the largest (never below float32's resolution) are dropped, so the step stays in the
    factor's own span; a full-rank Gram gets its ordinary inverse."""
    M = 0.5 * (M + M.t())
    evals, evecs = torch.linalg.eigh(M.double())
    tol = max(float(eps), 1e-6) * evals.abs().max().clamp_min(1e-30)
    inv = torch.where(evals > tol, 1.0 / evals.clamp_min(tol), torch.zeros_like(evals))
    return ((evecs * inv) @ evecs.t()).to(M.dtype)


# Research-branch repairs inspired by Grok's independent audits and reproduction
# package (rounds 1 and 2, 2026-11-25); integration and conservative stored-step
# acceptance implemented for Fizgig. No claim of GPU speed or diffusion quality.
@torch.no_grad()
def _factor_geometry(F, eps):
    """F is tall; return masked column basis and Gram pseudoinverse.

    Fixed-width zero columns avoid arbitrary QR completion directions and
    preserve the same numerical rank threshold in both operations.
    """
    f = F.double()
    gram = f.T @ f
    ev, v = torch.linalg.eigh((gram + gram.T) * 0.5)
    tol = max(float(eps), 1e-6) * ev.abs().max().clamp_min(1e-30)
    keep = ev > tol
    inv = torch.where(keep, ev.clamp_min(tol).reciprocal(), 0.0)
    return ((f @ v) * inv.sqrt()).to(F.dtype), ((v * inv) @ v.T).to(F.dtype)


@torch.no_grad()
def _rank_polar(g, eps):
    """Truncated polar factor, using only the small Gram in FP64."""
    transpose = g.shape[0] < g.shape[1]
    f = g.T if transpose else g
    # Compute polar in the same eigenbasis (separate helper avoids arbitrary
    # completion and avoids the BF16 NS noise amplification on low-rank input).
    x = f.double()
    gram = x.T @ x
    ev, v = torch.linalg.eigh((gram + gram.T) * 0.5)
    tol = max(float(eps), 1e-6) * ev.abs().max().clamp_min(1e-30)
    invroot = torch.where(ev > tol, ev.clamp_min(tol).rsqrt(), 0.0)
    out = ((x @ v) * invroot) @ v.T
    return (out.T if transpose else out).to(g.dtype)


@torch.no_grad()
def _product_norm(left, right):
    """Frobenius norm of a factored product, without materializing it.

    Use QR rather than cancellation-prone Gram contractions for the finite
    stored-step acceptance check. FP64 is a correctness-first research choice.
    """
    rl = torch.linalg.qr(left.double(), mode="r").R
    rr = torch.linalg.qr(right.double().T, mode="r").R
    return (rl @ rr.T).norm()


@torch.no_grad()
def _stored_change(A, B, an, bn):
    da = an.double() - A.double()
    db = bn.double() - B.double()
    return _product_norm(torch.cat((bn.double(), db), dim=1),
                         torch.cat((da, A.double()), dim=0))


@torch.no_grad()
def _bounded_candidate(A, B, da, db, cap, balance, zero_eps):
    """Conservative path-safe cap plus stored-dtype acceptance, including balance.

    Triangle bound: t*||B da+db A|| + t^2*||db da|| <= cap. It may
    underuse the cap when the terms cancel, but cannot jump a forbidden
    interval as a naive quartic root finder can. Retry after rounding, and
    leave factors unchanged if no safe representable candidate is found.
    """
    p = float(_product_norm(torch.cat((B.double(), db.double()), 1),
                           torch.cat((da.double(), A.double()), 0)))
    q = float(_product_norm(db, da))
    if not (math.isfinite(p) and math.isfinite(q)):
        raise FloatingPointError("nonfinite LoRA-TSD proposed update")
    t = min(1.0, 2*cap/(p + math.sqrt(p*p+4*q*cap))) if p+q else 1.0
    for _ in range(12):
        an = A + (da*t).to(A.dtype)
        bn = B + (db*t).to(B.dtype)
        c = None
        if balance:
            na, nb = float(an.float().norm()), float(bn.float().norm())
            if min(na, nb) >= zero_eps and math.isfinite(na+nb):
                c = math.sqrt(nb/na)
                # Use one stored scalar on both sides; verify actual result.
                ct = torch.as_tensor(c, dtype=A.dtype, device=A.device)
                if float(ct) > 0 and math.isfinite(float(ct)):
                    c = float(ct)
                    an, bn = an * ct, bn / ct.to(B.dtype)
                else:
                    c = None
        if torch.isfinite(an).all() and torch.isfinite(bn).all():
            actual = float(_stored_change(A, B, an, bn))
            if math.isfinite(actual) and actual <= cap:
                return an, bn, c, actual, t < 1.0
            # If balancing itself causes rounding drift, try the unbalanced
            # candidate before shrinking the update. No momentum transport then.
            if c is not None:
                an, bn = A + (da*t).to(A.dtype), B + (db*t).to(B.dtype)
                actual = float(_stored_change(A, B, an, bn))
                if math.isfinite(actual) and actual <= cap:
                    return an, bn, None, actual, t < 1.0
        t *= 0.5
    return A, B, None, 0.0, True


class LoRATSD(Optimizer):
    """`pairs`: [(down, up)] (fizgig.training.optimizers.polora_pairs) - each pair is its own param group."""

    def __init__(self, pairs, lr: float = 5e-3, momentum: float = 0.95, ball_iters: int = 5, ns_steps: int = 5,
                 max_delta_norm: float = 0.1, ridge_eps: float = 1e-8, balance: bool = True, zero_eps: float = 1e-12,
                 magnitude_scale: float = 1.0, log_every: int = 50, tangent_lr_scale: float = 1.0,
                 polar: str = "ns", polar_rank_tau: float = 1e-3, tangent_decay: str = "none", tangent_horizon: int = 0,
                 tangent_floor: float = 0.0, budget: str = "none", budget_min: float = 0.05, budget_beta: float = 0.99):
        pairs = list(pairs)
        if not pairs or any(a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]
                            or a.device != b.device or a.dtype != b.dtype for a,b in pairs):
            raise ValueError("LoRA-TSD requires compatible plain 2D factor pairs")
        if not math.isfinite(max_delta_norm) or max_delta_norm <= 0 or not math.isfinite(magnitude_scale) or magnitude_scale <= 0:
            raise ValueError("LoRA-TSD requires positive finite cap and magnitude")
        tangent = dict(tangent_lr_scale=float(tangent_lr_scale), polar=str(polar), polar_rank_tau=float(polar_rank_tau),
                       tangent_decay=str(tangent_decay), tangent_horizon=int(tangent_horizon),
                       tangent_floor=float(tangent_floor), budget=str(budget), budget_min=float(budget_min),
                       budget_beta=float(budget_beta))
        _check_tangent_options(tangent)
        if budget != "none" and not 0 <= momentum < 1:
            raise ValueError("TSD adaptive budget requires momentum in [0, 1)")
        groups = [{"params": [a, b]} for a, b in pairs]
        self._adapter_scales = [1.0] * len(groups)
        super().__init__(groups, dict(lr=lr, momentum=momentum, ball_iters=int(ball_iters), ns_steps=int(ns_steps),
                                      max_delta_norm=max_delta_norm, ridge_eps=ridge_eps, **tangent))
        self.balance, self.zero_eps = bool(balance), zero_eps
        # an attribute, not a group key: a resumed state keeps the magnitude set now (and an older state has none)
        self.magnitude_scale = float(magnitude_scale)
        self.log_every = int(log_every or 0)
        self._step_count = 0
        self.last_stats = None           # {"step", "magnitude", "avg_change", "capped", "pairs"} at its logging steps

    def set_adapter_scales(self, scales):
        """The live model's alpha/r*strength, not stale checkpoint metadata."""
        scales = [abs(float(s)) for s in scales]
        if len(scales) != len(self.param_groups) or any(not math.isfinite(s) for s in scales):
            raise ValueError("one finite adapter scale required per LoRA pair")
        self._adapter_scales = scales

    def load_state_dict(self, state_dict):
        # Restore FP32 momentum from the serialized source rather than from
        # Optimizer's parameter-dtype cast (which can lose BF16 precision).
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for k in _TANGENT_DEFAULTS:
                group.setdefault(k, self.defaults[k])
            _check_tangent_options(group)
        for old_group, new_group in zip(state_dict["param_groups"], self.param_groups):
            for old_id, p in zip(old_group["params"], new_group["params"]):
                old = state_dict["state"].get(old_id, {}).get("momentum_buffer")
                if old is not None:
                    self.state[p]["momentum_buffer"] = old.to(device=p.device, dtype=torch.float32).clone()

    def _buf(self, p, g, mom):
        st = self.state[p]
        if "momentum_buffer" not in st:
            st["momentum_buffer"] = torch.zeros_like(g)
        # Optimizer.load_state_dict may cast momentum to the parameter dtype.
        # Restore FP32 arithmetic for BF16 parameter storage.
        if st["momentum_buffer"].dtype != g.dtype:
            st["momentum_buffer"] = st["momentum_buffer"].to(g.dtype)
        st["momentum_buffer"].mul_(mom).add_(g, alpha=1.0 - mom)
        return st["momentum_buffer"]

    def _budget_update(self, group, A, gA, gB, mA, mB):
        """Round 10 factor-space EMA of energy or directional agreement."""
        kind = group.get("budget", "none")
        if kind == "none":
            return
        st, b2 = self.state[A], group.get("budget_beta", 0.99)
        g2 = float(gA.double().pow(2).sum() + gB.double().pow(2).sum())
        if kind == "cos":
            if mA is not None:
                mg = float((mA.double() * gA.double()).sum() + (mB.double() * gB.double()).sum())
                mn = math.sqrt(float(mA.double().pow(2).sum() + mB.double().pow(2).sum()))
                cos = mg / (mn * math.sqrt(g2)) if mn > 0 and g2 > 0 else 0.0
                st["tsd_cos_ema"] = b2 * st.get("tsd_cos_ema", 0.0) + (1 - b2) * cos
                st["tsd_cos_n"] = st.get("tsd_cos_n", 0) + 1
            return
        st["tsd_v"] = b2 * st.get("tsd_v", 0.0) + (1 - b2) * g2
        st["tsd_n"] = st.get("tsd_n", 0) + 1

    def _budget_factor(self, group, A, mA, mB):
        kind = group.get("budget", "none")
        if kind == "none":
            return 1.0
        st, lo = self.state[A], group.get("budget_min", 0.05)
        b2 = group.get("budget_beta", 0.99)
        if kind == "cos":
            n = st.get("tsd_cos_n", 0)
            if n == 0:
                return 1.0
            f = st["tsd_cos_ema"] / (1 - b2 ** n)
        else:
            n = st.get("tsd_n", 0)
            if n == 0:
                return 1.0
            beta = group["momentum"]
            k = min(n, 10000)
            mhat2 = float(mA.double().pow(2).sum() + mB.double().pow(2).sum()) / (1 - beta ** k) ** 2
            vhat = st["tsd_v"] / (1 - b2 ** n)
            c = (1 - beta) / (1 + beta) * (1 + beta ** k) / (1 - beta ** k)
            if vhat <= 0 or c >= 1 - 1e-6:
                return 1.0
            f = math.sqrt(min(max((mhat2 / vhat - c) / (1 - c), 0.0), 1.0))
        return min(1.0, max(lo, f))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._step_count += 1
        mag = self.magnitude_scale
        changes, capped, warm_pairs, unchanged_pairs = [], 0, 0, 0
        budget_factors, tangent_factors = [], []
        for group, scale in zip(self.param_groups, self._adapter_scales):
            A, B = group["params"]
            if A.grad is None and B.grad is None:
                continue
            if A.grad is None or B.grad is None:
                raise ValueError("LoRA-TSD needs both gradients or neither")
            if scale == 0:
                continue
            lr, cap = group["lr"] * mag, group["max_delta_norm"] * mag / scale
            if not math.isfinite(lr) or lr < 0 or not math.isfinite(cap) or cap <= 0:
                raise ValueError("invalid LoRA-TSD learning rate or effective cap")
            if lr == 0:
                continue
            with torch.autocast(device_type=A.device.type, enabled=False):
                A32, B32 = A.float(), B.float()
                if group.get("budget", "none") != "none":
                    pa, pb = self.state[A].get("momentum_buffer"), self.state[B].get("momentum_buffer")
                    self._budget_update(group, A, A.grad.float(), B.grad.float(),
                                        None if pa is None else pa.float(), None if pb is None else pb.float())
                G_A = self._buf(A, A.grad.float(), group["momentum"])
                G_B = self._buf(B, B.grad.float(), group["momentum"])
                za, zb = float(A32.norm()) < self.zero_eps, float(B32.norm()) < self.zero_eps
                if za and zb:
                    continue
                # Until the effective adapter is sufficiently large, one-factor
                # updates are linear and do not invert a nearly empty factor.
                warm = not za and float(_product_norm(B32,A32)) < 2*cap
                if za or zb or warm:
                    warm_pairs += 1
                    if za:
                        dA, dB = -lr * _rank_polar(G_A, group["ridge_eps"]), torch.zeros_like(B32)
                    else:
                        dA, dB = torch.zeros_like(A32), -lr * _rank_polar(G_B, group["ridge_eps"])
                    rebalance = False
                else:
                    U_B, BtB_i = _factor_geometry(B32, group["ridge_eps"])
                    V_A, AAt_i = _factor_geometry(A32.T, group["ridge_eps"])
                    M = BtB_i @ (G_A @ A32.T) @ AAt_i
                    L = torch.cat([B32, G_B @ AAt_i], dim=1)
                    R = torch.cat([BtB_i @ G_A - M @ A32, A32], dim=0)
                    for _ in range(group["ball_iters"]):
                        Q_L, R_L = torch.linalg.qr(L, mode="reduced")
                        Q_R, R_R = torch.linalg.qr(R.T, mode="reduced")
                        if group.get("polar", "ns") == "svd":
                            T_orth = _svd_polar(R_L @ R_R.T, group.get("polar_rank_tau", 1e-3))
                        else:
                            T_orth = _ns_sq(R_L @ R_R.T, steps=group["ns_steps"])
                        L, R = _proj_T_factored(Q_L @ T_orth, Q_R.T, U_B, V_A)
                    st = self.state[A]
                    k = st.get("tsd_tangent_steps", 0)
                    budget_factor = self._budget_factor(group, A, G_A, G_B)
                    fac = (group.get("tangent_lr_scale", 1.0)
                           * _decay_factor(group.get("tangent_decay", "none"), k, group.get("tangent_horizon", 0),
                                           group.get("tangent_floor", 0.0))
                           * budget_factor)
                    if group.get("tangent_decay", "none") != "none" or group.get("budget", "none") != "none" or "tsd_tangent_steps" in st:
                        st["tsd_tangent_steps"] = k + 1
                    budget_factors.append(budget_factor)
                    tangent_factors.append(fac)
                    if fac != 1.0:
                        lr = lr * fac
                    dB = -lr * (L @ (R @ A32.T)) @ AAt_i
                    dA = BtB_i @ (-lr * ((B32.T @ L) @ R) - (B32.T @ dB) @ A32)
                    rebalance = self.balance
                an, bn, c, actual, hit = _bounded_candidate(A, B, dA, dB, cap, rebalance, self.zero_eps)
                A.copy_(an)
                B.copy_(bn)
                if c is not None:
                    self.state[A]["momentum_buffer"].div_(c)
                    self.state[B]["momentum_buffer"].mul_(c)
                changes.append(scale * actual)
                capped += int(hit)
                unchanged_pairs += int(actual == 0.0)
        if changes and self.log_every and (self._step_count == 1 or self._step_count % self.log_every == 0):
            lrs = [g["lr"] for g in self.param_groups]
            self.last_stats = {"step": self._step_count, "magnitude": mag, "avg_change": sum(changes) / len(changes),
                               "capped": capped, "pairs": len(changes)}
            logging.getLogger("fizgig.lora_tsd").info(
                "[lora_tsd] step %d: magnitude %.3g x LR %.3g, avg weight change a pair %.4f (cap %.3g), "
                "%d of %d pairs at the cap", self._step_count, mag, sum(lrs) / len(lrs),
                self.last_stats["avg_change"], self.param_groups[0]["max_delta_norm"] * mag, capped, len(changes))
            logging.getLogger("fizgig.lora_tsd").info(
                "[geometry-tsd] update=%d max_change=%.9g warm=%d unchanged=%d shrunk=%d pairs=%d",
                self._step_count, max(changes), warm_pairs, unchanged_pairs, capped, len(changes))
            if budget_factors:
                logging.getLogger("fizgig.lora_tsd").info(
                    "[geometry-tangent] update=%d pairs=%d budget_min=%.9g budget_mean=%.9g budget_max=%.9g lr_factor_mean=%.9g change_sum=%.9g",
                    self._step_count, len(budget_factors), min(budget_factors), sum(budget_factors)/len(budget_factors),
                    max(budget_factors), sum(tangent_factors)/len(tangent_factors), sum(changes))
        return loss


__all__ = ["LoRATSD"]
