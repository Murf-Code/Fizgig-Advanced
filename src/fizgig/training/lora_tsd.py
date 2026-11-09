"""LoRA-TSD - tangent-space spectral descent for LoRA pairs.

Adapted from https://github.com/brain-lab-research/LoRA-TSD (src/optimizers/lora_tsd.py, commit e6ef0e16; a fork
of dimativator/LoRA-TSD), MIT License, Copyright (c) 2026 Dmitrii Andriianov - the licence is in
docs/THIRD_PARTY_NOTICES.md.

Each LoRA's down (A, r x n) and up (B, m x r) train together: the factors' gradients rebuild the weight gradient's
projection onto the tangent space of rank-r matrices at W = B A, a few rounds of matrix-sign (Newton-Schulz on a
small QR core) and re-projection turn it into a steepest-descent step under the spectral norm, and that step is
split back into dA and dB. The weight's change is capped at max_delta_norm (Frobenius) a step.

Changes from the reference: trimmed to the update itself (its diagnostics, logging and lr_A / lr_B schedulers
removed); one learning rate per pair in the group's "lr" (the reference's lr_A), so Fizgig's scheduler and
Adaptive LR move it as for any optimizer; the reference config's defaults (momentum 0.95, 5 ball iterations,
5 Newton-Schulz steps, max_delta_norm 0.1, norm balancing on); and a start for a zero factor (Fizgig's addition):
a LoRA's up starts at zero, where the tangent space is undefined, so until both factors are non-zero the zero one
steps alone by the matrix sign of its momentum (a Muon step), capped the same way.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from fizgig.training.muon import DEFAULT_A, DEFAULT_B, DEFAULT_C, EPS, _zeropower_via_newtonschulz


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
def _proj_T_factored(L, R, U_B, V_A):
    """The tangent projector applied to L @ R, kept in factored form."""
    C = U_B.t() @ L
    D = R @ V_A
    L_new = torch.cat([U_B, L @ D], dim=1)
    R_new = torch.cat([C @ R - (C @ D) @ V_A.t(), V_A.t()], dim=0)
    return L_new, R_new


@torch.no_grad()
def _spd_inv(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Inverse of a small SPD matrix after a relative ridge."""
    ridge = eps * M.diagonal().mean().clamp_min(1e-12)
    return torch.linalg.inv(M + ridge * torch.eye(M.shape[0], device=M.device, dtype=M.dtype))


class LoRATSD(Optimizer):
    """`pairs`: [(down, up)] (fizgig.training.optimizers.polora_pairs) - each pair is its own param group."""

    def __init__(self, pairs, lr: float = 5e-3, momentum: float = 0.95, ball_iters: int = 5, ns_steps: int = 5,
                 max_delta_norm: float = 0.1, ridge_eps: float = 1e-8, balance: bool = True, zero_eps: float = 1e-12):
        groups = [{"params": [a, b]} for a, b in pairs]
        super().__init__(groups, dict(lr=lr, momentum=momentum, ball_iters=int(ball_iters), ns_steps=int(ns_steps),
                                      max_delta_norm=max_delta_norm, ridge_eps=ridge_eps))
        self.balance, self.zero_eps = bool(balance), zero_eps

    def _buf(self, p, g, mom):
        st = self.state[p]
        if "momentum_buffer" not in st:
            st["momentum_buffer"] = torch.zeros_like(g)
        st["momentum_buffer"].mul_(mom).add_(g, alpha=1.0 - mom)
        return st["momentum_buffer"]

    @torch.no_grad()
    def _rebalance(self, A, B):
        """Even the factors' sizes without changing B A (the reference's balance_mode="norm")."""
        nA, nB = A.float().norm(), B.float().norm()
        if nA < self.zero_eps or nB < self.zero_eps:
            return
        c = (nB / nA).sqrt()
        A.mul_(c.to(A.dtype))
        B.div_(c.to(B.dtype))
        if "momentum_buffer" in self.state[A]:
            self.state[A]["momentum_buffer"].div_(c)
        if "momentum_buffer" in self.state[B]:
            self.state[B]["momentum_buffer"].mul_(c)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            A, B = group["params"]
            if A.grad is None or B.grad is None:
                continue
            lr, cap = group["lr"], group["max_delta_norm"]
            A32, B32 = A.float(), B.float()
            G_A = self._buf(A, A.grad.float(), group["momentum"])
            G_B = self._buf(B, B.grad.float(), group["momentum"])
            if B32.norm() < self.zero_eps or A32.norm() < self.zero_eps:
                # a zero factor (the up at a LoRA's start): no tangent space yet - step the zero one alone
                if B32.norm() < self.zero_eps and A32.norm() >= self.zero_eps:
                    d = -lr * _zeropower_via_newtonschulz(G_B, (DEFAULT_A, DEFAULT_B, DEFAULT_C),
                                                          group["ns_steps"], EPS).float()
                    dw = (d @ A32).norm()
                    if dw > cap:
                        d.mul_(cap / dw)
                    B.add_(d.to(B.dtype))
                elif A32.norm() < self.zero_eps and B32.norm() >= self.zero_eps:
                    d = -lr * _zeropower_via_newtonschulz(G_A, (DEFAULT_A, DEFAULT_B, DEFAULT_C),
                                                          group["ns_steps"], EPS).float()
                    dw = (B32 @ d).norm()
                    if dw > cap:
                        d.mul_(cap / dw)
                    A.add_(d.to(A.dtype))
                continue
            U_B = torch.linalg.qr(B32, mode="reduced")[0]
            V_A = torch.linalg.qr(A32.t(), mode="reduced")[0]
            BtB_i = _spd_inv(B32.t() @ B32, group["ridge_eps"])
            AAt_i = _spd_inv(A32 @ A32.t(), group["ridge_eps"])
            # the tangent-projected weight gradient as L_g @ R_g
            M = BtB_i @ (G_A @ A32.t()) @ AAt_i
            L = torch.cat([B32, G_B @ AAt_i], dim=1)
            R = torch.cat([BtB_i @ G_A - M @ A32, A32], dim=0)
            for _ in range(group["ball_iters"]):     # msign through the small QR core, then back onto T
                Q_L, R_L = torch.linalg.qr(L, mode="reduced")
                Q_R, R_R = torch.linalg.qr(R.t(), mode="reduced")
                T_orth = _ns_sq(R_L @ R_R.t(), steps=group["ns_steps"])
                L, R = _proj_T_factored(Q_L @ T_orth, Q_R.t(), U_B, V_A)
            X_fro = ((L.t() @ L) * (R @ R.t())).sum().clamp_min(0).sqrt()
            # factor updates whose linearised product is -lr L R
            dB = -lr * (L @ (R @ A32.t())) @ AAt_i
            dA = BtB_i @ (-lr * ((B32.t() @ L) @ R) - (B32.t() @ dB) @ A32)
            dw = lr * X_fro
            if dw > cap:
                dA.mul_(cap / dw)
                dB.mul_(cap / dw)
            A.add_(dA.to(A.dtype))
            B.add_(dB.to(B.dtype))
            if self.balance:
                self._rebalance(A, B)
        return loss


__all__ = ["LoRATSD"]
