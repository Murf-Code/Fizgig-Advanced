"""Read-only research helpers for plain 2D LoRA pairs. No training integration."""
from __future__ import annotations

import torch


def _check(a, b):
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
        raise ValueError("Expected A[r,n], B[m,r] with matching rank")
    if min(*a.shape, *b.shape) <= 0 or a.device != b.device:
        raise ValueError("Nonempty factors on the same device required")


def _dtype(*xs):
    return torch.float64 if any(x.dtype == torch.float64 for x in xs) else torch.float32


@torch.no_grad()
def core_svd(a, b):
    """Reduced QR then compact SVD; never construct the full B @ A."""
    _check(a, b)
    dtype = _dtype(a, b)
    with torch.autocast(device_type=a.device.type, enabled=False):
        qb, rb = torch.linalg.qr(b.detach().to(dtype), mode="reduced")
        qa, ra = torch.linalg.qr(a.detach().to(dtype).T, mode="reduced")
        u, s, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
        return qb, u, s, vh, qa


@torch.no_grad()
def spectrum(a, b, scale=1.0):
    return core_svd(a, b)[2] * abs(float(scale))


@torch.no_grad()
def metrics(a, b, scale=1.0):
    """Spectral entropy uses singular values; energy uses squared singular values."""
    s = spectrum(a, b, scale)
    total = s.sum()
    energy = s.square().sum()
    tiny = torch.finfo(s.dtype).tiny
    p = s / total.clamp_min(tiny)
    entropy = -(p * p.clamp_min(tiny).log()).sum()
    effective = torch.where(total > 0, entropy.exp(), torch.zeros_like(total))
    stable = energy / s[0].square().clamp_min(tiny)
    tol = s[0] * max(a.shape[1], b.shape[0]) * torch.finfo(s.dtype).eps
    numerical = (s > tol).sum()
    na, nb = a.detach().to(s.dtype).norm(), b.detach().to(s.dtype).norm()
    # Return device tensors so the caller can batch host transfers.
    return {
        "singular_values": s, "effective_rank": effective, "stable_rank": stable,
        "spectral_entropy": entropy, "numerical_rank": numerical,
        "adapter_frobenius": energy.sqrt(), "spectral_norm": s[0],
        "a_norm": na, "b_norm": nb,
        "norm_ratio_a_over_b": torch.where(nb > 0, na / nb.clamp_min(tiny),
                                          torch.full_like(nb, float("nan"))),
    }


@torch.no_grad()
def balanced_factors(a, b, rank=None):
    """Return FP32/FP64 SVD-balanced factors; never mutate training state.

    For full rank preservation, keep the nominal rank and zero-pad if necessary.
    Truncation is an offline compression experiment, not a live optimizer step.
    """
    qb, u, s, vh, qa = core_svd(a, b)
    rank = a.shape[0] if rank is None else int(rank)
    if not 1 <= rank <= a.shape[0]:
        raise ValueError("rank must be within the original adapter rank")
    k = min(rank, s.numel())
    root = s[:k].sqrt()
    bn = (qb @ u[:, :k]) * root
    an = root[:, None] * (vh[:k] @ qa.T)
    if k < rank:
        bn = torch.nn.functional.pad(bn, (0, rank-k))
        an = torch.nn.functional.pad(an, (0, 0, 0, rank-k))
    return an, bn


@torch.no_grad()
def delta_norm(a0, b0, a1, b1, scale=1.0):
    """Exact Frobenius norm of scale*(B1 A1 - B0 A0), via a <=2r core.

    Use the difference expansion to avoid subtracting two large products.
    Includes the quadratic term. Actual stored factor rounding is included.
    """
    _check(a0, b0)
    _check(a1, b1)
    if a0.shape != a1.shape or b0.shape != b1.shape:
        raise ValueError("Snapshots must have identical shapes")
    dtype = _dtype(a0, b0, a1, b1)
    a0, b0, a1, b1 = (x.detach().to(dtype) for x in (a0,b0,a1,b1))
    # B1*(A1-A0) + (B1-B0)*A0
    left = torch.cat((b1, b1-b0), dim=1)
    right = torch.cat((a1-a0, a0), dim=0)
    with torch.autocast(device_type=a0.device.type, enabled=False):
        rb = torch.linalg.qr(left, mode="r").R
        ra = torch.linalg.qr(right.T, mode="r").R
        return (rb @ ra.T).norm() * abs(float(scale))
