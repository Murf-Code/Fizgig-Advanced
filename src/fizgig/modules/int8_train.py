"""INT8 (W8A8) matmul for a FROZEN training base — differentiable, unlike modules/int8.py.

modules/int8.py runs int8 under no_grad for previews. Training needs a gradient path: the LoRA
sits downstream of these Linears, so grad must flow *through* the frozen base to reach it.

Two things make this cheaper and safer here than a general W8A8 implementation:

1. **The base is frozen.** We never need grad w.r.t. the weight — only w.r.t. the input. Weight
   gradients are where int8 quantisation does the most damage, and we simply don't compute them.

2. **Gradient checkpointing is on by default**, so the forward runs TWICE per step (once
   forward, once recomputed inside backward). An int8 forward therefore covers two thirds of
   the matmul work even if the backward stays in bf16.

The backward is bf16 (exact gradients, the weight dequantised in the compute dtype). An INT8 backward was
measured and removed (#221): rounding each gradient row to INT8 lets its outliers set the scale and rounds most
of the row to zero, so a LoRA's gradients came out at cosine 0.2-0.6 to the exact ones and likeness lagged.

Opt-in, experimental (set_rotated_backward, the "Rotated INT8 backward" tick box): an INT8 backward that first
ROTATES the contracted dimension with a block Hadamard matrix H (orthonormal, so g @ W = (g H)(H W) exactly). The
rotation spreads each row's outliers over its block, so the per-row INT8 scale no longer flattens the rest. The
rotated weight H W is quantised once per layer (per input column) and kept beside the INT8 weight - about the INT8
block weights' size again in memory. checks/test_int8_rotated_backward.py measures it against the exact backward.

Measured at Krea 2's main shape (M=1024, K=N=6144), RTX 5090:
    dequant + bf16 matmul (current)   0.767 ms
    _int_mm + activation quant        0.224 ms
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_WARNED_FALLBACK = False
_ROTATED = {"on": False}
_HADAMARD = {}


def set_rotated_backward(on=True):
    """The rotated INT8 backward on or off for every INT8 frozen Linear (off by default: the exact bf16 backward)."""
    _ROTATED["on"] = bool(on)


def rotated_backward_on():
    return _ROTATED["on"]


def _block(n, most=256):
    """The Hadamard block for a contracted size n: the largest power of two up to `most` dividing it (0: none of
    16 or more, and that layer keeps the exact backward)."""
    b = most
    while b >= 16:
        if n % b == 0:
            return b
        b //= 2
    return 0


def _hadamard(b, device):
    """The b x b orthonormal Hadamard matrix (Sylvester's doubling), fp32, cached per device."""
    key = (b, str(device))
    h = _HADAMARD.get(key)
    if h is None:
        h = torch.ones(1, 1)
        while h.shape[0] < b:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        h = (h / b ** 0.5).to(device)
        _HADAMARD[key] = h
    return h


@torch.no_grad()
def rotated_weight(w_i8_nk, w_scale_1n):
    """(H W) for the backward, quantised per input column: (int8 (K, N) contiguous, fp32 scale (1, K)), or None when
    the layer has no Hadamard block. W = the dequantised (N, K) weight; H rotates its N (output) dimension."""
    n, k = w_i8_nk.shape
    b = _block(n)
    if not b:
        return None
    w = w_i8_nk.to(torch.float32) * w_scale_1n.reshape(-1, 1).to(torch.float32)        # (N, K)
    hw = torch.einsum("ij,bjk->bik", _hadamard(b, w.device), w.view(n // b, b, k)).reshape(n, k)
    del w
    q_kn, s_k1 = _quant_per_token(hw.t())                                                 # per input column
    return q_kn.contiguous(), s_k1.reshape(1, -1)


def _int_mm_padded(a_i8, b_kn_t):
    """torch._int_mm with M <= 16 padded (as the forward does); None when it can't run (CPU builds, odd shapes)."""
    m = a_i8.shape[0]
    pad = 17 - m if m <= 16 else 0
    try:
        acc = torch._int_mm(torch.nn.functional.pad(a_i8, (0, 0, 0, pad)) if pad else a_i8, b_kn_t)
    except Exception:
        return None
    return acc[:m] if pad else acc


def _quant_per_token(t: torch.Tensor):
    """Symmetric per-row int8 quantisation. Returns (int8 tensor, fp32 scale of shape (M,1))."""
    scale = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8).to(torch.float32) / 127.0
    q = (t.to(torch.float32) / scale).round_().clamp_(-127, 127).to(torch.int8)
    return q, scale


class _Int8FrozenLinear(torch.autograd.Function):
    """y = (x @ W) with W frozen int8. Only grad_input is produced — the base never trains."""

    @staticmethod
    def forward(ctx, x, w_i8_nk, w_scale_1n, bias, rot_kn=None, rot_s=None):
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        a_i8, a_scale = _quant_per_token(x2d)
        try:
            # torch._int_mm refuses M <= 16. Rather than let those fall through to the fp32 path,
            # pad the rows with zeros and slice the result back: zero rows produce zero output and
            # are discarded, so this is exact. Without it, whether a matmul runs quantised depends
            # on how many tokens happen to be in the tensor — which made the same weights give
            # different answers for short captions vs long ones, silently.
            m = x2d.shape[0]
            pad_rows = 17 - m if m <= 16 else 0
            if pad_rows:
                a_i8 = torch.nn.functional.pad(a_i8, (0, 0, 0, pad_rows))
            acc = torch._int_mm(a_i8, w_i8_nk.t())                 # (M,N) int32, free view
            if pad_rows:
                acc = acc[:m]
            # Scale in the compute dtype: an fp32 (M,N) intermediate is 4x the bytes and this
            # step is bandwidth-bound, not arithmetic-bound.
            out = acc.to(x.dtype) * a_scale.to(x.dtype) * w_scale_1n.to(x.dtype)
            ctx._fell_back = False
        except Exception:
            # _int_mm requires M > 16 (and dislikes some alignments), so short sequences land
            # here and run in fp32 instead — MORE accurate than int8, not less, but it means
            # numerics depend on token count: the same weights give different answers either
            # side of that threshold. Worth knowing about, because it silently made a text
            # padding experiment look like a masking bug (see docs/PERF_ROADMAP.md), so say it
            # once rather than never.
            w = w_i8_nk.to(torch.float32) * w_scale_1n.reshape(-1, 1)
            out = (x2d.to(torch.float32) @ w.t()).to(x.dtype)
            ctx._fell_back = True
            # Global write + logging are ONLY legal in eager: dynamo rejects a module-global
            # mutation inside an autograd.Function outright, and this branch is rare enough
            # that a compiled run would train for hours before an odd shape finally took it
            # and died. is_compiling() is a trace-time constant, so the compiled graph simply
            # omits this block (the warning still fires on the first EAGER fallback).
            if not torch.compiler.is_compiling():
                global _WARNED_FALLBACK
                if not _WARNED_FALLBACK:
                    _WARNED_FALLBACK = True
                    logger.info("[int8] some matmuls fall back to fp32 (torch._int_mm needs M > 16, "
                                "got %d) — those run exact rather than quantised, so int8 numerics "
                                "vary with sequence length", x2d.shape[0])
        if bias is not None:
            out = out + bias
        ctx.save_for_backward(w_i8_nk, w_scale_1n, rot_kn, rot_s)
        ctx.orig_shape = orig_shape
        ctx.needs_bias_grad = bias is not None and bias.requires_grad
        return out.reshape(*orig_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_out):
        w_i8_nk, w_scale_1n, rot_kn, rot_s = ctx.saved_tensors
        g2d = grad_out.reshape(-1, grad_out.shape[-1])
        grad_bias = g2d.sum(dim=0).to(grad_out.dtype) if ctx.needs_bias_grad else None
        if rot_kn is not None:
            # the rotated INT8 backward (experimental): g @ W = (g H) @ (H W), both sides rounded to INT8
            n, b = g2d.shape[1], _block(g2d.shape[1])
            gh = (g2d.to(torch.float32).view(-1, n // b, b) @ _hadamard(b, g2d.device)).view(-1, n)
            a_i8, a_scale = _quant_per_token(gh)
            del gh
            acc = _int_mm_padded(a_i8, rot_kn.t())
            if acc is None:              # no int8 matmul here: the same INT8 numbers, multiplied in fp32
                acc = a_i8.to(torch.float32) @ rot_kn.t().to(torch.float32)
            grad_x = (acc.to(torch.float32) * a_scale * rot_s).to(grad_out.dtype)
            return grad_x.reshape(*ctx.orig_shape), None, None, grad_bias, None, None
        # bf16 backward: exact gradients. Dequantise in the COMPUTE dtype - an fp32 (K,N) copy of a 6144x6144 weight
        # is 150 MB and made this path slower than dense bf16.
        w = w_i8_nk.to(grad_out.dtype) * w_scale_1n.reshape(-1, 1).to(grad_out.dtype)  # (N,K)
        grad_x = g2d @ w
        return grad_x.reshape(*ctx.orig_shape), None, None, grad_bias, None, None


def int8_weight_only_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Weight-only INT8: the int8 weight (and its per-channel scale) back to the activations' dtype for a plain matmul -
    the activations are never quantised. The memory of INT8 without its per-token activation rounding, for a model whose
    activations carry outliers that rounding crushes (Vimoo: mottled, flickering backgrounds). Slower than _int_mm."""
    w = self.weight.data.to(x.dtype) * self._int8_wscale.reshape(-1, 1).to(x.dtype)
    return F.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))


def int8_train_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    rot = None
    if _ROTATED["on"] and torch.is_grad_enabled():
        rot = getattr(self, "_int8_rot", None)
        if rot is not None and rot[0].device != self.weight.device:   # the weight moved (block swap): follow it
            rot = tuple(t.to(self.weight.device, non_blocking=True) for t in rot)
            self._int8_rot = rot
        elif rot is None and not getattr(self, "_int8_rot_none", False):
            rot = rotated_weight(self.weight.data, self._int8_wscale)
            if rot is None:
                self._int8_rot_none = True            # no Hadamard block for this size: the exact backward
            else:
                self._int8_rot = rot
    if rot is None:
        return _Int8FrozenLinear.apply(x, self.weight.data, self._int8_wscale, self.bias)
    return _Int8FrozenLinear.apply(x, self.weight.data, self._int8_wscale, self.bias, rot[0], rot[1])
