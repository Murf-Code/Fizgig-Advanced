"""Krea 2 Text Enhancer: bounded conditioning gain and residual, adapted from Fizgig PR #393.

The 12-tap encoder output (12 x 2560) is viewed as 24 x 1280 chunks. A second text-fusion pass
runs on the gain-scaled context; its difference from the plain pass is added back under a
per-token RMS cap. The functions remain differentiable so the training loss can update trainable
LoRA weights in txtfusion and txtmlp.
"""
from __future__ import annotations

import torch


PROFILE_12 = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 5.0, 1.1, 4.0, 1.0)
CHUNK_PROFILE = PROFILE_12 + PROFILE_12
CHUNKS, CHUNK_DIM = 24, 1280
TAPS, TAP_DIM = 12, 2560
MAX_STRENGTH = 2.0
GLOBAL_MULTIPLIER = 15.0
TOKEN_REL_CAP = 0.75
SCALE_MIN, SCALE_MAX = 0.25, 4.0


def clamped_strength(value) -> float:
    """Clamp the donor's strength to 0..2; invalid/negative values safely disable the enhancement."""
    try:
        strength = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not strength > 0.0:  # includes zero, negative values, and NaN
        return 0.0
    return min(strength, MAX_STRENGTH)


def clamped_text_scale(value) -> float:
    """Clamp txtmlp's output multiplier to 0.25..4; blank or invalid values are neutral."""
    try:
        scale = float(value)
    except (TypeError, ValueError):
        return 1.0
    if scale != scale:  # NaN
        return 1.0
    return min(max(scale, SCALE_MIN), SCALE_MAX)


def applies(context: torch.Tensor) -> bool:
    """Only Krea 2's (B, sequence, 12 taps, 2560) conditioning is eligible."""
    return context.dim() == 4 and tuple(context.shape[-2:]) == (TAPS, TAP_DIM)


def gains(strength, device, dtype) -> torch.Tensor:
    """The donor's 24 per-chunk gains, calculated in float32."""
    base = torch.tensor(CHUNK_PROFILE, device=device, dtype=torch.float32)
    return (1.0 + clamped_strength(strength) * (base - 1.0)).to(dtype)


def scaled_context(context: torch.Tensor, strength) -> torch.Tensor:
    """Build the second pass' conditioning; return the exact input tensor when off or ineligible."""
    value = clamped_strength(strength)
    if value <= 0.0 or not applies(context):
        return context
    batch, tokens = context.shape[:2]
    gain = gains(value, context.device, context.dtype).view(1, 1, CHUNKS, 1)
    global_gain = 1.0 + value * (GLOBAL_MULTIPLIER - 1.0)
    return (context.reshape(batch, tokens, CHUNKS, CHUNK_DIM) * gain * global_gain).reshape(
        batch, tokens, TAPS, TAP_DIM
    )


def capped_residual(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Add candidate-reference without exceeding 0.75 of each reference token's RMS."""
    ref32, cand32 = reference.float(), candidate.float()
    delta = cand32 - ref32
    base_rms = ref32.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    relative_rms = delta.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8) / base_rms
    factor = (TOKEN_REL_CAP / relative_rms).clamp(max=1.0)
    return (ref32 + delta * factor).to(candidate.dtype)


def enhance(run, context: torch.Tensor, strength) -> torch.Tensor:
    """Run both numerically matched fusion passes and return the bounded residual enhancement."""
    value = clamped_strength(strength)
    reference = run(context)
    if value <= 0.0 or not applies(context):
        return reference
    candidate = run(scaled_context(context, value))
    return capped_residual(reference, candidate)
