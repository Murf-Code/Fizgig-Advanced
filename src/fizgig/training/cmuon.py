"""CMuon - Muon with each fused weight split into its parts before the orthogonal step.

Our own code, written from the method in arXiv 2608.02502 (no reference code used). A DiT fuses several
projections into one Linear: Q, K and V in one `qkv`, the gate and up halves of a SwiGLU in one `gate_up`, the
shift / scale / gate rows of an AdaLN modulation in one `adaLN_modulation`. Plain Muon orthogonalises the whole
fused matrix, so one part's large singular directions set the step for the others. CMuon orthogonalises each part
(a chunk of the output rows) on its own and scales each to AdamW's step size for its own shape
(0.2 x sqrt(max(rows, cols)), Moonlight's match), so every part moves at the size it would as a separate Linear.

Momentum and Newton-Schulz are torch.optim.Muon's (via fizgig.training.muon, BSD-3-Clause). A weight with no chunk
plan steps exactly as Muon. 2D weights only - the fine-tune trains Linear weights alone.
"""

from __future__ import annotations

import math
import re

import torch

from fizgig.training.muon import (DEFAULT_A, DEFAULT_B, DEFAULT_C, DEFAULT_NS_STEPS, EPS,
                                  _zeropower_via_newtonschulz)

# name patterns of fused output projections: (regex on the weight's name, fixed part count or None = out / width)
FUSED = (
    (re.compile(r"(^|[._])(to_)?qkv(_proj)?([._]|$)|in_proj_qkv|attn_qkv|\.qkv\b", re.I), None),
    (re.compile(r"gate_up|w13|w12|fc_gate_up", re.I), 2),
    (re.compile(r"ada_?ln|modulation|(^|[._])mod(_lin)?([._]|$)", re.I), None),
)


def model_width(shapes) -> int:
    """The model's hidden size: the commonest input width among the weights (a DiT's Linears mostly read it)."""
    seen = {}
    for s in shapes:
        if len(s) == 2:
            seen[int(s[1])] = seen.get(int(s[1]), 0) + 1
    return max(seen, key=lambda w: (seen[w], w)) if seen else 0


def chunk_plan(named, width=None) -> dict:
    """{name: number of row chunks} for the fused weights among `named` [(name, tensor or shape)]. QKV and AdaLN
    split into out / width parts (3 for QKV; 4, 6 or 9 for a modulation), gate+up into 2. Anything that doesn't
    divide evenly is left whole (steps as plain Muon)."""
    shapes = [(n, tuple(getattr(t, "shape", t))) for n, t in named]
    width = width or model_width(s for _n, s in shapes)
    out = {}
    for n, s in shapes:
        if len(s) != 2:
            continue
        for rx, k in FUSED:
            if not rx.search(n):
                continue
            parts = k if k else (s[0] // width if width and s[0] % width == 0 else 0)
            if parts and parts > 1 and s[0] % parts == 0:
                out[n] = int(parts)
            break
    return out


class CMuon(torch.optim.Optimizer):
    """`chunks`: {param: number of equal row chunks} (missing = 1, plain Muon). `rescale` multiplies each chunk's
    step by sqrt(chunks) (the paper's optional rescale, off by default)."""

    def __init__(self, params, lr: float = 1e-4, weight_decay: float = 0.0, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = DEFAULT_NS_STEPS, chunks=None, rescale: bool = False,
                 eps: float = EPS):
        if lr < 0 or not 0 <= momentum < 1:
            raise ValueError("CMuon needs lr >= 0 and 0 <= momentum < 1")
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, rescale=rescale, eps=eps))
        for g in self.param_groups:
            for p in g["params"]:
                if p.ndim != 2:
                    raise ValueError(f"CMuon trains 2D weights only (got shape {tuple(p.shape)})")
        self.chunks = {p: int(n) for p, n in (chunks or {}).items() if int(n) > 1 and p.shape[0] % int(n) == 0}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            lr, mom = g["lr"], g["momentum"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p)        # the weight's dtype: bf16 in a fine-tune
                buf = st["momentum_buffer"]
                m = buf.float().lerp_(grad, 1.0 - mom)
                buf.copy_(m)
                upd = grad.lerp(m, mom) if g["nesterov"] else m
                n = self.chunks.get(p, 1)
                parts = upd.chunk(n, dim=0) if n > 1 else (upd,)
                outs = []
                for c in parts:
                    o = _zeropower_via_newtonschulz(c, (DEFAULT_A, DEFAULT_B, DEFAULT_C), g["ns_steps"], g["eps"])
                    s = 0.2 * math.sqrt(max(c.shape)) * (math.sqrt(n) if g["rescale"] and n > 1 else 1.0)
                    outs.append(o.float() * s)
                step = torch.cat(outs, dim=0) if n > 1 else outs[0]
                if g["weight_decay"]:
                    p.mul_(1.0 - lr * g["weight_decay"])
                p.add_(step.to(p.dtype), alpha=-lr)
        return loss


__all__ = ["CMuon", "FUSED", "chunk_plan", "model_width"]
