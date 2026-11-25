"""Experimental scaled-gradient LoRA optimization, not production trainer code.

Inspired by Zhang & Pilanci (ICML 2024), arXiv:2402.02347.
This is a damped scaled-GD variant, not a reproduction of their full experiments.
No dense weight gradients or full adapter matrices; no live Adam moment rotation.
"""
from __future__ import annotations

import torch
from torch.optim import Optimizer
from fizgig.training.matrix_geometry import balanced_factors


@torch.no_grad()
def scaled_directions(a, b, ga, gb, damping=1e-3):
    """Solve opposite-factor damped Gram systems using pre-step factors.

    Positive absolute damping makes the zero-up initialization well-defined.
    Damping retains orthogonal equivariance but breaks general GL(r) invariance.
    """
    if damping <= 0:
        raise ValueError("positive damping required")
    dtype = torch.float64 if a.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=a.device.type, enabled=False):
        a,b,ga,gb = (v.to(dtype) for v in (a,b,ga,gb))
        eye = torch.eye(a.shape[0], device=a.device, dtype=dtype)
        da = torch.linalg.solve(b.T @ b + damping*eye, ga)
        db = torch.linalg.solve(a @ a.T + damping*eye, gb.T).T
    return da,db


class ScaledLoRASGD(Optimizer):
    """Stateless pair optimizer for synthetic tests only.

    Optional spectral balancing is restricted to full-rank pairs and performed
    after the step. It changes future gradient dynamics, not the intended current
    product. Do not use with rank dropout, EMA or low-precision live weights.
    """
    def __init__(self, pairs, lr=1e-2, damping=1e-3, scaled=True,
                 balance_every=0):
        if lr <= 0 or damping <= 0 or balance_every < 0:
            raise ValueError("invalid learning rate, damping or balance interval")
        groups=[]
        for a,b in pairs:
            if (a.ndim!=2 or b.ndim!=2 or a.shape[0]!=b.shape[1]
                    or a.shape[0]>min(a.shape[1],b.shape[0])
                    or a.dtype not in (torch.float32,torch.float64) or a.dtype!=b.dtype
                    or a.device!=b.device):
                raise ValueError("compatible FP32/FP64 2D LoRA pairs required")
            groups.append({"params":[a,b], "pair_steps":0})
        super().__init__(groups,dict(lr=lr,damping=damping,scaled=scaled,
                                    balance_every=int(balance_every)))

    @torch.no_grad()
    def step(self, closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():
                loss=closure()
        for g in self.param_groups:
            a,b=g["params"]
            if a.grad is None or b.grad is None:
                continue
            if g["scaled"]:
                da,db=scaled_directions(a,b,a.grad,b.grad,g["damping"])
            else:
                da,db=a.grad,b.grad
            a.add_(da,alpha=-g["lr"])
            b.add_(db,alpha=-g["lr"])
            g["pair_steps"]+=1
            period=g["balance_every"]
            if period and g["pair_steps"]%period==0:
                # Avoid collapsing latent dimensions at a singular point.
                if (torch.linalg.matrix_rank(a)==a.shape[0] and
                    torch.linalg.matrix_rank(b)==a.shape[0]):
                    an,bn=balanced_factors(a,b)
                    a.copy_(an)
                    b.copy_(bn)
        return loss
