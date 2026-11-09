"""A frozen copy of BatchedMuon's step as it was before its Nesterov look-ahead became a foreach op (Fizgig
commit 045ba6e4): checks/test_muon_batched_lerp.py runs the new one against it. Test reference only, never imported by
Fizgig. Adapted from PyTorch's torch.optim.Muon, BSD-3-Clause (docs/THIRD_PARTY_NOTICES.md)."""
import torch

from fizgig.training.muon import _adjust_lr, _zeropower_batched, DEFAULT_A, DEFAULT_B, DEFAULT_C, EPS, DEFAULT_NS_STEPS


class BatchedMuonReference(torch.optim.Optimizer):
    """torch.optim.Muon's update (same momentum, Newton-Schulz and adjust_lr rules) for all-2D parameter lists, with
    the per-parameter work batched: momentum and weight decay as foreach ops, Newton-Schulz once per group of
    same-shaped matrices."""

    def __init__(self, params, lr: float = 1e-3, weight_decay: float = 0.1, momentum: float = 0.95,
                 nesterov: bool = True, ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
                 eps: float = EPS, ns_steps: int = DEFAULT_NS_STEPS, adjust_lr_fn: str | None = None) -> None:
        if adjust_lr_fn is not None and adjust_lr_fn not in ("original", "match_rms_adamw", "spectral_unclamped"):
            raise ValueError(f"Adjust learning rate function {adjust_lr_fn} is not supported")
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                                      ns_coefficients=ns_coefficients, eps=eps, ns_steps=ns_steps,
                                      adjust_lr_fn=adjust_lr_fn))
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"BatchedMuon takes 2D parameters only, got {tuple(p.shape)}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd, mom = float(group["lr"]), float(group["weight_decay"]), float(group["momentum"])
            ps = [p for p in group["params"] if p.grad is not None]
            if not ps:
                continue
            grads = [p.grad for p in ps]
            bufs = []
            for p, g in zip(ps, grads):
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g, memory_format=torch.preserve_format)
                bufs.append(st["momentum_buffer"])
            torch._foreach_lerp_(bufs, grads, 1.0 - mom)
            ups = [g.lerp(b_, mom) for g, b_ in zip(grads, bufs)] if group["nesterov"] else bufs
            if wd != 0.0:
                torch._foreach_mul_(ps, 1.0 - lr * wd)
            by_shape = {}
            for i, p in enumerate(ps):
                by_shape.setdefault((tuple(p.shape), p.device, p.dtype), []).append(i)
            for (shape, _dev, dtype), idx in by_shape.items():
                ortho = _zeropower_batched(torch.stack([ups[i] for i in idx]), group["ns_coefficients"],
                                           int(group["ns_steps"]), float(group["eps"]))
                alr = _adjust_lr(lr, group["adjust_lr_fn"], shape)
                torch._foreach_add_([ps[i] for i in idx], list(ortho.to(dtype).unbind(0)), alpha=-alr)
        return loss
