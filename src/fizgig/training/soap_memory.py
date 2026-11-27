"""Experimental SOAP memory strategies for the optimizer research queue.

ChunkedSOAP retains full state and tiles covariance updates only.
OffloadedSOAP retains full state and executes the original step on CPU.
CompactSOAP approximates covariance with truncated factors and represents the
orthogonal completion implicitly with Householder reflectors. Its second moment
is reset when the coordinate system changes, rather than pretending it is exact.
No CUDA fit or quality guarantees. Based on vendored MIT SOAP.
"""
import math
import torch
from fizgig.training.soap import SOAP


class ChunkedSOAP(SOAP):
    def __init__(self, params, *, chunk_rows=128, **kwargs):
        super().__init__(params, **kwargs)
        self.chunk_rows = int(chunk_rows)
        if self.chunk_rows < 1:
            raise ValueError("chunk_rows must be positive")

    def update_preconditioner(self, grad, state, max_precond_dim=10000,
                              merge_dims=False, precondition_1d=False):
        if grad.ndim != 2 or merge_dims:
            return super().update_preconditioner(grad, state, max_precond_dim,
                                                  merge_dims, precondition_1d)
        if state["Q"] is not None:
            state["exp_avg"] = self.project_back(state["exp_avg"], state)
        beta = state["shampoo_beta"]
        for axis, covariance in enumerate(state["GG"]):
            if not len(covariance):
                continue
            x = (grad if axis == 0 else grad.T).float()
            for start in range(0, x.shape[0], self.chunk_rows):
                stop = min(start + self.chunk_rows, x.shape[0])
                covariance[start:stop].lerp_(x[start:stop] @ x.T, 1-beta)
        if state["Q"] is None:
            state["Q"] = self.get_orthogonal_matrix(state["GG"])
        elif state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0:
            state["Q"] = self.get_orthogonal_matrix_QR(state, max_precond_dim, False)
        if state["step"] > 0:
            state["exp_avg"] = self.project(state["exp_avg"], state)


class OffloadedSOAP(SOAP):
    """CPU state and CPU arithmetic, one factor at a time. Research fresh runs only."""
    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        groups = self.param_groups
        try:
            for group in groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    # FP32 CPU arithmetic, independent of accelerator storage dtype.
                    local = torch.nn.Parameter(p.detach().to("cpu", torch.float32, copy=True))
                    local.grad = p.grad.detach().to("cpu", torch.float32, copy=True)
                    self.state[local] = self.state[p]
                    self.param_groups = [{**group, "params": [local]}]
                    try:
                        super().step()
                        p.copy_(local)
                    finally:
                        self.state.pop(local, None)
        finally:
            self.param_groups = groups
        return loss


def _covariance_factor(old, x, beta, rank):
    """Approximate beta*old@old.T + (1-beta)*x@x.T, without a wide square."""
    x = x.float()
    if x.shape[0] <= rank:
        cov = (x @ x.T) * (1-beta)
        if old is not None:
            cov.addmm_(old, old.T, beta=1., alpha=beta)
        eig, u = torch.linalg.eigh(cov)
        return u.flip(1) * eig.flip(0).clamp_min(0).sqrt()
    stack = x * math.sqrt(1-beta)
    if old is not None:
        stack = torch.cat((old * math.sqrt(beta), stack), dim=1)
    # For LoRA's large axis, stack is thin; QR reduces the SVD to a small core.
    q, r = torch.linalg.qr(stack, mode="reduced")
    u, sigma, _ = torch.linalg.svd(r, full_matrices=False)
    k = min(rank, sigma.numel())
    return (q @ u[:, :k]) * sigma[:k]


def _reflectors(factor):
    # Columns are ordered covariance eigenvectors times singular values.
    # QR encodes an orthogonal completion without storing a dense square.
    return torch.geqrf(factor.contiguous())


def _rotate(x, bases, inverse=False):
    out = x
    for axis, (reflectors, tau) in enumerate(bases):
        if axis == 0:
            out = torch.ormqr(reflectors, tau, out.contiguous(),
                              left=True, transpose=not inverse)
        else:
            out = torch.ormqr(reflectors, tau, out.contiguous(),
                              left=False, transpose=inverse)
    return out


class CompactSOAP(SOAP):
    """Rank-limited covariance, implicit full orthogonal basis, FP32 moments."""
    def __init__(self, params, *, compact_rank=64, **kwargs):
        super().__init__(params, **kwargs)
        self.compact_rank = int(compact_rank)
        if self.compact_rank < 1:
            raise ValueError("compact_rank must be positive")

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        for group in self.param_groups:
            if group["merge_dims"] or group["normalize_grads"]:
                raise ValueError("CompactSOAP pilot supports ordinary 2D LoRA factors only")
            b1, b2 = group["betas"]
            beta = group["shampoo_beta"] if group["shampoo_beta"] >= 0 else b2
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                if g.ndim != 2:
                    raise ValueError("CompactSOAP requires 2D LoRA factors")
                s = self.state[p]
                if "factors" not in s:
                    s["factors"] = [_covariance_factor(None, x, beta, self.compact_rank)
                                    for x in (g, g.T)]
                    s["bases"] = [_reflectors(f) for f in s["factors"]]
                    s["exp_avg"] = torch.zeros_like(g)
                    s["exp_avg_sq"] = torch.zeros_like(g)
                    s["step"] = s["variance_age"] = 0
                    continue
                s["step"] += 1
                s["variance_age"] += 1
                projected = _rotate(g, s["bases"])
                avg, var = s["exp_avg"], s["exp_avg_sq"]
                avg.mul_(b1).add_(projected, alpha=1-b1)
                var.mul_(b2).addcmul_(projected, projected, value=1-b2)
                rate = group["lr"]
                if group["correct_bias"]:
                    rate *= math.sqrt(1-b2**s["variance_age"]) / (1-b1**s["step"])
                direction = _rotate(avg / (var.sqrt() + group["eps"]), s["bases"], True)
                p.add_(direction.to(p.dtype), alpha=-rate)
                p.mul_(1-group["lr"]*group["weight_decay"])
                s["factors"] = [_covariance_factor(old, x, beta, self.compact_rank)
                                for old, x in zip(s["factors"], (g, g.T))]
                if s["step"] % group["precondition_frequency"] == 0:
                    original = _rotate(avg, s["bases"], True)
                    s["bases"] = [_reflectors(f) for f in s["factors"]]
                    s["exp_avg"] = _rotate(original, s["bases"])
                    # Coordinate-wise variance cannot be rotated exactly without
                    # cross-covariance. Explicitly restart it on a basis change.
                    s["exp_avg_sq"].zero_()
                    s["variance_age"] = 0
        return loss
