"""Experimental SOAP memory strategies for the optimizer research queue.

ChunkedSOAP retains full state and tiles covariance updates only.
OffloadedSOAP retains full state and executes the original step on CPU.
CompactSOAP approximates covariance with truncated factors and represents the
orthogonal completion implicitly with Householder reflectors. Diagonal second
moments are transported through a compact relative Householder span on wide
axes (tiled reference on small axes). The old covariance is approximated as
independent in the old coordinates; off-diagonal history is not retained.
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
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self._ram_checked = False

    def _check_ram(self):
        # Conservative planning estimate, not an OS reservation. Full CPU
        # state is huge even if GPU memory is free.
        import psutil
        persistent, biggest = 0, 0
        for group in self.param_groups:
            for p in group["params"]:
                persistent += 8*p.numel()
                for d in p.shape:
                    if d <= group["max_precond_dim"]:
                        persistent += 8*d*d
                        biggest = max(biggest, d*d)
        required = persistent + 32*biggest + 2*1024**3
        available = psutil.virtual_memory().available
        if available < required:
            raise RuntimeError(f"SOAP offload RAM preflight: need about {required/1024**3:.1f} GiB "
                               f"available, found {available/1024**3:.1f} GiB")
        self._ram_checked = True

    @torch.no_grad()
    def step(self, closure=None):
        if not self._ram_checked:
            self._check_ram()
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


def _transport_axis(var, old, new, tile=32):
    """diag(R diag(v) R.T), R=Q_new.T @ Q_old, in tiles of new coordinates.

    var has the transformed axis first. Never materialize a full wide R.
    This is exact for a diagonal old covariance, not for the true covariance
    with cross-coordinate correlations. ORMQR's internal workspace is backend
    dependent, so the tile limits explicit tensors, not total CUDA allocations.
    """
    n = var.shape[0]
    result = torch.empty_like(var)
    for start in range(0, n, tile):
        end = min(n, start + tile)
        eye = torch.zeros((n, end-start), device=var.device, dtype=var.dtype)
        ids = torch.arange(end-start, device=var.device)
        eye[start+ids, ids] = 1.
        new_columns = torch.ormqr(new[0], new[1], eye, left=True, transpose=False)
        overlap = torch.ormqr(old[0], old[1], new_columns.contiguous(),
                              left=True, transpose=True)
        result[start:end] = overlap.square().T @ var
    return result


def _reflector_vectors(basis):
    packed, tau = basis
    vectors = torch.tril(packed, diagonal=-1)
    vectors.diagonal().fill_(1.)
    return vectors


def _transport_axis_lowrank(var, old, new, tile=32, columns=4):
    """Apply the same squared-overlap transform as the tiled reference.

    The relative orthogonal transform R=Q_new.T@Q_old fixes the orthogonal
    complement of the union of reflector spans. For an orthonormal basis U
    of that union, R-I=W@U.T with W=R@U-U. This gives the diagonal of
    R diag(v) R.T via small weighted Gram matrices. No wide square is stored.
    Use the nonnegative diagonal/off-diagonal split to limit cancellation.
    CPU parity is tested; CUDA workspace and runtime remain backend-dependent.
    """
    n = var.shape[0]
    k = old[1].numel() + new[1].numel()
    if 2*k >= n:
        return _transport_axis(var, old, new, tile)
    u, _ = torch.linalg.qr(torch.cat((_reflector_vectors(old),
                                      _reflector_vectors(new)), dim=1), mode="reduced")
    ru = torch.ormqr(old[0], old[1], u.contiguous(), left=True, transpose=False)
    ru = torch.ormqr(new[0], new[1], ru.contiguous(), left=True, transpose=True)
    w = ru-u
    diag = (w*u).sum(dim=1)
    result = torch.empty_like(var)
    for start in range(0, var.shape[1], columns):
        v = var[:, start:start+columns].T.contiguous()
        gram = u.T.unsqueeze(0) @ (u.unsqueeze(0)*v.unsqueeze(-1))
        quadratic = ((w.unsqueeze(0) @ gram)*w.unsqueeze(0)).sum(dim=-1)
        off = (quadratic-v*diag.square()).clamp_min_(0.)
        result[:, start:start+columns] = ((1+diag).square()*v+off).T
    return result


def _transport_variance(var, old, new, tile=32):
    out = _transport_axis_lowrank(var, old[0], new[0], tile)
    out = _transport_axis_lowrank(out.T.contiguous(), old[1], new[1], tile).T.contiguous()
    return out.clamp_min_(0.)


def _variance_floor(avg, b1, b2, age):
    """Cauchy bound for EMA history: repair impossible m/v combinations.

    For beta1**2 < beta2, m**2 <= v*C where C is the finite geometric
    sum of squared first-moment EMA weights divided by second-moment weights.
    Applying the bound after approximate diagonal transport avoids retaining
    large momentum with an artificially tiny denominator.
    """
    if not (0 < b2 < 1 and 0 <= b1 < 1):
        raise ValueError("CompactSOAP requires 0 < beta2 < 1 and 0 <= beta1 < 1")
    ratio = b1*b1/b2
    if ratio >= 1:
        raise ValueError("CompactSOAP requires beta1**2 < beta2")
    bound = (1-b1)**2 / (1-b2) * (1-ratio**age) / (1-ratio)
    return avg.square() / max(bound, 1e-30)


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
            if not (0 < b2 < 1 and 0 <= b1 < 1 and b1*b1 < b2):
                raise ValueError("CompactSOAP requires beta1**2 < beta2 and valid betas")
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
                    old_bases = s["bases"]
                    new_bases = [_reflectors(f) for f in s["factors"]]
                    original = _rotate(avg, old_bases, True)
                    new_avg = _rotate(original, new_bases)
                    new_var = _transport_variance(var, old_bases, new_bases)
                    floor = _variance_floor(new_avg, b1, b2, s["variance_age"])
                    repair = (floor-new_var).clamp_min_(0.)
                    # Device scalars only; avoid a synchronization per factor.
                    s["refresh_repaired_count"] = torch.count_nonzero(repair)
                    s["refresh_added_variance"] = repair.sum()
                    s["refresh_transported_variance"] = new_var.sum()
                    new_var = torch.maximum(new_var, floor)
                    s["bases"] = new_bases
                    s["exp_avg"] = new_avg
                    s["exp_avg_sq"] = new_var
        return loss
