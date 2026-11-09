"""PoLoRA — spectral-preconditioned LoRA optimizer.

Polora adapts each LoRA factor pair ``(A, B)`` (PEFT convention: ``A`` is
``(r, d_in)``, ``B`` is ``(d_out, r)``, adapter output ``(alpha/r) * B @ A``)
with a preconditioned polar step in a diagonal-Kronecker curvature metric:

  1. Build the ``r x r`` matrices ``C_A = A Q A^T`` and ``C_B = B^T P B``
     from damped, max-normalized versions of the diagonal preconditioners
     ``Q`` / ``P``.
  2. Whiten the Nesterov-corrected momentum by the inverse square root of
     ``C_B`` (resp. ``C_A``), computed eigh-free by Gram Newton-Schulz, apply
     the matrix sign, then unwhiten.
  3. Rescale each factor update to spectral norm
     ``rho = lr / (sigma_max(A) + sigma_max(B))``.

The preconditioners are coupled across the two sides and kept diagonal.
Stored state per pair is the first moment plus two diagonal vectors (``Q`` of
length ``d_in``, ``P`` of length ``d_out``) — about half of Adam's state; the
``r x r`` matrices ``C_A``/``C_B`` are rebuilt from the weights each step, not stored.

``Polora(model, lr=...)`` constructs directly from a PEFT-wrapped model;
non-PEFT use is supported via ``Polora(pairs=[(A, B), ...], lr=...)``.
"""
# Modified for Fizgig: one console line per logged step, state_dict() carries the pair state so a
# resume keeps it, math imported at the top. The pair state is kept stacked per shape group between steps (no per-pair
# stacking and copying back), the factor updates and grad zeroing are written back with foreach ops, and the four
# sigma_max power iterations of a group run as two batched calls (compiled with the other kernels when compile=True).
# The update itself is unchanged (checks/test_polora_stacked.py runs it against the per-pair step it replaced).
from __future__ import annotations

import logging
import math
from collections import defaultdict

import torch
from torch.optim import Optimizer

from .utils import polar_express_gram_batched, ns_inv_sqrt, power_iter_top

# the sigma_max warm-start vectors per pair, as state_dict() names them
_VKEYS = ("v_sigma_A", "v_sigma_DA", "v_sigma_B", "v_sigma_DB")


def _smax_floored(M, v, n_iters=8):
    """``sigma_max`` of a batch by warm-started power iteration, floored at ``max(row L2, col L2)`` - a lower bound
    on ``sigma_max`` - so a cold or degenerate start vector cannot under-estimate the top value. A zero ``v`` row
    starts cold (``M @ 1``), as no warm start does."""
    s, v = power_iter_top(M, v_init=v, n_iters=n_iters)
    Mf = M.float()
    rn = Mf.pow(2).sum(dim=-1).amax(dim=-1).sqrt()
    cn = Mf.pow(2).sum(dim=-2).amax(dim=-1).sqrt()
    return torch.maximum(s, torch.maximum(rn, cn).reshape(s.shape).to(s.dtype)), v


def smax_groups(WA, vA, WB, vB, n_iters=8):
    """A shape group's four ``sigma_max`` in two batched calls: ``WA`` stacks ``A`` over ``D_A`` and ``WB`` stacks
    ``B`` over ``D_B``, each with its warm-start vectors. Returns ``(sigma_WA, v_WA, sigma_WB, v_WB)``."""
    sA, vA = _smax_floored(WA, vA, n_iters)
    sB, vB = _smax_floored(WB, vB, n_iters)
    return sA, vA, sB, vB


class _Group:
    """One shape group's stacked state: the pairs' params and, stacked along dim 0, the momentum, the Q / P
    diagonals and the sigma_max warm-start vectors (``vA`` = A's over D_A's, ``vB`` = B's over D_B's)."""
    __slots__ = ("idxs", "As", "Bs", "f32", "M_A", "M_B", "Q", "P", "vA", "vB")


def collect_lora_pairs(model):
    """Discovers the trainable LoRA ``(A, B)`` weight tensors on a PEFT model.

    Pairs whose factors are frozen (``requires_grad=False``) are skipped, so a
    frozen adapter sharing the model (e.g. a DPO reference adapter) is not
    collected.

    Args:
        model: A PEFT-wrapped model (its modules carry ``lora_A`` / ``lora_B``).

    Returns:
        List of ``(A, B)`` parameter pairs — ``A`` is ``(r, in)``, ``B`` is
        ``(out, r)`` — in module-discovery order.
    """
    pairs = []
    for _, mod in model.named_modules():
        if not (hasattr(mod, "lora_A") and hasattr(mod, "lora_B")):
            continue
        lora_A, lora_B = mod.lora_A, mod.lora_B
        if hasattr(lora_A, "keys"):  # ModuleDict of adapters (standard PEFT)
            for k in lora_A:
                if k in lora_B:
                    pairs.append((lora_A[k].weight, lora_B[k].weight))
        elif hasattr(lora_A, "weight") and hasattr(lora_B, "weight"):
            pairs.append((lora_A.weight, lora_B.weight))
    return [(A, B) for A, B in pairs if A.requires_grad and B.requires_grad]


class Polora(Optimizer):
    """Spectral-preconditioned LoRA optimizer.

    Each LoRA pair is whitened in a diagonal-Kronecker curvature metric, passed
    through the matrix sign, unwhitened, and rescaled to spectral norm
    ``rho``. See the module docstring for the full update.

    Construct after the model is on its target device; optimizer state is
    allocated on the params' current device. ``state_dict()`` carries the
    momentum/preconditioner state and step count (key ``"polora"``), so a
    resume with the same LoRA layout picks them up.

    Args:
        model: PEFT-wrapped model; ``(A, B)`` adapter pairs are discovered
            automatically. Provide either ``model`` or ``pairs``.
            (Default: None)
        lr: Learning rate; each factor update is rescaled to spectral norm
            ``rho = lr / (sigma_max(A) + sigma_max(B))``. (Default: 2e-4)
        beta1: Momentum coefficient. (Default: 0.9)
        epsilon: Numerical floor for the preconditioner initialization, the
            damping of the ``C_A``/``C_B`` inverse square roots, and the
            spectral-norm normalizations. (Default: 1e-12)
        delta: Relative damping for the ``C_A``/``C_B`` inverse square roots
            and for the normalized ``Q``/``P`` inverse square roots.
            (Default: 1e-4)
        curvature_beta: EMA coefficient for the diagonal preconditioners
            ``Q``/``P``. (Default: 0.99)
        ns_steps: Iterations of the PolarExpress matrix-sign solver. (Default: 8)
        higham_iters: Newton-Schulz iterations for the ``C_A``/``C_B``
            inverse square roots. (Default: 8)
        compile: If True, wraps the two spectral kernels in ``torch.compile``;
            the update is unchanged. (Default: False)
        pairs: Explicit ``[(A, B), ...]`` parameter list for non-PEFT use.
            (Default: None)
    """

    def __init__(self, model=None, lr=2e-4, beta1=0.9, epsilon=1e-12,
                 delta=1e-4, curvature_beta=0.99, ns_steps=8, higham_iters=8,
                 compile=False, magnitude_scale=1.0, decay_start=1.0,
                 decay_type="none", log_norms_every=0, *, pairs=None):
        if pairs is None:
            if model is None:
                raise ValueError("Polora requires either `model` (PEFT) or `pairs=[(A,B),...]`.")
            pairs = collect_lora_pairs(model)
        else:
            pairs = list(pairs)
        if not pairs:
            raise ValueError("No trainable LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.beta1 = beta1
        self.curvature_beta = float(curvature_beta)
        self.ns_steps = int(ns_steps)
        self.higham_iters = int(higham_iters)
        self.magnitude_scale = float(magnitude_scale)
        self.decay_start = float(decay_start)
        self.decay_type = str(decay_type).lower()
        self.log_norms_every = int(log_norms_every)
        self._step_count = 0
        self._grad_norm_ema = None
        self._timestep_scale = 1.0
        # Per-step multipliers on magnitude_scale set by the training loop (Fizgig): the per-image loss watch and
        # Adaptive LR. effective magnitude = magnitude_scale * per_image * adaptive; 1.0 each = unchanged.
        self._image_factor = 1.0
        self._adaptive_factor = 1.0
        self.record_every = 0           # Fizgig's loss graph: also measure the factor sizes on these steps
        self.last_norms = None          # {"step", "sigma_A", "sigma_B", "magnitude"} from the last measured step
        # dynamic=True lets the LoRA shape groups share one compiled kernel. A compile that cannot run (Windows
        # without triton, an old torch) falls back to the plain kernels with one line, at the first use.
        self._polar_fn = polar_express_gram_batched
        self._invsqrt_fn = ns_inv_sqrt
        self._smax_fn = smax_groups
        self._compiled = False
        if compile:
            self._compiled = self._try_compile()

        # The pair state lives stacked per shape group (_Group); pair_state[i] holds views into it under the
        # per-pair names, so state_dict() / load_state_dict() keep their per-pair format.
        self._groups = []
        self._layout = None
        self.pair_state = {}
        self._build_groups()

    def _layout_now(self):
        return [(tuple(A.shape), tuple(B.shape), A.device, A.dtype, B.dtype) for A, B in self.pairs]

    def _build_groups(self, layout=None):
        """(Re)builds the stacked state and the group membership - at construction, and again only if a pair's
        shape, device or dtype changes. A pair whose state still fits keeps it; anything else starts as at
        construction (zero momentum, Q / P at epsilon, no warm start)."""
        layout = layout or self._layout_now()
        old = self.pair_state
        by = defaultdict(list)
        for i, (shA, shB, dev, dtA, dtB) in enumerate(layout):
            by[(shA, shB, dev, dtA, dtB)].append(i)
        groups, state = [], {}
        f32 = torch.float32
        for (shA, shB, dev, dtA, dtB), idxs in by.items():
            (r, d_in), d_out, n = shA, shB[0], len(idxs)
            g = _Group()
            g.idxs = idxs
            g.As = [self.pairs[i][0] for i in idxs]
            g.Bs = [self.pairs[i][1] for i in idxs]
            g.f32 = dtA == f32 and dtB == f32
            g.M_A = torch.zeros((n, r, d_in), dtype=f32, device=dev)
            g.M_B = torch.zeros((n, d_out, shB[1]), dtype=f32, device=dev)
            g.Q = torch.full((n, d_in), self.epsilon, dtype=f32, device=dev)
            g.P = torch.full((n, d_out), self.epsilon, dtype=f32, device=dev)
            # zero = no warm start yet: power_iter_top restarts a zero vector from M @ 1, the cold start
            g.vA = torch.zeros((2 * n, min(r, d_in)), dtype=f32, device=dev)
            g.vB = torch.zeros((2 * n, min(d_out, shB[1])), dtype=f32, device=dev)
            for j, i in enumerate(idxs):
                st = state[i] = {"M_A": g.M_A[j], "M_B": g.M_B[j], "Q": g.Q[j], "P": g.P[j],
                                 "v_sigma_A": g.vA[j], "v_sigma_DA": g.vA[n + j],
                                 "v_sigma_B": g.vB[j], "v_sigma_DB": g.vB[n + j]}
                prev = old.get(i)
                if prev is not None and prev["M_A"].shape == st["M_A"].shape and prev["M_B"].shape == st["M_B"].shape:
                    for k, v in st.items():
                        if torch.is_tensor(prev.get(k)) and prev[k].shape == v.shape:
                            v.copy_(prev[k])
            groups.append(g)
        self._groups, self.pair_state, self._layout = groups, state, layout

    def _try_compile(self):
        log = logging.getLogger("fizgig.polora")
        dev = self.pairs[0][0].device
        if dev.type == "cuda":
            try:
                import triton  # noqa: F401
            except Exception:
                log.info("[polora] compile is off: triton is not installed (usual on Windows) - running uncompiled")
                return False
        try:
            self._polar_fn = torch.compile(polar_express_gram_batched, dynamic=True, fullgraph=False)
            self._invsqrt_fn = torch.compile(ns_inv_sqrt, dynamic=True, fullgraph=False)
            self._smax_fn = torch.compile(smax_groups, dynamic=True, fullgraph=False)
        except Exception as e:
            self._polar_fn, self._invsqrt_fn, self._smax_fn = polar_express_gram_batched, ns_inv_sqrt, smax_groups
            log.info("[polora] compile is off: %s - running uncompiled", str(e).splitlines()[0][:200] if str(e) else type(e).__name__)
            return False
        log.info("[polora] compiling its polar, inverse-square-root and sigma_max kernels (the first steps are slower)")
        return True

    def _kernel(self, name, *args, **kw):
        """Run a compiled kernel; if it fails (compile errors surface on the first call), switch to the plain
        kernels for good, with one line."""
        fn = getattr(self, name)
        if not self._compiled:
            return fn(*args, **kw)
        try:
            return fn(*args, **kw)
        except Exception as e:
            self._compiled = False
            self._polar_fn, self._invsqrt_fn, self._smax_fn = polar_express_gram_batched, ns_inv_sqrt, smax_groups
            msg = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
            logging.getLogger("fizgig.polora").warning("[polora] compile failed (%s) - running uncompiled from here", msg)
            return getattr(self, name)(*args, **kw)

    @torch.no_grad()
    def set_magnitude_factors(self, image=None, adaptive=None):
        """Fizgig: the per-image loss-watch factor (this step) and the Adaptive LR factor, each multiplying
        magnitude_scale. None leaves that one as it is."""
        if image is not None:
            self._image_factor = float(image)
        if adaptive is not None:
            self._adaptive_factor = float(adaptive)

    def effective_magnitude(self):
        return self.magnitude_scale * self._image_factor * self._adaptive_factor

    @staticmethod
    def _sym(M):
        return 0.5 * (M + M.transpose(-2, -1))

    def _rdinv(self, x):
        """Relative-damped inverse square root along the last dim:
        ``(x / x_max + delta)^{-1/2}``. ``x_max > 0`` from step 1 because the
        preconditioner EMA is seeded at ``epsilon``."""
        xmax = x.amax(dim=-1, keepdim=True)
        return (x / xmax + self.delta).rsqrt()

    def _polar_batched(self, X):
        """Matrix sign via PolarExpress (Frobenius-prescaled internally)."""
        return self._kernel("_polar_fn", X, nsteps=self.ns_steps)

    @torch.no_grad()
    def set_timestep_scale(self, scale):
        """Optional: training loop can call this with a scale based on the
        current noise level / timestep. 1.0 = no change. Values > 1 push harder
        on easy timesteps; values < 1 are more conservative on hard ones."""
        self._timestep_scale = float(scale)

    def _decay_factor(self):
        """Optional linear or cosine decay that only applies after decay_start
        fraction of a nominal horizon. The optimizer does not know the total
        steps, so this is a simple step-count based decay that the user can
        tune. decay_start=1.0 (default) means no decay."""
        if self.decay_type == "none" or self.decay_start >= 1.0:
            return 1.0
        # Simple decay over a long horizon; user can adjust decay_start
        # to control when it begins. This is a relative decay, not absolute.
        t = self._step_count
        # Use a soft horizon of 10000 steps for the decay calculation
        horizon = 10000
        start = int(self.decay_start * horizon)
        if t < start:
            return 1.0
        progress = min(1.0, (t - start) / max(1, horizon - start))
        if self.decay_type == "linear":
            return 1.0 - 0.5 * progress  # decay to 50%
        if self.decay_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))  # decay to 0
        return 1.0

    def step(self, closure=None):
        """Applies one Polora update to every LoRA pair, then zeros their grads.

        Gradients on all ``(A, B)`` factors must already be populated; raises
        ``ValueError`` otherwise. See the module docstring for the update.

        Args:
            closure: Callable that re-evaluates the model and returns the loss
                (standard ``torch.optim`` protocol); called under grad.
                (Default: None)

        Returns:
            The loss returned by ``closure``, or None if no closure is given.
        """
        self._step_count += 1
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for A, B in self.pairs:
            if A.grad is None or B.grad is None:
                raise ValueError("Gradients are required for Polora update.")
        layout = self._layout_now()
        if layout != self._layout:
            self._build_groups(layout)
        with torch.no_grad():
            self._update()
        return loss

    def _update(self):
        lr = self.param_groups[0]["lr"]
        cb = self.curvature_beta
        b1, eps = self.beta1, self.epsilon
        f32 = torch.float32
        log_now = bool(self.log_norms_every) and self._step_count % self.log_norms_every == 0
        record_now = bool(self.record_every) and self._step_count % self.record_every == 0
        magnitude = self.effective_magnitude()
        scale = magnitude * self._timestep_scale * self._decay_factor()
        sig_A, sig_B = [], []
        # One batched bmm / Newton-Schulz / sigma_max per (r, d_in, d_out, device, dtype) shape group.
        for g in self._groups:
            n = len(g.idxs)
            gA, gB = [A.grad for A in g.As], [B.grad for B in g.Bs]
            G_A = torch.stack(gA) if g.f32 else torch.stack(gA).float()
            G_B = torch.stack(gB) if g.f32 else torch.stack(gB).float()
            # A over D_A (and B over D_B) in one tensor: their sigma_max run as one batched call
            WA = torch.empty((2 * n, *g.M_A.shape[1:]), dtype=f32, device=G_A.device)
            WB = torch.empty((2 * n, *g.M_B.shape[1:]), dtype=f32, device=G_B.device)
            Aw, Bw = WA[:n], WB[:n]
            if g.f32:
                torch.stack(g.As, out=Aw)
                torch.stack(g.Bs, out=Bw)
            else:
                Aw.copy_(torch.stack(g.As))
                Bw.copy_(torch.stack(g.Bs))
            M_A = g.M_A.mul_(b1).add_(G_A, alpha=1.0 - b1)
            M_B = g.M_B.mul_(b1).add_(G_B, alpha=1.0 - b1)
            Q, P = g.Q, g.P                                  # d_in / d_out metric diagonals

            Q_isqrt = self._rdinv(Q)
            P_isqrt = self._rdinv(P)
            Q_dmp = (Q_isqrt * Q_isqrt).reciprocal()
            P_dmp = (P_isqrt * P_isqrt).reciprocal()
            # Curvature factors, rebuilt each step (not stored).
            C_B = Bw.transpose(-2, -1) @ (P_dmp.unsqueeze(-1) * Bw)
            C_A = (Aw * Q_dmp.unsqueeze(1)) @ Aw.transpose(-2, -1)
            C_B_isqrt = self._kernel("_invsqrt_fn", self._sym(C_B), nsteps=self.higham_iters,
                                     eps=self.delta, eps_relative=True, floor=self.epsilon)
            C_A_isqrt = self._kernel("_invsqrt_fn", self._sym(C_A), nsteps=self.higham_iters,
                                     eps=self.delta, eps_relative=True, floor=self.epsilon)
            C_B_inv = C_B_isqrt @ C_B_isqrt
            C_A_inv = C_A_isqrt @ C_A_isqrt

            # Nesterov look-ahead.
            M_hat_A = M_A.mul(b1).add(G_A, alpha=1.0 - b1)
            M_hat_B = M_B.mul(b1).add(G_B, alpha=1.0 - b1)

            # Whiten -> polar -> unwhiten (D_A / D_B written under A / B for the sigma_max call).
            zA = (C_B_isqrt @ M_hat_A) * Q_isqrt.unsqueeze(1)
            zB = (M_hat_B @ C_A_isqrt) * P_isqrt.unsqueeze(-1)
            zA = self._polar_batched(zA)
            zB = self._polar_batched(zB)
            D_A = torch.mul(C_B_isqrt @ zA, Q_isqrt.unsqueeze(1), out=WA[n:])
            D_B = torch.mul(zB @ C_A_isqrt, P_isqrt.unsqueeze(-1), out=WB[n:])

            sWA, vA, sWB, vB = self._kernel("_smax_fn", WA, g.vA, WB, g.vB)
            g.vA.copy_(vA)
            g.vB.copy_(vB)
            sigma_A, sigma_DA = sWA[:n], sWA[n:]
            sigma_B, sigma_DB = sWB[:n], sWB[n:]
            # Magnitude scale + optional timestep scale + optional decay
            rho = lr / (sigma_B + sigma_A)
            rho = rho * magnitude * self._timestep_scale * self._decay_factor()
            if log_now or record_now:
                sig_A.append(sigma_A.reshape(-1))
                sig_B.append(sigma_B.reshape(-1))
            dA = -(rho / sigma_DA.clamp_min(eps)).view(-1, 1, 1) * D_A
            dB = -(rho / sigma_DB.clamp_min(eps)).view(-1, 1, 1) * D_B

            # Preconditioner update: each diagonal accumulates the gradient
            # whitened by the inverse of the other side's r x r matrix.
            r = G_A.shape[1]
            Q.mul_(cb).add_((G_A * (C_B_inv @ G_A)).sum(dim=1), alpha=(1.0 - cb) / r)
            P.mul_(cb).add_((G_B * (G_B @ C_A_inv)).sum(dim=2), alpha=(1.0 - cb) / r)

            # write back: every pair of the group at once, then its grads zeroed
            if not g.f32:
                dA, dB = dA.to(g.As[0].dtype), dB.to(g.Bs[0].dtype)
            torch._foreach_add_(g.As, dA.unbind(0))
            torch._foreach_add_(g.Bs, dB.unbind(0))
            torch._foreach_zero_(gA + gB)

        if log_now or record_now:       # one sync for both averages
            sa, sb = torch.stack([torch.cat(sig_A).float().mean(), torch.cat(sig_B).float().mean()]).tolist()
            self.last_norms = {"step": self._step_count, "sigma_A": sa, "sigma_B": sb, "magnitude": magnitude}
        if log_now:                     # one line over every pair, not one per shape group
            logging.getLogger("fizgig.polora").info(
                "[polora] step %d: avg sigma_A %.4f, avg sigma_B %.4f, magnitude %.3f (set %.3g x image %.2f x adaptive %.2f), rho_scale %.3f",
                self._step_count, sa, sb, magnitude, self.magnitude_scale, self._image_factor, self._adaptive_factor,
                scale)

    def state_dict(self):
        sd = super().state_dict()
        sd["polora"] = {"step": self._step_count, "adaptive_factor": self._adaptive_factor,
                        "pairs": {i: {k: v.detach().cpu().clone() for k, v in st.items() if torch.is_tensor(v)}
                                  for i, st in self.pair_state.items()}}
        return sd

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        extra = state_dict.pop("polora", None)
        super().load_state_dict(state_dict)
        if not extra:
            return
        saved = extra.get("pairs") or {}
        if len(saved) != len(self.pair_state) or any(
                saved[i].get("M_A") is None or saved[i]["M_A"].shape != self.pair_state[i]["M_A"].shape
                or saved[i]["M_B"].shape != self.pair_state[i]["M_B"].shape for i in self.pair_state):
            logging.getLogger("fizgig.polora").info(
                "[polora] saved state does not match this LoRA's layout: starting its momentum fresh")
            return
        # into the stacked state (pair_state holds views of it); a save without warm-start vectors (one made before
        # the first step) leaves them at zero, the cold start
        for i, st in self.pair_state.items():
            for k, v in saved[i].items():
                if k in st and torch.is_tensor(v) and v.shape == st[k].shape:
                    st[k].copy_(v)
        self._step_count = int(extra.get("step", 0))
        self._adaptive_factor = float(extra.get("adaptive_factor", 1.0))
