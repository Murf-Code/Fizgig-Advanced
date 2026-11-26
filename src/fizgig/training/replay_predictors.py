"""6F.3: observational predictors only; no candidate interventions or optimizer changes."""
import math
from contextlib import nullcontext

import torch

from .progress import DiagnosticProgress
from .replay_diagnostics import digest
from .replay_isolation import precision


@torch.no_grad()
def activation_metrics(x, y):
    # FP32 differences; double scalar accumulation. Full combined token stream.
    x, y = x.detach().float(), y.detach().float()
    if x.shape != y.shape:
        raise ValueError("block activation shapes differ")
    xx = float(x.square().sum(dtype=torch.float64))
    yy = float(y.square().sum(dtype=torch.float64))
    xy = float((x*y).sum(dtype=torch.float64))
    rr = float((y-x).square().sum(dtype=torch.float64))
    if not all(math.isfinite(v) for v in (xx, yy, xy, rr)):
        raise ValueError("nonfinite activation predictor")
    return {"input_output_cosine": max(-1., min(1., xy/math.sqrt(xx*yy))) if xx and yy else None,
            "residual_ratio": math.sqrt(rr/xx) if xx else None}


@torch.no_grad()
def collect_activations(dit, net, panel, predict, guard, step, precision_context=precision):
    progress = DiagnosticProgress(f"6F.3 update {step} activation panels", 2*len(panel))
    result = {}
    for mode in ("native", "weight_only"):
        rows = []
        current = {}
        handles = []
        def hook(index):
            def observe(module, inputs, output):
                if index in current:
                    raise RuntimeError("duplicate block forward")
                current[index] = activation_metrics(inputs[0], output)
            return observe
        def evaluate():
            for case, inputs, _ in panel:
                current.clear()
                progress.begin(f"{mode} · {case.unit} · t={case.timestep}")
                out = predict(inputs)
                if set(current) != set(range(len(dit.blocks))):
                    raise RuntimeError("incomplete predictor block forward")
                rows.append({"unit": case.unit, "timestep": case.timestep,
                             "input_hash": case.input_hash, "prediction_hash": digest(out),
                             "blocks": dict(current)})
                progress.advance(f"{mode} case complete")
        with precision_context(net, True, False) if mode == "weight_only" else nullcontext():
            try:
                for i, block in enumerate(dit.blocks):
                    handles.append(block.register_forward_hook(hook(i)))
                guard(evaluate)
            finally:
                for handle in handles:
                    handle.remove()
        result[mode] = rows
    return result


class UpdateObserver:
    """Observe the actual LR-scaled additive Muon proposal, excluding decay.
    Aggregate inner products over concatenated A/B factor coordinates per block.
    Positive descent alignment means the learning update opposes the gradient.
    No copies are retained after the callback, no optimizer state is changed.
    """
    def __init__(self, pairs):
        self.params = {id(p): (pair.block, p) for pair in pairs for p in (pair.A, pair.B)}
        self.seen = set()
        self.sums = {b: [0., 0., 0., 0] for b, _ in self.params.values()}

    @torch.no_grad()
    def __call__(self, params, updates, alpha, batched):
        if len(params) != len(updates) or not math.isfinite(alpha):
            raise ValueError("invalid Muon proposal")
        for p, update in zip(params, updates):
            key = id(p)
            if key not in self.params or key in self.seen or p.grad is None or update.shape != p.shape:
                raise ValueError("unexpected Muon proposal factor")
            self.seen.add(key)
            block, _ = self.params[key]
            g, u = p.grad.detach().double().reshape(-1), update.detach().double().reshape(-1)
            vals = (float(torch.dot(g,g)), float(torch.dot(u,u))*alpha*alpha,
                    -float(torch.dot(g,u))*alpha)
            if not all(math.isfinite(v) for v in vals):
                raise ValueError("nonfinite Muon predictor")
            for i, v in enumerate(vals):
                self.sums[block][i] += v
            self.sums[block][3] += p.numel()

    def result(self):
        if self.seen != set(self.params):
            raise RuntimeError("incomplete Muon proposal coverage")
        return {b: {"gradient_norm": math.sqrt(gg), "gradient_rms": math.sqrt(gg/n),
                    "learning_update_norm": math.sqrt(uu), "signed_first_order_utility": gu,
                    "descent_cosine": max(-1., min(1., gu/math.sqrt(gg*uu))) if gg and uu else None,
                    "factor_elements": n}
                for b, (gg, uu, gu, n) in sorted(self.sums.items())}
