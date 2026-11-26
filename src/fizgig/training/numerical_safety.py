"""Fail closed at numerical safety boundaries, without changing valid updates."""
import math
import torch


class NumericalError(RuntimeError):
    """An invalid training value; do not retry or publish this checkpoint."""


def require_finite(tensor, where):
    if not bool(torch.isfinite(tensor.detach()).all()):
        raise NumericalError(f"[numerical safety] nonfinite {where}; training stopped")


def require_finite_tensors(tensors, where):
    # Aggregate on each device; only synchronize once per device on the healthy path.
    groups = {}
    for name, t in tensors:
        if torch.is_tensor(t) and (t.is_floating_point() or t.is_complex()):
            groups.setdefault(t.device, []).append((name, t.detach()))
    for group in groups.values():
        flags = torch.stack([torch.isfinite(t).all() for _, t in group])
        if not bool(flags.all()):
            bad = [name for (name, _), ok in zip(group, flags.cpu().tolist()) if not ok]
            raise NumericalError(f"[numerical safety] nonfinite {where}: {bad[:8]}; training stopped")


@torch.no_grad()
def checked_clip_grad_norm(parameters, max_norm, where):
    """Use PyTorch's normal clipping unless its reduction overflows."""
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.tensor(0.0)
    if max_norm:
        try:
            return torch.nn.utils.clip_grad_norm_(params, max_norm, error_if_nonfinite=True)
        except RuntimeError as exc:
            if "non-finite" not in str(exc):
                raise
    grads = [(str(i), p.grad) for i, p in enumerate(params)]
    require_finite_tensors(grads, f"gradients before optimizer ({where})")
    if not max_norm:
        return None
    # This path is exceptional. FP64 avoids overflow from squaring FP32/BF16 values.
    norm = math.hypot(*(float(torch.linalg.vector_norm(g.detach(), dtype=torch.float64))
                        for _, g in grads))
    if not math.isfinite(norm):
        raise NumericalError(f"[numerical safety] gradient norm overflow ({where}); training stopped")
    factor = min(1.0, float(max_norm) / (norm + 1e-6))
    for _, g in grads:
        g.copy_((g.double() * factor).to(g.dtype))
    return torch.tensor(norm, dtype=torch.float64, device=params[0].device)


def require_finite_score(value, where):
    if not math.isfinite(float(value)):
        raise NumericalError(f"[numerical safety] nonfinite {where}; training stopped")


def tensor_leaves(value, name=""):
    if torch.is_tensor(value):
        yield name, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from tensor_leaves(child, f"{name}/{key}")
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            yield from tensor_leaves(child, f"{name}/{i}")


def save_finite_safetensors(tensors, path, metadata=None):
    import os
    import tempfile
    from safetensors.torch import save_file
    require_finite_tensors(tensors.items(), f"checkpoint {path}")
    fd, tmp = tempfile.mkstemp(prefix=".fizgig-save-", suffix=".safetensors", dir=os.path.dirname(os.path.abspath(path)))
    os.close(fd)
    try:
        save_file(tensors, tmp, metadata=metadata)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
