"""Explicit experimental stream controls shared by launcher and trainer.

Calibration profiles describe a fixed diagnostic workload, not all training runs.
Normal training never silently imports one; a user may opt into its strategy.
"""
STREAM_STRATEGIES = ("default", "staged3", "staged4", "direct2", "direct3", "direct4")


def clean_stream_strategy(value):
    value = str(value or "default").strip().lower()
    if value not in STREAM_STRATEGIES:
        raise ValueError("Unknown faster block swap strategy")
    return value


def apply_stream_strategy(config, strategy):
    """Only change an existing CUDA send-only training request.

    The central offloader factory still checks frozen weights, compatible
    layouts and classic VRAM capacity. Host pinning is separately preflighted.
    """
    from dataclasses import replace
    strategy = clean_stream_strategy(strategy)
    if (strategy == "default" or not config.h2d_only
            or not config.supports_backward or config.device.type != "cuda"):
        return config
    return replace(config, ring_size=int(strategy[-1]),
                   use_pinned_memory=strategy.startswith("direct"))
