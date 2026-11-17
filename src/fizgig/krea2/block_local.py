"""Small scheduling helpers for Krea 2's experimental block-local training path."""


def warmup_phase(optimizer_updates, head_warmup_steps, lora_ramp_steps):
    """Return (head_only, LoRA LR multiplier) for the next optimizer update.

    Counts optimizer updates, not data-loader micro-batches. The first ``head_warmup_steps`` updates train only the
    temporary heads; the next ``lora_ramp_steps`` updates ramp the LoRA from 1/N to full configured LR.
    """
    step = max(0, int(optimizer_updates or 0))
    head_steps = max(0, int(head_warmup_steps or 0))
    ramp_steps = max(0, int(lora_ramp_steps or 0))
    if step < head_steps:
        return True, 0.0
    if ramp_steps <= 0:
        return False, 1.0
    ramp_index = step - head_steps + 1
    return False, min(1.0, ramp_index / ramp_steps)
