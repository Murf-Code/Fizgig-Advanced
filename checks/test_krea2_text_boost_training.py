"""CPU-only tests for Krea 2's differentiable training Text Enhancer."""

import torch

from fizgig.krea2 import text_boost as tb
from fizgig.krea2.driver import Krea2Driver
from fizgig.krea2.model import SingleStreamDiT


def test_strength_and_text_scale_are_bounded():
    for bad in (None, "", "bad", -1, float("nan")):
        assert tb.clamped_strength(bad) == 0.0
    assert tb.clamped_strength(0.5) == 0.5
    assert tb.clamped_strength(99) == 2.0
    assert tb.clamped_text_scale(None) == 1.0
    assert tb.clamped_text_scale(0.01) == 0.25
    assert tb.clamped_text_scale(1.5) == 1.5
    assert tb.clamped_text_scale(99) == 4.0


def test_gain_profile_and_global_lift_match_the_donor():
    gains = tb.gains(1.0, "cpu", torch.float32)
    assert torch.allclose(gains, torch.tensor(list(tb.PROFILE_12) * 2))
    half = tb.gains(0.5, "cpu", torch.float32)
    assert abs(half[8].item() - 3.0) < 1e-6
    assert abs(half[9].item() - 1.05) < 1e-6

    context = torch.ones(1, 1, 12, 2560)
    chunks = tb.scaled_context(context, 1.0)[0, 0].reshape(24, 1280)
    assert abs(chunks[8, 0].item() - 75.0) < 1e-5
    assert abs(chunks[0, 0].item() - 15.0) < 1e-5


def test_shape_guard_and_disabled_path_do_not_transform_conditioning():
    context = torch.ones(1, 2, 12, 2560)
    assert tb.applies(context)
    for shape in ((1, 2, 12, 2048), (1, 2, 2560), (1, 2, 12, 2560, 1)):
        assert not tb.applies(torch.zeros(shape))
    assert tb.scaled_context(context, 0.0) is context

    calls = []
    result = tb.enhance(lambda value: (calls.append(value), value.sum(dim=-2))[1], context, 0.0)
    assert len(calls) == 1 and calls[0] is context
    assert result.shape == (1, 2, 2560)


def test_residual_is_capped_per_token_and_preserves_dtype():
    generator = torch.Generator().manual_seed(7)
    for scale in (1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0):
        reference = torch.randn(2, 3, 2560, generator=generator)
        candidate = reference + scale * torch.randn(2, 3, 2560, generator=generator)
        result = tb.capped_residual(reference, candidate)
        base = reference.pow(2).mean(-1, keepdim=True).sqrt()
        moved = (result - reference).pow(2).mean(-1, keepdim=True).sqrt()
        assert torch.all(moved <= 0.75 * base + 1e-5)
        assert torch.all(moved <= (candidate - reference).pow(2).mean(-1, keepdim=True).sqrt() + 1e-5)

    bf16 = tb.capped_residual(
        torch.ones(1, 1, 2560, dtype=torch.bfloat16),
        torch.full((1, 1, 2560), 3.0, dtype=torch.bfloat16),
    )
    assert bf16.dtype == torch.bfloat16


def test_enhancement_runs_two_passes_with_shared_attention_params_and_gradients():
    class Stub:
        attn_mode = "torch"
        split_attn = False

        def __init__(self):
            self.gain = torch.nn.Parameter(torch.tensor(0.75))
            self.calls = []
            self.txtmlp = torch.nn.Identity()

        def txtfusion(self, context, no_mask, with_mask):
            self.calls.append((context, no_mask, with_mask))
            return context.mean(dim=-2) * self.gain

    stub = Stub()
    context = torch.randn(1, 2, 12, 2560)
    mask = torch.ones(1, 2, dtype=torch.bool)
    result = SingleStreamDiT.fuse_text_context(
        stub, context, mask, text_boost=1.0, text_scale=1.25
    )

    assert len(stub.calls) == 2
    assert stub.calls[0][0] is context
    assert stub.calls[0][1] is stub.calls[1][1]
    assert stub.calls[0][2] is stub.calls[1][2]
    assert result.shape == (1, 2, 2560)
    result.square().mean().backward()
    assert stub.gain.grad is not None
    assert torch.isfinite(stub.gain.grad)


def test_disabled_model_path_keeps_one_fusion_pass():
    class Stub:
        attn_mode = "torch"
        split_attn = False

        def __init__(self):
            self.calls = 0
            self.txtmlp = torch.nn.Identity()

        def txtfusion(self, context, no_mask, with_mask):
            self.calls += 1
            return context.sum(dim=-2)

    stub = Stub()
    context = torch.ones(1, 2, 12, 2560)
    mask = torch.ones(1, 2, dtype=torch.bool)
    result = SingleStreamDiT.fuse_text_context(stub, context, mask)
    assert stub.calls == 1
    assert torch.equal(result, context.sum(dim=-2))


def test_driver_requires_explicit_opt_in_and_clamps_user_values():
    driver = Krea2Driver()
    driver.set_options({})
    assert driver._text_enhancement_settings() == (0.0, 1.0)

    driver.set_options({
        "krea2_text_enhance": "1",
        "krea2_text_boost": "9",
        "krea2_text_scale": "9",
    })
    assert driver._text_enhancement_settings() == (2.0, 4.0)

    driver.set_options({
        "krea2_text_enhance": "0",
        "krea2_text_boost": "1",
        "krea2_text_scale": "2",
    })
    assert driver._text_enhancement_settings() == (0.0, 1.0)
