"""CPU-sized integration checks for Krea 2's experimental block-local flow path."""

import pytest
import torch

from fizgig.krea2.driver import Krea2Driver
from fizgig.krea2.model import SingleMMDiTConfig, SingleStreamDiT


def _tiny_dit(layers=2):
    config = SingleMMDiTConfig(
        features=16,
        tdim=16,
        txtdim=16,
        heads=1,
        multiplier=2,
        layers=layers,
        patch=1,
        channels=2,
        bias=True,
        txtlayers=1,
        txtheads=1,
        txtkvheads=1,
    )
    dit = SingleStreamDiT(config, attn_mode="torch", split_attn=False)
    dit.eval().requires_grad_(False)
    # Simulate the trainable adapter weights that are attached inside the main blocks in a real run.
    for block in dit.blocks:
        block.mlp.gate.weight.requires_grad_(True)
    return dit


def test_block_local_global_replay_matches_full_model_loss_and_gradients():
    import copy

    torch.manual_seed(7)
    dit_replay = _tiny_dit()
    dit_direct = copy.deepcopy(dit_replay)
    replay_driver = Krea2Driver()
    replay_driver.set_options({"training_method": "block_local"})
    # No auxiliary heads are required when head-only warm-up is disabled.
    replay_driver.set_block_local_head_only(False)

    direct_driver = Krea2Driver()
    latents = torch.randn(1, 2, 4, 4)
    cond = {
        "hidden_states": torch.randn(1, 3, 1, 16),
        "attention_mask": torch.tensor([[True, True, True]]),
    }
    loss_replay, info = replay_driver.training_loss(
        dit_replay, latents, cond, torch.Generator().manual_seed(1234))
    loss_direct, _ = direct_driver.training_loss(
        dit_direct, latents, cond, torch.Generator().manual_seed(1234))

    assert info["block_local"] is True
    assert info["block_local_phase"] == "global_replay"
    assert len(dit_replay.blocks) == 2
    assert getattr(replay_driver, "_block_local_heads", None) is None
    assert loss_replay.ndim == 0 and torch.isfinite(loss_replay)
    assert torch.allclose(loss_replay, loss_direct.detach(), rtol=2e-3, atol=2e-3)

    replay_driver.backward_training_loss(0.5)
    (loss_direct * 0.5).backward()

    for replay_block, direct_block in zip(dit_replay.blocks, dit_direct.blocks):
        replay_grad = replay_block.mlp.gate.weight.grad
        direct_grad = direct_block.mlp.gate.weight.grad
        assert replay_grad is not None and direct_grad is not None
        assert torch.allclose(replay_grad, direct_grad, rtol=2e-2, atol=2e-3)
    assert all(parameter.grad is None for parameter in dit_replay.first.parameters())
    assert all(parameter.grad is None for parameter in dit_replay.tmlp.parameters())
    assert all(parameter.grad is None for parameter in dit_replay.last.parameters())
    # The auxiliary heads live outside the DiT, and therefore outside a normal LoRA export.
    assert not any("block_local" in key for key in dit_replay.state_dict())


def test_block_local_global_replay_matches_full_model_across_28_blocks():
    import copy

    torch.manual_seed(29)
    dit_replay = _tiny_dit(layers=28)
    dit_direct = copy.deepcopy(dit_replay)
    replay_driver = Krea2Driver()
    replay_driver.set_options({"training_method": "block_local"})
    replay_driver.set_block_local_head_only(False)
    direct_driver = Krea2Driver()

    latents = torch.randn(1, 2, 4, 4)
    cond = {
        "hidden_states": torch.randn(1, 3, 1, 16),
        "attention_mask": torch.tensor([[True, True, True]]),
    }
    replay_loss, _ = replay_driver.training_loss(
        dit_replay, latents, cond, torch.Generator().manual_seed(7654))
    direct_loss, _ = direct_driver.training_loss(
        dit_direct, latents, cond, torch.Generator().manual_seed(7654))
    assert torch.allclose(replay_loss, direct_loss.detach(), rtol=2e-3, atol=2e-3)

    replay_driver.backward_training_loss(0.25)
    (direct_loss * 0.25).backward()
    for index, (replay_block, direct_block) in enumerate(zip(dit_replay.blocks, dit_direct.blocks)):
        replay_grad = replay_block.mlp.gate.weight.grad
        direct_grad = direct_block.mlp.gate.weight.grad
        assert replay_grad is not None and direct_grad is not None, f"missing gradient at block {index}"
        assert torch.allclose(replay_grad, direct_grad, rtol=2e-2, atol=2e-3), f"gradient mismatch at block {index}"


def test_block_local_head_warmup_trains_heads_without_replaying_frozen_blocks():
    dit = _tiny_dit()
    driver = Krea2Driver()
    driver.set_options({"training_method": "block_local"})
    heads = driver.make_block_local_heads(dit, "cpu")
    for block in dit.blocks:
        for parameter in block.parameters():
            parameter.requires_grad_(False)

    latents = torch.randn(1, 2, 4, 4)
    cond = {
        "hidden_states": torch.randn(1, 3, 1, 16),
        "attention_mask": torch.tensor([[True, True, True]]),
    }
    loss, info = driver.training_loss(dit, latents, cond, torch.Generator().manual_seed(4321))
    assert info["block_local"] is True
    assert info["block_local_phase"] == "head_warmup"
    assert loss.ndim == 0 and torch.isfinite(loss)
    driver.backward_training_loss()

    assert all(parameter.grad is None for block in dit.blocks for parameter in block.parameters())
    assert all(parameter.grad is not None for parameter in heads.parameters())


def test_block_local_head_warmup_backward_matches_reported_mean_loss():
    import copy
    import torch.nn.functional as F

    torch.manual_seed(19)
    dit = _tiny_dit()
    driver = Krea2Driver()
    driver.set_options({"training_method": "block_local"})
    heads = driver.make_block_local_heads(dit, "cpu")
    reference_heads = copy.deepcopy(heads)

    latents = torch.randn(1, 2, 4, 4)
    cond = {
        "hidden_states": torch.randn(1, 3, 1, 16),
        "attention_mask": torch.tensor([[True, True, True]]),
    }
    loss, _ = driver.training_loss(dit, latents, cond, torch.Generator().manual_seed(991))
    state = driver._pending_block_local
    outputs = state["block_outputs"]
    target = state["target"]
    t_emb = state["t_emb"]
    imglen = state["imglen"]
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        reference_losses = [
            F.mse_loss(head(output, t_emb)[:, :imglen, :].float(), target.float())
            for head, output in zip(reference_heads, outputs)
        ]
        reference_loss = torch.stack(reference_losses).mean()
    assert torch.allclose(loss, reference_loss.detach(), rtol=1e-5, atol=1e-6)

    scale = 0.37
    (reference_loss * scale).backward()
    driver.backward_training_loss(scale)

    for actual, expected in zip(heads.parameters(), reference_heads.parameters()):
        assert actual.grad is not None and expected.grad is not None
        assert torch.allclose(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)


def test_validation_uses_the_normal_global_forward_path():
    dit = _tiny_dit()
    driver = Krea2Driver()
    driver.set_options({"training_method": "block_local"})
    driver.make_block_local_heads(dit, "cpu")

    latents = torch.randn(1, 2, 4, 4)
    cond = {
        "hidden_states": torch.randn(1, 3, 1, 16),
        "attention_mask": torch.tensor([[True, True, True]]),
    }
    with torch.no_grad():
        loss, info = driver.training_loss(dit, latents, cond, torch.Generator().manual_seed(1234))

    assert "block_local" not in info
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert driver._pending_block_local is None


def test_temporary_heads_drop_any_frozen_lora_attached_to_the_final_readout():
    from fizgig.families.lora import LoRALinear

    dit = _tiny_dit()
    dit.last.linear = LoRALinear(dit.last.linear)
    dit.last.linear.add("frozen_context", rank=2, alpha=2, trainable=False)

    driver = Krea2Driver()
    heads = driver.make_block_local_heads(dit, "cpu")

    assert all(isinstance(head.linear, LoRALinear) for head in heads)
    assert all(len(head.linear.adapters) == 0 for head in heads)
    assert all(parameter.requires_grad for parameter in heads.parameters())
    assert all(not parameter.requires_grad for parameter in dit.last.linear.adapters.parameters())


def test_block_local_head_only_warmup_and_lora_ramp_boundaries():
    from fizgig.krea2.block_local import warmup_phase as _block_local_warmup_phase

    # The configured head-only calibration precedes the first LoRA update, which starts at 1/40 LR.
    assert _block_local_warmup_phase(0, 120, 40) == (True, 0.0)
    assert _block_local_warmup_phase(119, 120, 40) == (True, 0.0)
    assert _block_local_warmup_phase(120, 120, 40) == (False, 1 / 40)
    assert _block_local_warmup_phase(159, 120, 40) == (False, 1.0)
    assert _block_local_warmup_phase(160, 120, 40) == (False, 1.0)
    assert _block_local_warmup_phase(7, 0, 0) == (False, 1.0)
    assert _block_local_warmup_phase(0, 0, 4) == (False, 0.25)


def test_krea2_block_local_warmup_options_are_regular_trainer_args():
    from fizgig.families.registry import get as get_family
    from fizgig.families.launch import option_tokens

    desc = get_family("krea2")
    base = {"KREA2_TRAINING_METHOD": "Block-local flow matching (experimental)"}
    values = {**base, "KREA2_BLOCK_LOCAL_HEAD_WARMUP_STEPS": "120",
              "KREA2_BLOCK_LOCAL_LORA_RAMP_STEPS": "40"}
    args, aux = option_tokens(desc, {"FAMILY_OPTIONS": values})
    assert args == ["--family_option", "training_method=block_local",
                    "--block_local_head_warmup_steps", "120",
                    "--block_local_lora_ramp_steps", "40"]
    assert aux == []

    args, _ = option_tokens(desc, {"FAMILY_OPTIONS": {"KREA2_TRAINING_METHOD": "Standard backprop",
                                                        "KREA2_BLOCK_LOCAL_HEAD_WARMUP_STEPS": "120",
                                                        "KREA2_BLOCK_LOCAL_LORA_RAMP_STEPS": "40"}})
    assert "--block_local_head_warmup_steps" not in args
    assert "--block_local_lora_ramp_steps" not in args


def test_training_parser_declares_block_local_warmup_flags_as_top_level_args():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "src/fizgig/families/train.py").read_text(encoding="utf-8")
    assert 'p.add_argument("--block_local_head_warmup_steps", type=int, default=0' in source
    assert 'p.add_argument("--block_local_lora_ramp_steps", type=int, default=0' in source
    assert "block_local_head_warmup_steps=a.block_local_head_warmup_steps" in source
    assert "block_local_lora_ramp_steps=a.block_local_lora_ramp_steps" in source


def test_training_parser_declares_residual_path_approx_lr_multiplier():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "src/fizgig/families/train.py").read_text(encoding="utf-8")
    assert 'p.add_argument("--residual_path_approx_lr_multiplier", type=float, default=2.0' in source
    assert "residual_path_approx_lr_multiplier=a.residual_path_approx_lr_multiplier" in source


def test_block_local_heads_can_train_while_lora_blocks_are_frozen():
    dit = _tiny_dit()
    driver = Krea2Driver()
    driver.set_options({"training_method": "block_local"})
    heads = driver.make_block_local_heads(dit, "cpu")
    lora_parameters = [block.mlp.gate.weight for block in dit.blocks]
    for parameter in lora_parameters:
        parameter.requires_grad_(False)

    latents = torch.randn(1, 2, 4, 4)
    cond = {
        "hidden_states": torch.randn(1, 3, 1, 16),
        "attention_mask": torch.tensor([[True, True, True]]),
    }
    driver.set_block_local_head_only(True)
    loss, info = driver.training_loss(dit, latents, cond, torch.Generator().manual_seed(4321))
    assert info["block_local"] is True
    driver.backward_training_loss()

    assert all(parameter.grad is None for parameter in lora_parameters)
    assert all(parameter.grad is not None for parameter in heads.parameters())


def _selective_tiny_dit(layers=2):
    from fizgig.families.lora import LoRALinear, TRAINABLE

    dit = _tiny_dit(layers=layers)
    wrapped = {}
    for index, block in enumerate(dit.blocks):
        name = f"blocks.{index}.mlp.gate"
        layer = LoRALinear(block.mlp.gate)
        layer.add(TRAINABLE, rank=2, alpha=2, trainable=True)
        block.mlp.gate = layer
        wrapped[name] = layer
    net = type("FakeLoRANet", (), {"wrapped": wrapped})()
    return dit, net


def _residual_hybrid_tiny_dit(layers=2):
    from fizgig.families.lora import LoRALinear, TRAINABLE

    dit = _tiny_dit(layers=layers)
    dit.requires_grad_(False)
    wrapped = {}
    for index, block in enumerate(dit.blocks):
        for suffix, parent, attr in (("attn.wo", block.attn, "wo"), ("mlp.down", block.mlp, "down")):
            base = getattr(parent, attr)
            layer = LoRALinear(base)
            layer.add(TRAINABLE, rank=2, alpha=2, trainable=True)
            setattr(parent, attr, layer)
            wrapped[f"blocks.{index}.{suffix}"] = layer
    net = type("FakeLoRANet", (), {"wrapped": wrapped})()
    return dit, net


def test_residual_path_approx_lr_multiplier_applies_only_between_exact_calibrations():
    dit, net = _residual_hybrid_tiny_dit(layers=2)
    driver = Krea2Driver()
    driver.set_options({"training_method": "residual_path_hybrid"})
    driver.configure_residual_path_hybrid(dit, net, exact_every=3, approx_lr_multiplier=2.0)

    # Updates 1 and 2 are approximate; update 3 is exact and must retain the base LR.
    assert driver.residual_path_update_lr_multiplier == 2.0
    driver.finish_residual_path_optimizer_step()
    assert driver.residual_path_update_lr_multiplier == 2.0
    driver.finish_residual_path_optimizer_step()
    assert driver.residual_path_exact_due is True
    assert driver.residual_path_update_lr_multiplier == 1.0
    driver.finish_residual_path_optimizer_step()
    assert driver.residual_path_exact_due is False
    assert driver.residual_path_update_lr_multiplier == 2.0


def test_residual_path_approx_lr_multiplier_validation_and_h2d_cadence():
    dit, net = _residual_hybrid_tiny_dit(layers=2)
    driver = Krea2Driver()
    driver.set_options({"training_method": "residual_path_hybrid_h2d"})
    with pytest.raises(ValueError, match="finite number greater than zero"):
        driver.configure_residual_path_hybrid(dit, net, exact_every=2, approx_lr_multiplier=0.0)
    driver.configure_residual_path_hybrid(dit, net, exact_every=2, approx_lr_multiplier=1.75)
    assert driver.h2d_block_swap is True
    assert driver.residual_path_update_lr_multiplier == 1.75
    driver.finish_residual_path_optimizer_step()
    assert driver.residual_path_update_lr_multiplier == 1.0


def test_residual_path_hybrid_uses_identity_credit_then_periodic_exact_replay():
    torch.manual_seed(20261116)
    dit, net = _residual_hybrid_tiny_dit(layers=3)
    driver = Krea2Driver()
    driver.set_options({"training_method": "residual_path_hybrid"})
    driver.configure_residual_path_hybrid(dit, net, exact_every=2)

    latents = torch.randn(1, 2, 4, 4)
    cond = {
        "hidden_states": torch.randn(1, 3, 1, 16),
        "attention_mask": torch.tensor([[True, True, True]]),
    }

    # Update 1: a single model forward plus the readout-only derivative; no transformer-block backward/replay.
    approx_loss, approx_info = driver.training_loss(
        dit, latents, cond, torch.Generator().manual_seed(8801))
    assert approx_info["residual_path_hybrid"] is True
    assert approx_info["block_local_phase"] == "identity_credit"
    assert torch.isfinite(approx_loss)
    driver.backward_training_loss(0.5)
    for block in dit.blocks:
        for suffix in ("attn.wo", "mlp.down"):
            layer = block.attn.wo if suffix == "attn.wo" else block.mlp.down
            adapter = layer.adapters["lora"]
            assert adapter[0].weight.grad is None  # A is refreshed by exact calibration only.
            assert adapter[1].weight.grad is not None
            assert torch.isfinite(adapter[1].weight.grad).all()
            assert adapter[1].weight.grad.abs().sum() > 0

    # Simulate the optimizer boundary. Update 2 is the scheduled exact calibration step.
    from fizgig.training.optimizers import create_optimizer
    trainable = [p for p in dit.parameters() if p.requires_grad]
    optimizer, _ = create_optimizer("muon", trainable, 1e-4, args_str="")
    optimizer.step()  # Muon must safely skip the A factors whose gradients are intentionally absent this update.
    optimizer.zero_grad(set_to_none=True)
    status = driver.finish_residual_path_optimizer_step()
    assert status["next_exact"] is True
    exact_loss, exact_info = driver.training_loss(
        dit, latents, cond, torch.Generator().manual_seed(8802))
    assert exact_info["block_local_phase"] == "exact_calibration"
    assert torch.isfinite(exact_loss)
    driver.backward_training_loss(0.5)
    for block in dit.blocks:
        for layer in (block.attn.wo, block.mlp.down):
            adapter = layer.adapters["lora"]
            assert adapter[0].weight.grad is not None
            assert adapter[1].weight.grad is not None

    state = driver.residual_path_state_dict()
    assert state == {"version": 3, "exact_every": 2, "approx_lr_multiplier": 2.0,
                     "optimizer_steps": 1, "approximate_steps": 1, "selective_a_every": 0}
    restored = Krea2Driver()
    restored.set_options({"training_method": "residual_path_hybrid"})
    restored.configure_residual_path_hybrid(dit, net, exact_every=2, approx_lr_multiplier=2.0)
    restored.load_residual_path_state_dict(state)
    assert restored.residual_path_state_dict() == state

    with pytest.raises(RuntimeError, match="approximate-update LR multiplier differs"):
        changed = Krea2Driver()
        changed.set_options({"training_method": "residual_path_hybrid"})
        changed.configure_residual_path_hybrid(dit, net, exact_every=2, approx_lr_multiplier=1.0)
        changed.load_residual_path_state_dict(state)

    # Older v1 states from the first Residual-path Hybrid build remain resumable under the new LR setting.
    legacy = {"version": 1, "exact_every": 2, "optimizer_steps": 1}
    restored.load_residual_path_state_dict(legacy)
    assert restored.residual_path_state_dict()["optimizer_steps"] == 1

    with pytest.raises(ValueError, match="at least 2"):
        restored.configure_residual_path_hybrid(dit, net, exact_every=1)


def test_residual_path_hybrid_captures_all_28_main_blocks():
    dit, net = _residual_hybrid_tiny_dit(layers=28)
    driver = Krea2Driver()
    driver.set_options({"training_method": "residual_path_hybrid"})
    driver.configure_residual_path_hybrid(dit, net, exact_every=8)
    latents = torch.randn(1, 2, 4, 4)
    cond = {
        "hidden_states": torch.randn(1, 3, 1, 16),
        "attention_mask": torch.tensor([[True, True, True]]),
    }

    loss, info = driver.training_loss(
        dit, latents, cond, torch.Generator().manual_seed(2816))
    assert info["block_local_phase"] == "identity_credit"
    assert torch.isfinite(loss)
    assert len(driver._pending_block_local["captures"]) == 28
    driver.backward_training_loss()

    assert all(
        block.attn.wo.adapters["lora"][0].weight.grad is None
        and block.mlp.down.adapters["lora"][0].weight.grad is None
        and block.attn.wo.adapters["lora"][1].weight.grad is not None
        and block.mlp.down.adapters["lora"][1].weight.grad is not None
        for block in dit.blocks
    )



def test_residual_path_selective_a_updates_a_every_fourth_approximate_step():
    torch.manual_seed(20261116)
    from fizgig.krea2 import offloading

    dit, net = _residual_hybrid_tiny_dit(layers=3)
    driver = Krea2Driver()
    driver.set_options({"training_method": "residual_path_hybrid_h2d_selective_a"})
    why = offloading.send_only_unfit(dit.blocks, len(dit.blocks), 3)
    assert why is None, why
    driver.enable_block_swap(dit, 1, "cpu", supports_backward=True, send_only=True)
    assert isinstance(dit.offloader, offloading.LoRAStreamOffloader)
    driver.configure_residual_path_hybrid(dit, net, exact_every=5, approx_lr_multiplier=2.0)

    assert driver.residual_path_hybrid is True
    assert driver.residual_path_selective_a is True
    assert driver.h2d_block_swap is True
    assert driver.residual_path_a_update_due is False

    latents = torch.randn(1, 2, 4, 4)
    cond = {
        "hidden_states": torch.randn(1, 3, 1, 16),
        "attention_mask": torch.tensor([[True, True, True]]),
    }
    params = [p for p in dit.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.02)

    # A/B gradients are both accumulated only on approximate update 4. B is trained on updates 1-4.
    for approximate_index in range(1, 5):
        optimizer.zero_grad(set_to_none=True)
        loss, info = driver.training_loss(
            dit, latents, cond, torch.Generator().manual_seed(9000 + approximate_index))
        assert info["block_local_phase"] == "identity_credit"
        state = driver._pending_block_local
        assert state["update_a"] is (approximate_index == 4)
        for block_index, block in enumerate(dit.blocks):
            captured = state["captures"][block_index]
            assert set(captured["projections"]) == {"attn.wo", "mlp.down"}
            for projection in captured["projections"].values():
                if approximate_index == 4:
                    assert projection["x_host"].device.type == "cpu"
                else:
                    assert "x_host" not in projection
        driver.backward_training_loss()
        for block in dit.blocks:
            for layer in (block.attn.wo, block.mlp.down):
                a, b = layer.adapters["lora"]
                assert b.weight.grad is not None
                assert (a.weight.grad is not None) is (approximate_index == 4)
                if approximate_index == 4:
                    assert torch.isfinite(a.weight.grad).all()
                    assert a.weight.grad.abs().sum() > 0
        optimizer.step()
        status = driver.finish_residual_path_optimizer_step()

    assert driver.residual_path_a_update_due is False  # selective cadence counts approximate updates only
    assert driver.residual_path_state_dict()["approximate_steps"] == 4

    # The fifth optimizer update is exact (exact_every=5), so it is NOT counted as an approximate A update.
    optimizer.zero_grad(set_to_none=True)
    exact_loss, exact_info = driver.training_loss(
        dit, latents, cond, torch.Generator().manual_seed(9005))
    assert exact_info["block_local_phase"] == "exact_calibration"
    driver.backward_training_loss()
    for block in dit.blocks:
        for layer in (block.attn.wo, block.mlp.down):
            a, b = layer.adapters["lora"]
            assert a.weight.grad is not None and b.weight.grad is not None
    optimizer.step()
    driver.finish_residual_path_optimizer_step()

    # After an exact calibration, approximate update 5 is still not an A update; update 8 will be.
    assert driver.residual_path_a_update_due is False
    state = driver.residual_path_state_dict()
    assert state["approximate_steps"] == 4
    restored = Krea2Driver()
    restored.set_options({"training_method": "residual_path_hybrid_h2d_selective_a"})
    restored.configure_residual_path_hybrid(dit, net, exact_every=5, approx_lr_multiplier=2.0)
    restored.load_residual_path_state_dict(state)
    assert restored.residual_path_state_dict() == state
    assert restored.residual_path_a_update_due is False

    # Three more B-only updates follow (approximate #5-7); the next approximate update (#8) is A+B.
    for approximate_index in range(5, 8):
        optimizer.zero_grad(set_to_none=True)
        _, _ = driver.training_loss(
            dit, latents, cond, torch.Generator().manual_seed(9010 + approximate_index))
        assert driver._pending_block_local["update_a"] is False
        driver.backward_training_loss()
        for block in dit.blocks:
            for layer in (block.attn.wo, block.mlp.down):
                assert layer.adapters["lora"][0].weight.grad is None
                assert layer.adapters["lora"][1].weight.grad is not None
        optimizer.step()
        driver.finish_residual_path_optimizer_step()
    assert driver.residual_path_state_dict()["approximate_steps"] == 7
    assert driver.residual_path_a_update_due is True


def test_residual_path_selective_a_can_resume_legacy_b_only_state():
    dit, net = _residual_hybrid_tiny_dit(layers=2)
    driver = Krea2Driver()
    driver.set_options({"training_method": "residual_path_hybrid_h2d_selective_a"})
    driver.configure_residual_path_hybrid(dit, net, exact_every=8, approx_lr_multiplier=2.0)
    # A v2 B-only state at optimizer update 5 implies five approximate updates have elapsed.
    driver.load_residual_path_state_dict({"version": 2, "exact_every": 8,
                                          "approx_lr_multiplier": 2.0, "optimizer_steps": 5})
    assert driver._residual_path_approx_steps == 5
    assert driver.residual_path_a_update_due is False


def test_residual_path_hybrid_classic_swap_forward_only_then_exact_replay():
    from fizgig.krea2.offloading import ModelOffloader

    dit, net = _residual_hybrid_tiny_dit(layers=3)
    driver = Krea2Driver()
    driver.set_options({"training_method": "residual_path_hybrid"})
    driver.enable_block_swap(dit, 1, "cpu", supports_backward=True, send_only=False)
    assert isinstance(dit.offloader, ModelOffloader)
    driver.configure_residual_path_hybrid(dit, net, exact_every=2)
    latents = torch.randn(1, 2, 4, 4)
    cond = {
        "hidden_states": torch.randn(1, 3, 1, 16),
        "attention_mask": torch.tensor([[True, True, True]]),
    }

    approx_loss, approx_info = driver.training_loss(
        dit, latents, cond, torch.Generator().manual_seed(301))
    assert approx_info["block_local_phase"] == "identity_credit"
    driver.backward_training_loss()
    driver.finish_residual_path_optimizer_step()

    exact_loss, exact_info = driver.training_loss(
        dit, latents, cond, torch.Generator().manual_seed(302))
    assert exact_info["block_local_phase"] == "exact_calibration"
    driver.backward_training_loss()
    assert torch.isfinite(approx_loss) and torch.isfinite(exact_loss)


def test_residual_path_hybrid_h2d_choice_uses_h2d_offloader_mode():
    driver = Krea2Driver()
    driver.set_options({"training_method": "residual_path_hybrid_h2d"})
    assert driver.residual_path_hybrid is True
    assert driver.block_local_flow is True
    assert driver.h2d_block_swap is True
    assert driver.manual_loss_backward is True


def test_residual_path_hybrid_h2d_forward_only_pass_can_be_followed_by_exact_replay():
    from fizgig.krea2 import offloading

    dit, net = _residual_hybrid_tiny_dit(layers=3)
    driver = Krea2Driver()
    driver.set_options({"training_method": "residual_path_hybrid_h2d"})
    why = offloading.send_only_unfit(dit.blocks, len(dit.blocks), 3)
    assert why is None, why
    driver.enable_block_swap(dit, 1, "cpu", supports_backward=True, send_only=True)
    assert isinstance(dit.offloader, offloading.LoRAStreamOffloader)
    driver.configure_residual_path_hybrid(dit, net, exact_every=2)
    latents = torch.randn(1, 2, 4, 4)
    cond = {
        "hidden_states": torch.randn(1, 3, 1, 16),
        "attention_mask": torch.tensor([[True, True, True]]),
    }

    approx_loss, approx_info = driver.training_loss(
        dit, latents, cond, torch.Generator().manual_seed(102))
    assert approx_info["block_local_phase"] == "identity_credit"
    driver.backward_training_loss()
    for block in dit.blocks:
        for layer in (block.attn.wo, block.mlp.down):
            assert layer.adapters["lora"][0].weight.grad is None
            assert layer.adapters["lora"][1].weight.grad is not None
    driver.finish_residual_path_optimizer_step()

    exact_loss, exact_info = driver.training_loss(
        dit, latents, cond, torch.Generator().manual_seed(103))
    assert exact_info["block_local_phase"] == "exact_calibration"
    driver.backward_training_loss()
    assert torch.isfinite(approx_loss) and torch.isfinite(exact_loss)


def test_selective_global_replay_scores_actual_optimizer_direction_and_reactivates():
    dit, net = _selective_tiny_dit(layers=2)
    driver = Krea2Driver()
    driver.set_options({"training_method": "selective_global_replay"})
    driver.configure_selective_global_replay(dit, net, warmup_updates=1, reactivate_every=2, score_decay=0.5)

    block_params = driver._selective_params_by_block
    for values in block_params.values():
        for parameter in values:
            parameter.grad = torch.ones_like(parameter)

    snapshot = driver.selective_snapshot_before_step()
    assert snapshot is not None and set(snapshot["blocks"]) == {0, 1}
    # With g = +1, block 0's parameter displacement is downhill; block 1's is uphill.
    with torch.no_grad():
        for parameter in block_params[0]:
            parameter.add_(-0.1)
        for parameter in block_params[1]:
            parameter.add_(0.1)
    status = driver.finish_selective_optimizer_step(snapshot)

    assert status["kind"] == "probe"
    assert status["active"] == 1
    assert driver.selective_active_blocks() == [0]
    assert all(p.requires_grad for p in block_params[0])
    assert all(not p.requires_grad for p in block_params[1])
    assert driver._selective_scores[0] > 0
    assert driver._selective_scores[1] < 0

    # After one selective window, all blocks wake for the next full probe update.
    status = driver.finish_selective_optimizer_step(None)
    assert status["kind"] == "reactivation"
    assert driver.selective_active_blocks() == [0, 1]
    assert all(p.requires_grad for values in block_params.values() for p in values)


def test_selective_global_replay_skips_adapter_grads_but_propagates_through_block():
    dit, net = _selective_tiny_dit(layers=2)
    driver = Krea2Driver()
    driver.set_options({"training_method": "selective_global_replay"})
    driver.configure_selective_global_replay(dit, net, warmup_updates=1, reactivate_every=2)
    driver._selective_active = {0}
    driver._set_selective_active_parameters()

    latents = torch.randn(1, 2, 4, 4)
    cond = {
        "hidden_states": torch.randn(1, 3, 1, 16),
        "attention_mask": torch.tensor([[True, True, True]]),
    }
    loss, info = driver.training_loss(dit, latents, cond, torch.Generator().manual_seed(2468))
    assert info["block_local"] is True
    assert info["block_local_phase"] == "global_replay"
    assert torch.isfinite(loss)
    driver.backward_training_loss()

    assert any(p.grad is not None for p in driver._selective_params_by_block[0])
    assert all(p.grad is None for p in driver._selective_params_by_block[1])


def test_krea2_training_method_options_include_selective_global_replay_and_legacy_label():
    from fizgig.families.registry import get as get_family
    from fizgig.families.launch import option_tokens

    desc = get_family("krea2")
    options = next(option for option in desc.options if option.key == "KREA2_TRAINING_METHOD")
    assert options.choice_labels() == ["Standard backprop", "Blockwise flow (experimental)",
                                       "Selective global replay (experimental)",
                                       "Blockwise flow H2D (experimental)",
                                       "Selective global replay H2D (experimental)",
                                       "Residual-path hybrid (experimental)",
                                       "Residual-path hybrid H2D (experimental)",
                                       "Residual-path hybrid H2D — selective A (experimental)"]
    assert options.pick("Block-local flow matching (experimental)") == "Blockwise flow (experimental)"

    args, _ = option_tokens(desc, {"FAMILY_OPTIONS": {
        "KREA2_TRAINING_METHOD": "Selective global replay (experimental)"}})
    assert args == ["--family_option", "training_method=selective_global_replay"]

    args, _ = option_tokens(desc, {"FAMILY_OPTIONS": {
        "KREA2_TRAINING_METHOD": "Blockwise flow (experimental)",
        "KREA2_BLOCK_LOCAL_HEAD_WARMUP_STEPS": "120"}})
    assert args == ["--family_option", "training_method=block_local",
                    "--block_local_head_warmup_steps", "120",
                    "--block_local_lora_ramp_steps", "0"]

    args, _ = option_tokens(desc, {"FAMILY_OPTIONS": {
        "KREA2_TRAINING_METHOD": "Blockwise flow H2D (experimental)",
        "KREA2_BLOCK_LOCAL_HEAD_WARMUP_STEPS": "120",
        "KREA2_BLOCK_LOCAL_LORA_RAMP_STEPS": "40"}})
    assert args == ["--family_option", "training_method=block_local_h2d",
                    "--block_local_head_warmup_steps", "120",
                    "--block_local_lora_ramp_steps", "40"]

    args, _ = option_tokens(desc, {"FAMILY_OPTIONS": {
        "KREA2_TRAINING_METHOD": "Selective global replay H2D (experimental)"}})
    assert args == ["--family_option", "training_method=selective_global_replay_h2d"]

    args, _ = option_tokens(desc, {"FAMILY_OPTIONS": {
        "KREA2_TRAINING_METHOD": "Residual-path hybrid (experimental)",
        "KREA2_RESIDUAL_PATH_EXACT_EVERY": "4",
        "KREA2_RESIDUAL_PATH_APPROX_LR_MULTIPLIER": "2.5"}})
    assert args == ["--family_option", "training_method=residual_path_hybrid",
                    "--residual_path_exact_every", "4",
                    "--residual_path_approx_lr_multiplier", "2.5"]

    args, _ = option_tokens(desc, {"FAMILY_OPTIONS": {
        "KREA2_TRAINING_METHOD": "Residual-path hybrid H2D (experimental)"}})
    assert args == ["--family_option", "training_method=residual_path_hybrid_h2d",
                    "--residual_path_exact_every", "8",
                    "--residual_path_approx_lr_multiplier", "2.0"]
    args, _ = option_tokens(desc, {"FAMILY_OPTIONS": {
        "KREA2_TRAINING_METHOD": "Residual-path hybrid H2D — selective A (experimental)"}})
    assert args == ["--family_option", "training_method=residual_path_hybrid_h2d_selective_a",
                    "--residual_path_exact_every", "8",
                    "--residual_path_approx_lr_multiplier", "2.0"]


def test_selective_global_replay_state_round_trips():
    dit, net = _selective_tiny_dit(layers=2)
    driver = Krea2Driver()
    driver.set_options({"training_method": "selective_global_replay"})
    driver.configure_selective_global_replay(dit, net, warmup_updates=1, reactivate_every=3, score_decay=0.5)
    driver._selective_scores = [0.2, -0.1]
    driver._selective_active = {0}
    driver._selective_optimizer_steps = 9
    driver._set_selective_active_parameters()
    state = driver.selective_state_dict()

    restored_dit, restored_net = _selective_tiny_dit(layers=2)
    restored = Krea2Driver()
    restored.set_options({"training_method": "selective_global_replay"})
    restored.configure_selective_global_replay(restored_dit, restored_net, warmup_updates=1, reactivate_every=3,
                                               score_decay=0.5)
    restored.load_selective_state_dict(state)

    assert restored.selective_state_dict() == state
    assert all(p.requires_grad for p in restored._selective_params_by_block[0])
    assert all(not p.requires_grad for p in restored._selective_params_by_block[1])


@pytest.mark.parametrize(
    "method,active_blocks",
    [
        ("block_local_h2d", None),
        ("selective_global_replay_h2d", {0, 2, 4, 6}),
    ],
)
def test_h2d_replay_matches_global_loss_and_active_lora_gradients(method, active_blocks):
    """H2D-only ring reuse is safe for manual replay: each block's graph is backpropagated before a slot is reused."""
    import copy
    from fizgig.families.lora import LoRALinear, TRAINABLE
    from fizgig.krea2 import offloading

    def make_lora_dit(seed=9102):
        torch.manual_seed(seed)
        dit = _tiny_dit(layers=8)
        wrapped = {}
        for index, block in enumerate(dit.blocks):
            base = block.mlp.gate
            base.weight.requires_grad_(False)  # base weights stay frozen and are eligible for H2D streaming
            layer = LoRALinear(base)
            layer.add(TRAINABLE, rank=2, alpha=2, trainable=True)
            torch.nn.init.normal_(layer.adapters[TRAINABLE][1].weight, std=0.02)
            block.mlp.gate = layer
            wrapped[f"blocks.{index}.mlp.gate"] = layer
        return dit, type("FakeLoRANet", (), {"wrapped": wrapped})()

    dit_replay, net = make_lora_dit()
    dit_direct = copy.deepcopy(dit_replay)
    replay_driver = Krea2Driver()
    replay_driver.set_options({"training_method": method})
    assert replay_driver.block_local_flow is True
    assert replay_driver.h2d_block_swap is True
    assert replay_driver.selective_global_replay is (method == "selective_global_replay_h2d")
    replay_driver.enable_block_swap(dit_replay, 2, "cpu", supports_backward=True, send_only=True)
    assert isinstance(dit_replay.offloader, offloading.LoRAStreamOffloader)
    if method == "selective_global_replay_h2d":
        replay_driver.configure_selective_global_replay(dit_replay, net, warmup_updates=1, reactivate_every=2)
        replay_driver._selective_active = set(active_blocks)
        replay_driver._set_selective_active_parameters()

    direct_driver = Krea2Driver()
    latents = torch.randn(1, 2, 4, 4)
    cond = {
        "hidden_states": torch.randn(1, 3, 1, 16),
        "attention_mask": torch.tensor([[True, True, True]]),
    }
    replay_loss, info = replay_driver.training_loss(
        dit_replay, latents, cond, torch.Generator().manual_seed(24680))
    direct_loss, _ = direct_driver.training_loss(
        dit_direct, latents, cond, torch.Generator().manual_seed(24680))
    assert info["block_local"] is True
    assert torch.allclose(replay_loss, direct_loss.detach(), rtol=2e-3, atol=2e-3)

    replay_driver.backward_training_loss(0.4)
    (direct_loss * 0.4).backward()

    for index, (replay_block, direct_block) in enumerate(zip(dit_replay.blocks, dit_direct.blocks)):
        replay_params = list(replay_block.mlp.gate.adapters[TRAINABLE].parameters())
        direct_params = list(direct_block.mlp.gate.adapters[TRAINABLE].parameters())
        if active_blocks is None or index in active_blocks:
            for actual, expected in zip(replay_params, direct_params):
                assert actual.grad is not None and expected.grad is not None, f"missing gradient at block {index}"
                assert torch.allclose(actual.grad, expected.grad, rtol=2e-2, atol=2e-3), f"gradient mismatch at block {index}"
        else:
            assert all(parameter.grad is None for parameter in replay_params), f"inactive block {index} received a gradient"
