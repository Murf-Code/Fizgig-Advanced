"""CPU-sized integration checks for Krea 2's Global replay path."""

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
    replay_driver.set_options({"training_method": "global_replay"})

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
    replay_driver.set_options({"training_method": "global_replay"})
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


def test_validation_uses_the_normal_global_forward_path():
    dit = _tiny_dit()
    driver = Krea2Driver()
    driver.set_options({"training_method": "global_replay"})

    latents = torch.randn(1, 2, 4, 4)
    cond = {
        "hidden_states": torch.randn(1, 3, 1, 16),
        "attention_mask": torch.tensor([[True, True, True]]),
    }
    with torch.no_grad():
        loss, info = driver.training_loss(dit, latents, cond, torch.Generator().manual_seed(1234))

    assert "block_local" not in info
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert getattr(driver, "_pending_block_local", None) is None


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


def test_selective_global_replay_scores_actual_optimizer_direction_and_reactivates():
    dit, net = _selective_tiny_dit(layers=2)
    driver = Krea2Driver()
    driver.set_options({"training_method": "global_replay", "selective_block_check": "1"})
    driver.configure_selective_global_replay(dit, net, warmup_updates=1, reactivate_every=2, score_decay=0.5)

    block_params = driver._selective_params_by_block
    # Displacement since the last probe (here: since configure), then the fresh probe gradient g = +1:
    # block 0 moved against g (downhill); block 1 moved along it (uphill).
    with torch.no_grad():
        for parameter in block_params[0]:
            parameter.add_(-0.1)
        for parameter in block_params[1]:
            parameter.add_(0.1)
    for values in block_params.values():
        for parameter in values:
            parameter.grad = torch.ones_like(parameter)

    snapshot = driver.selective_snapshot_before_step()
    assert snapshot is not None and set(snapshot["cosines"]) == {0, 1}
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
    driver.set_options({"training_method": "global_replay", "selective_block_check": "1"})
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


def test_krea2_training_method_is_standard_or_global_replay_with_a_block_check_toggle():
    from fizgig.families.registry import get as get_family
    from fizgig.families.launch import option_tokens

    desc = get_family("krea2")
    assert [option.key for option in desc.options] == ["KREA2_TRAINING_METHOD", "KREA2_SELECTIVE_BLOCK_CHECK"]
    method, check = desc.options
    assert method.choice_labels() == ["Standard", "Global replay", "Text fusion only"] and method.default == "Standard"
    assert check.kind == "check" and check.default == "1"

    def tokens(values):
        return option_tokens(desc, {"FAMILY_OPTIONS": values})[0]

    # Standard sends nothing (it trains exactly as before), and the check is not sent with it
    assert tokens({}) == []
    assert tokens({"KREA2_TRAINING_METHOD": "Standard", "KREA2_SELECTIVE_BLOCK_CHECK": "1"}) == []
    assert tokens({"KREA2_TRAINING_METHOD": "Global replay", "KREA2_SELECTIVE_BLOCK_CHECK": "1"}) == [
        "--family_option", "training_method=global_replay", "--family_option", "selective_block_check=1"]
    assert tokens({"KREA2_TRAINING_METHOD": "Global replay", "KREA2_SELECTIVE_BLOCK_CHECK": ""}) == [
        "--family_option", "training_method=global_replay"]
    # Text fusion only sends its method; the Selective check stays with Global replay and is not sent
    assert tokens({"KREA2_TRAINING_METHOD": "Text fusion only", "KREA2_SELECTIVE_BLOCK_CHECK": "1"}) == [
        "--family_option", "training_method=text_fusion"]
    # the test build's SGR label lands on Global replay; its dropped methods land on Standard
    assert method.pick("Selective global replay H2D (experimental)") == "Global replay"
    for gone in ("Standard backprop", "Blockwise flow (experimental)", "Blockwise flow H2D (experimental)",
                 "Block-local flow matching (experimental)", "Residual-path hybrid (experimental)",
                 "Residual-path hybrid H2D (experimental)", "Residual-path hybrid H2D — selective A (experimental)"):
        assert method.pick(gone) == "Standard", gone


def test_driver_method_names_toggle_and_dropped_methods():
    def driver_for(options):
        driver = Krea2Driver()
        driver.set_options(options)
        return driver

    on = driver_for({"training_method": "global_replay", "selective_block_check": "1"})
    off = driver_for({"training_method": "global_replay"})
    assert on.global_replay and on.selective_global_replay and on.manual_loss_backward
    assert off.global_replay and not off.selective_global_replay and off.manual_loss_backward
    for old in ("selective_global_replay", "selective_global_replay_h2d"):    # the test build's names: check on
        assert driver_for({"training_method": old}).selective_global_replay is True
    fusion = driver_for({"training_method": "text_fusion"})
    assert fusion.text_fusion_only and not fusion.global_replay and not fusion.selective_global_replay
    assert driver_for({"training_method": "text_fusion_only"}).text_fusion_only
    for gone in ("", "block_local", "block_local_h2d", "residual_path_hybrid", "residual_path_hybrid_h2d_selective_a"):
        driver = driver_for({"training_method": gone})
        assert not driver.global_replay and not driver.selective_global_replay and not driver.manual_loss_backward
        assert not driver.text_fusion_only
    with pytest.raises(RuntimeError):
        dit, net = _selective_tiny_dit(layers=2)
        off.configure_selective_global_replay(dit, net)


def test_block_check_off_trains_every_block_with_no_probes():
    dit, net = _selective_tiny_dit(layers=2)
    driver = Krea2Driver()
    driver.set_options({"training_method": "global_replay"})
    latents = torch.randn(1, 2, 4, 4)
    cond = {"hidden_states": torch.randn(1, 3, 1, 16), "attention_mask": torch.tensor([[True, True, True]])}
    for step in range(20):
        loss, info = driver.training_loss(dit, latents, cond, torch.Generator().manual_seed(step))
        assert info["block_local_phase"] == "global_replay"
        driver.backward_training_loss()
        assert driver.selective_snapshot_before_step() is None
        assert driver.finish_selective_optimizer_step(None) is None
        assert driver.selective_state_dict() is None
        for layer in net.wrapped.values():
            assert all(p.grad is not None and p.requires_grad for p in layer.adapters.parameters())
            for p in layer.adapters.parameters():
                p.grad = None


def test_selective_global_replay_state_round_trips():
    dit, net = _selective_tiny_dit(layers=2)
    driver = Krea2Driver()
    driver.set_options({"training_method": "global_replay", "selective_block_check": "1"})
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
        ("global_replay", None),
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
    assert replay_driver.global_replay is True
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


def _run_selective_with_real_optimizer(driver, block_params, targets, curvature, lr, updates):
    """Drive the selective scorer with a real SGD optimizer on a per-block quadratic loss. A block whose
    curvature * lr > 1 overshoots its minimum on every step, so its loss climbs: it is genuinely uphill."""
    params = [p for values in block_params.values() for p in values]
    optimizer = torch.optim.SGD(params, lr=lr)
    statuses = []
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        loss = 0.0
        for index, values in block_params.items():
            for parameter, target in zip(values, targets[index]):
                if parameter.requires_grad:
                    loss = loss + 0.5 * curvature[index] * ((parameter - target) ** 2).sum()
        loss.backward()
        snapshot = driver.selective_snapshot_before_step()
        optimizer.step()
        statuses.append(driver.finish_selective_optimizer_step(snapshot))
    return statuses


def test_selective_global_replay_drops_a_block_whose_real_optimizer_steps_go_uphill():
    torch.manual_seed(3)
    dit, net = _selective_tiny_dit(layers=2)
    driver = Krea2Driver()
    driver.set_options({"training_method": "global_replay", "selective_block_check": "1"})
    driver.configure_selective_global_replay(dit, net, warmup_updates=4, reactivate_every=4, score_decay=0.5)
    block_params = driver._selective_params_by_block
    targets = {i: [torch.randn_like(p) for p in values] for i, values in block_params.items()}
    # lr 0.1: block 0 (curvature 1) descends smoothly; block 1 (curvature 15, lr*curvature = 1.5) overshoots.
    statuses = _run_selective_with_real_optimizer(driver, block_params, targets, {0: 1.0, 1: 15.0}, 0.1, 4)

    assert statuses[-1]["kind"] == "probe"
    assert driver._selective_scores[0] > 0
    assert driver._selective_scores[1] < 0
    assert driver.selective_active_blocks() == [0]
    assert all(not p.requires_grad for p in block_params[1])


def test_selective_global_replay_keeps_every_block_that_descends():
    torch.manual_seed(4)
    dit, net = _selective_tiny_dit(layers=2)
    driver = Krea2Driver()
    driver.set_options({"training_method": "global_replay", "selective_block_check": "1"})
    driver.configure_selective_global_replay(dit, net, warmup_updates=4, reactivate_every=4, score_decay=0.5)
    block_params = driver._selective_params_by_block
    targets = {i: [torch.randn_like(p) for p in values] for i, values in block_params.items()}
    _run_selective_with_real_optimizer(driver, block_params, targets, {0: 1.0, 1: 2.0}, 0.1, 12)
    assert driver.selective_active_blocks() == [0, 1]
