"""CPU checks for Krea 2's Text fusion only training method."""

import pytest

from fizgig.families.launch import option_tokens
from fizgig.families.lora import TRAINABLE, FamilyLoRA
from fizgig.families.registry import get as get_family
from fizgig.families.train import resolve_krea2_training_method
from fizgig.krea2.driver import TEXT_FUSION_BLOCK_IDS
from fizgig.krea2.model import SingleMMDiTConfig, SingleStreamDiT

_BLOCK_MODULES = ("attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo", "mlp.gate", "mlp.up", "mlp.down")
TEXT_FUSION_TARGETS = tuple(
    f"txtfusion.{stack}_blocks.{i}.{mod}"
    for stack in ("layerwise", "refiner")
    for i in (0, 1)
    for mod in _BLOCK_MODULES
)
assert len(TEXT_FUSION_TARGETS) == 32


def _tiny_dit(layers=28):
    config = SingleMMDiTConfig(
        features=16, tdim=16, txtdim=16, heads=1, multiplier=2, layers=layers,
        patch=1, channels=2, bias=True, txtlayers=1, txtheads=1, txtkvheads=1,
    )
    return SingleStreamDiT(config, attn_mode="torch", split_attn=False).eval().requires_grad_(False)


def _krea2_driver(options=None):
    driver = get_family("krea2").load_driver()
    if options:
        driver.set_options(options)
    else:
        driver.set_options({})
    return driver


def test_text_fusion_only_is_a_training_method_choice_with_rank_hint():
    desc = get_family("krea2")
    method = next(o for o in desc.options if o.key == "KREA2_TRAINING_METHOD")
    assert method.choice_labels() == ["Standard", "Global replay", "Text fusion only"]
    note = dict(method.choice_notes)["Text fusion only"]
    assert "four text-fusion blocks" in note
    assert "frozen" in note.lower() and "small" in note.lower()
    assert "standard LoRA" in note and "Rank 64" in note
    assert "Global replay" in note
    tokens = option_tokens(desc, {"FAMILY_OPTIONS": {"KREA2_TRAINING_METHOD": "Text fusion only",
                                                     "KREA2_SELECTIVE_BLOCK_CHECK": "1"}})[0]
    assert tokens == ["--family_option", "training_method=text_fusion"]


def test_text_fusion_only_forces_the_four_blocks_and_rejects_conflicts(caplog):
    import logging

    fusion = _krea2_driver({"training_method": "text_fusion"})
    assert resolve_krea2_training_method(fusion, {"training_method": "text_fusion"}) == list(TEXT_FUSION_BLOCK_IDS)
    # the same four blocks passed explicitly are accepted
    assert resolve_krea2_training_method(
        fusion, {"training_method": "text_fusion"}, list(TEXT_FUSION_BLOCK_IDS)
    ) == list(TEXT_FUSION_BLOCK_IDS)
    assert resolve_krea2_training_method(
        _krea2_driver({"training_method": "text_fusion_only"}), {"training_method": "text_fusion_only"}
    ) == list(TEXT_FUSION_BLOCK_IDS)

    with pytest.raises(RuntimeError, match="can't be combined with Global replay"):
        resolve_krea2_training_method(
            _krea2_driver({"training_method": "global_replay"}),
            {"training_method": "global_replay"},
            list(TEXT_FUSION_BLOCK_IDS),
        )
    with pytest.raises(RuntimeError, match="standard LoRA"):
        resolve_krea2_training_method(fusion, {"training_method": "text_fusion"}, network_type="lokr")
    with pytest.raises(RuntimeError, match="slider training"):
        resolve_krea2_training_method(fusion, {"training_method": "text_fusion"}, slider_pairs=True)
    with pytest.raises(RuntimeError, match="fine-tuning"):
        resolve_krea2_training_method(fusion, {"training_method": "text_fusion"}, finetune=True)
    with pytest.raises(RuntimeError, match="different --train_blocks"):
        resolve_krea2_training_method(
            fusion, {"training_method": "text_fusion"}, ["block_0", "block_1"])
    with pytest.raises(RuntimeError, match="Unknown Krea 2 training method 'not_a_method'"):
        resolve_krea2_training_method(_krea2_driver({"training_method": "not_a_method"}),
                                      {"training_method": "not_a_method"})

    # dropped test-build methods still fall back to Standard, with a log note
    with caplog.at_level(logging.WARNING):
        driver = _krea2_driver({"training_method": "block_local"})
        assert resolve_krea2_training_method(driver, {"training_method": "block_local"}) is None
        assert "training_method" not in driver.options
    assert "no longer offered" in caplog.text


def test_text_fusion_only_lora_targets_and_kohya_keys(tmp_path):
    dit = _tiny_dit()
    driver = _krea2_driver({"training_method": "text_fusion"})
    blocks = set(TEXT_FUSION_BLOCK_IDS)
    known = {b.id for g in driver.block_map(dit) for b in g.blocks}
    assert blocks <= known
    assert {f"block_{i}" for i in range(28)} <= known

    net = FamilyLoRA(dit, driver)
    net.add_trainable(2, 2, blocks=blocks, kind="lora")
    trainable = sorted(full for full, wrapped in net.wrapped.items() if TRAINABLE in wrapped.adapters)
    assert trainable == sorted(TEXT_FUSION_TARGETS)

    forbidden_prefixes = ("blocks.", "txtfusion.projector", "txtmlp.", "first", "tmlp.", "tproj.", "last.")
    for name in trainable:
        assert not any(name == p or name.startswith(p) for p in forbidden_prefixes), name
    # the 28 main blocks, the projector, txtmlp and the rest of the I/O group stay frozen
    for name in ("txtfusion.projector", "txtmlp.1", "txtmlp.3", "first", "last.linear"):
        assert name not in set(trainable)
    for i in range(28):
        assert not any(n.startswith(f"blocks.{i}.") for n in trainable)

    path = tmp_path / "text_fusion_only.safetensors"
    net.save(str(path))
    from safetensors.torch import load_file
    sd = load_file(str(path))

    stems = set()
    for key in sd:
        assert key.startswith("lora_unet_txtfusion_"), key
        assert "lora_unet_blocks_" not in key
        assert "txtmlp" not in key and "projector" not in key
        if key.endswith(".lora_down.weight"):
            stems.add(key[: -len(".lora_down.weight")])
    assert len(stems) == 32
    for stem in stems:
        assert f"{stem}.lora_up.weight" in sd
        assert f"{stem}.alpha" in sd

    mapped = []
    for stem in sorted(stems):
        full = net._module_for(stem)
        assert full is not None, stem
        mapped.append(full)
        flat = stem[len("lora_unet_"):]
        # the family's own keys already use txtfusion_; the alias table still maps the
        # ComfyUI / diffusers spellings back onto the same modules
        assert driver.alias_flat(flat) in (None, flat) or net._module_for("lora_unet_" + driver.alias_flat(flat)) == full
    assert sorted(mapped) == sorted(TEXT_FUSION_TARGETS)

    # a typical ComfyUI / diffusers alias for one of the 32 modules maps back
    alias_stem = "lora_unet_text_fusion_layerwise_blocks_0_attn_to_q"
    alias_flat = alias_stem[len("lora_unet_"):]
    renamed = driver.alias_flat(alias_flat)
    assert renamed == "txtfusion_layerwise_blocks_0_attn_wq"
    assert net._module_for(alias_stem) == "txtfusion.layerwise_blocks.0.attn.wq"
    assert net._module_for("lora_unet_" + renamed) == "txtfusion.layerwise_blocks.0.attn.wq"
