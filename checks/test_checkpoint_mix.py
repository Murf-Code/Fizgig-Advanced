"""CPU-only tests for experimental same-architecture base model checkpoint mixing.

No model download or GPU required. The tests cover exact copies, float blending,
quantization/layout refusal, cancellation, and safe atomic outputs.
"""
import json
import threading

import pytest

torch = pytest.importorskip("torch")
safe = pytest.importorskip("safetensors.torch")

from fizgig.families import checkpoint_mix as mix


def fixture_pair(tmp_path, family="anima", dtype=torch.bfloat16, extra=False):
    if family == "moss_image":
        ids = ["model.layers.0.attn.weight", "model.layers.1.attn.weight"]
    elif family == "sdxl":
        ids = ["model.diffusion_model.input_blocks.4.1.weight",
               "model.diffusion_model.output_blocks.0.1.weight"]
    else:
        ids = ["diffusion_model.blocks.0.attn.weight", "diffusion_model.blocks.1.attn.weight"]
    base, donor = tmp_path / "base.safetensors", tmp_path / "donor.safetensors"
    a = {ids[0]: torch.arange(8, dtype=torch.float32).reshape(2, 4).to(dtype),
         ids[1]: torch.arange(8, dtype=torch.float32).reshape(2, 4).to(dtype),
         "context.proj.weight": torch.ones((2, 3), dtype=torch.float32)}
    b = {ids[0]: torch.full((2, 4), 10., dtype=dtype),
         ids[1]: torch.full((2, 4), 15., dtype=dtype),
         "context.proj.weight": torch.full((2, 3), 99., dtype=torch.float32)}
    if extra:
        b["context.other_unmatched.weight"] = torch.ones(2)
    safe.save_file(a, str(base), metadata={"author": "original"})
    safe.save_file(b, str(donor))
    return base, donor, ids, a, b


@pytest.mark.parametrize("family", ["moss_image", "anima", "krea2", "sdxl"])
def test_inspect_and_mix_compatible_family(tmp_path, family, monkeypatch):
    base, donor, ids, a, b = fixture_pair(tmp_path, family)
    plan = mix.inspect_pair(str(base), str(donor), family)
    assert plan["compatible"]
    assert len(plan["blocks"]) == 2
    assert plan["outside_blocks"] == 1
    selected = plan["blocks"][0]["id"]
    second = plan["blocks"][1]["id"]
    out = tmp_path / "output.safetensors"
    seen = []
    monkeypatch.setattr(mix, "CHUNK_BYTES", 16)  # multiple tiny bounded chunks
    result = mix.merge_pair(str(base), str(donor), str(out), family,
                            {selected: 0.5, second: 0.0}, progress=lambda d, t: seen.append((d, t)))
    assert result["ok"] and result["groups_changed"] == 1
    assert seen and seen[-1][0] == seen[-1][1]
    loaded = safe.load_file(str(out))
    assert torch.equal(loaded["context.proj.weight"], a["context.proj.weight"])
    for key in ids:
        strength = 0.5 if mix.block_of(family, key) == selected else 0
        want = (a[key].float() * (1-strength) + b[key].float() * strength).to(a[key].dtype)
        assert torch.equal(loaded[key], want)
    from safetensors import safe_open
    with safe_open(str(out), framework="pt") as f:
        metadata = f.metadata()
    assert metadata["author"] == "original"
    info = json.loads(metadata["fizgig_checkpoint_mix"])
    assert info["family"] == family and info["blocks"] == {selected: 0.5}


def test_donor_endpoint_preserves_exact_tensor_bytes(tmp_path):
    base, donor, ids, a, b = fixture_pair(tmp_path, dtype=torch.float16)
    plan = mix.inspect_pair(str(base), str(donor), "anima")
    out = tmp_path / "full.safetensors"
    mix.merge_pair(str(base), str(donor), str(out), "anima",
                   {g["id"]: 1.0 for g in plan["blocks"]})
    loaded = safe.load_file(str(out))
    for k in ids:
        assert torch.equal(loaded[k], b[k])
    assert torch.equal(loaded["context.proj.weight"], a["context.proj.weight"])


def test_non_block_keys_may_differ_without_merging_unsafe_parts(tmp_path):
    base, donor, ids, a, b = fixture_pair(tmp_path, extra=True)
    plan = mix.inspect_pair(str(base), str(donor), "anima")
    output = tmp_path / "output.safetensors"
    mix.merge_pair(str(base), str(donor), str(output), "anima",
                   {plan["blocks"][0]["id"]: 0.25})
    loaded = safe.load_file(str(output))
    assert "context.other_unmatched.weight" not in loaded


def test_bad_layouts_and_quantized_weights_are_rejected(tmp_path):
    base, donor, ids, a, b = fixture_pair(tmp_path)
    del b[ids[0]]
    safe.save_file(b, str(donor))
    with pytest.raises(ValueError, match="expected Anima blocks|Different block layouts"):
        mix.inspect_pair(str(base), str(donor), "anima")
    b[ids[0]] = torch.ones((3, 4), dtype=torch.bfloat16)
    safe.save_file(b, str(donor))
    with pytest.raises(ValueError, match="shape or dtype"):
        mix.inspect_pair(str(base), str(donor), "anima")
    a[ids[0]] = torch.ones((2,4), dtype=torch.int8)
    b[ids[0]] = torch.ones((2,4), dtype=torch.int8)
    safe.save_file(a, str(base))
    safe.save_file(b, str(donor))
    with pytest.raises(ValueError, match="Quantized"):
        mix.inspect_pair(str(base), str(donor), "anima")


def test_invalid_family_and_pair_are_rejected(tmp_path):
    base, donor, ids, a, b = fixture_pair(tmp_path)
    with pytest.raises(ValueError, match="supported"):
        mix.inspect_pair(str(base), str(donor), "minimax")
    with pytest.raises(ValueError, match="different checkpoints"):
        mix.inspect_pair(str(base), str(base), "anima")
    with pytest.raises(ValueError, match="expected"):
        mix.inspect_pair(str(base), str(donor), "moss_image")


def test_atomic_output_and_existing_file_protection(tmp_path):
    base, donor, ids, a, b = fixture_pair(tmp_path)
    plan = mix.inspect_pair(str(base), str(donor), "anima")
    w = {plan["blocks"][0]["id"]: 0.5}
    out = tmp_path / "out.safetensors"
    out.write_bytes(b"keep-me")
    with pytest.raises(FileExistsError):
        mix.merge_pair(str(base), str(donor), str(out), "anima", w)
    assert out.read_bytes() == b"keep-me"
    mix.merge_pair(str(base), str(donor), str(out), "anima", w, overwrite=True)
    assert safe.load_file(str(out))
    assert not list(tmp_path.glob(".fizgig-mix-*.tmp"))
    with pytest.raises(ValueError, match="cannot replace"):
        mix.merge_pair(str(base), str(donor), str(base), "anima", w)


def test_cancel_before_start_and_cleanup(tmp_path):
    base, donor, ids, a, b = fixture_pair(tmp_path)
    plan = mix.inspect_pair(str(base), str(donor), "anima")
    ev = threading.Event()
    ev.set()
    output = tmp_path / "cancel.safetensors"
    with pytest.raises(mix.MixCancelled):
        mix.merge_pair(str(base), str(donor), str(output), "anima",
                       {plan["blocks"][0]["id"]: 0.8}, cancel=ev)
    assert not output.exists()
    assert not list(tmp_path.glob(".fizgig-mix-*.tmp"))


def test_weights_must_be_in_bounds_and_nonzero(tmp_path):
    base, donor, ids, a, b = fixture_pair(tmp_path)
    plan = mix.inspect_pair(str(base), str(donor), "anima")
    bid = plan["blocks"][0]["id"]
    for value in [-0.1, float("nan"), 1.2]:
        with pytest.raises(ValueError, match="between 0 and 1"):
            mix.merge_pair(str(base), str(donor), str(tmp_path / "out.safetensors"),
                           "anima", {bid: value})
    with pytest.raises(ValueError, match="Nothing to mix"):
        mix.merge_pair(str(base), str(donor), str(tmp_path / "out.safetensors"),
                       "anima", {bid: 0.0})
