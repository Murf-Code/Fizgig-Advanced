"""Fewer cached tag variants: variants that read the same are encoded and stored once (one reading as the caption
itself trains on the item's own cache), and a per-picture cap (default 8, the empty caption included) bounds how many
a caption gets. Old variants files still load."""
import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fizgig.dataset import tag_variants as TV


def _enc(log):
    def enc(caps):
        log.extend(caps)
        return [{"hidden_states": torch.full((4, 2), float(len(c))), "attention_mask": torch.ones(4, dtype=torch.bool)}
                for c in caps]
    return enc


def _te(tmp_path, cap):
    p = str(tmp_path / "img_krea2_te.safetensors")
    save_file({"cond__hidden_states": torch.zeros(4, 2), "cond__attention_mask": torch.ones(4, dtype=torch.bool)},
              p, metadata={"caption1": cap})
    return p


def test_few_tags_give_few_distinct_variants_encoded_once(tmp_path):
    cap = "trig, a, b"                                           # 2 tail tags: only 2 orders exist
    te = _te(tmp_path, cap)
    s = TV.TagSettings.from_params(shuffle=True, variants=8)
    texts = TV.variant_captions(cap, s)
    assert sorted(texts) == sorted({"trig, a, b", "trig, b, a"}) and len(texts) == len(set(texts))
    log = []
    assert TV.write_variants(_enc(log), te, cap, s) == 1         # only the order that differs from the caption
    assert log == ["trig, b, a"]
    with safe_open(TV.variants_path(te), framework="pt") as f:
        md, keys = f.metadata(), list(f.keys())
    same = json.loads(md["orig"])
    assert len(same) == 1 and json.loads(md["captions"])[same[0]] == cap
    assert not any(k.startswith(f"v{same[0]}__") for k in keys)      # never stored twice
    vp = TV.VariantPicker(s, seed=3)
    picks = {vp.choose("img", te, e) for e in range(40)}
    assert picks == {TV.ORIG, 1 - same[0]}                        # both orders train, the caption's from its cache
    assert TV.VariantPicker.load(te, 1 - same[0])["cond__hidden_states"][0, 0] == len("trig, b, a")


def test_the_cap_bounds_variants_per_picture_and_old_files_load(tmp_path):
    cap = "trig, " + ", ".join(f"t{i}" for i in range(12))
    s = TV.TagSettings.from_params(shuffle=True, variants=16, caption_dropout=0.1, cap=8)
    assert s.wanted == 7 and len(TV.variant_captions(cap, s)) == 7  # 7 variants + the empty caption = 8
    assert TV.TagSettings.from_params(shuffle=True, variants=4).wanted == 4      # the default cap leaves 4 as it was
    # an old file (every variant stored, no "orig"): picked and loaded as before
    te = _te(tmp_path, cap)
    old = TV.TagSettings.from_params(shuffle=True, variants=2)
    texts = TV.variant_captions(cap, old)
    save_file({f"v{i}__hidden_states": torch.full((4, 2), float(i + 1)) for i in range(2)}, TV.variants_path(te),
              metadata={"fizgig_tags": "1", "source": cap, "recipe": old.recipe(), "captions": json.dumps(texts),
                        "empty": "0"})
    vp = TV.VariantPicker(old)
    assert {vp.choose("img", te, e) for e in range(30)} == {0, 1}
    assert TV.VariantPicker.load(te, 1)["cond__hidden_states"][0, 0] == 2.0


def test_form_and_launch_carry_the_cap():
    from fizgig.families import form, launch
    from fizgig.families.registry import get as get_family
    desc = get_family("anima")
    v = {**form.DEFAULTS, "TAG_SHUFFLE": True, "TAG_VARIANTS": "16", "TAG_VARIANT_CAP": "8"}
    t = launch.tag_settings(desc, v)
    assert t["caption_variant_cap"] == 8 and "caption_variant_cap = 8" in launch._tag_lines(desc, v)
    assert "up to 8 variant(s)" in form.tag_cost_note(v, 10)
    assert launch.tag_problems(desc, {**v, "TAG_VARIANT_CAP": "1"})
