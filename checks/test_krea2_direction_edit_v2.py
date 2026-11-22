"""Checks for Krea 2 Text fusion direction edit v2 (fizgig.krea2.direction_edit_v2).

Runs under pytest or directly (python checks/test_krea2_direction_edit_v2.py). Uses a tiny text path with random
weights and a fake encoder, so it needs neither the Krea 2 checkpoint nor Qwen3-VL.
"""
import hashlib
import os
import random
import sys
import tempfile
import traceback

import torch

from fizgig.krea2 import direction_edit_v2 as de
from fizgig.krea2.model import SingleMMDiTConfig

D, L = 64, 3          # text width and number of selected encoder layers in the tiny model


def tiny_config():
    return SingleMMDiTConfig(features=32, tdim=16, txtdim=D, heads=4, multiplier=2, layers=1, patch=2, channels=4,
                             txtlayers=L, txtheads=4, txtkvheads=4)


def tiny_path(seed=0):
    torch.manual_seed(seed)
    return de.FusionPath(tiny_config()).eval().requires_grad_(False)


def _vec(tag, scale):
    g = torch.Generator().manual_seed(int(hashlib.md5(tag.encode()).hexdigest()[:8], 16))
    return torch.randn(D, generator=g) * scale


def fake_encoder(weak=()):
    """prompt -> hidden states (tokens, L, D): two tokens per comma-separated tag (a tag's vectors are small when it is
    in `weak`), plus a fixed 4-token suffix, with a running-mean mix so later tokens depend on earlier ones."""
    weak = {w.lower() for w in weak}

    def encode(prompts):
        out = []
        for p in prompts:
            toks = []
            for t in [x.strip() for x in p.split(",") if x.strip()]:
                s = 0.15 if t.lower() in weak else 1.0
                toks += [_vec(t + "#0", s), _vec(t + "#1", s)]
            toks += [_vec(f"suffix{i}", 1.0) for i in range(4)]
            x = torch.stack(toks)
            x = x + 0.3 * torch.cumsum(x, 0) / torch.arange(1, len(x) + 1)[:, None]
            h = torch.stack([x * (1 + 0.15 * l) + 0.05 * _vec(f"layer{l}", 1.0) for l in range(L)], 1)
            out.append({"hidden_states": h.to(torch.float32), "attention_mask": None})
        return out
    return encode


def make_folder(n=120, seed=5, extra_images=True, readme=True):
    """A caption folder: 'tgt a' in some captions, plenty of filler tags, a README and (optionally) pictures."""
    rng = random.Random(seed)
    d = tempfile.mkdtemp(prefix="de_v2_caps_")
    common = [f"tag{i:02d}" for i in range(40)]
    for i in range(n):
        tags = rng.sample(common, 9)
        if i % 3 == 0:
            tags.append("tgt a")
        rng.shuffle(tags)
        open(os.path.join(d, f"img{i:03d}.txt"), "w").write(", ".join(tags))
        if extra_images and i % 2 == 0:
            open(os.path.join(d, f"img{i:03d}.png"), "wb").write(b"\x89PNG")
    if readme:
        open(os.path.join(d, "README.txt"), "w").write("not a caption")
    open(os.path.join(d, "orphan.txt"), "w").write("tag01, tag02, tag03, tag04, tag05")   # no picture beside it
    return d


def caption_cfg(folder, **over):
    cfg = de.make_config({"direction_edit_caption_folder": folder, "direction_edit_target_tags": "tgt a",
                          "direction_edit_caption_count": "10"})
    cfg.update(over)
    return de._normalise(cfg)


# --- captions ---------------------------------------------------------------------------------------------------
def test_read_captions_ignores_images():
    d = make_folder()
    caps = de.read_captions(d)
    assert "README.txt" not in caps
    assert "orphan.txt" in caps, "a caption with no picture beside it must count: pictures are ignored"
    assert len(caps) == 121


def test_apply_captions_matches_v1_when_one_round_and_no_holdout():
    try:
        from fizgig.krea2 import direction_edit as v1
    except ImportError:
        return
    d = make_folder(extra_images=False, readme=False)    # v1 pairs captions with pictures and has no notes filter
    a = caption_cfg(d, verify_count=0, rounds=1, control_tags=["tag01", "tag02", "tag03"])
    b = v1.make_config({"direction_edit_caption_folder": d, "direction_edit_target_tags": "tgt a",
                        "direction_edit_caption_count": "10"})
    b["control_tags"] = ["tag01", "tag02", "tag03"]
    de.apply_captions(a)
    v1.apply_captions(b)
    assert a["base_prompts"] == b["base_prompts"], "round 0 must draw the same captions as v1"
    assert a["control_tags"] == b["control_tags"]


def test_rounds_and_holdout_are_disjoint_and_controls_rotate():
    d = make_folder()
    cfg = caption_cfg(d, rounds=3, verify_count=6)
    de.apply_captions(cfg)
    plan = cfg["round_plan"]
    assert len(plan) == 3
    files = [set(r["sampled_files"]) for r in plan]
    held = set(cfg["caption_source"]["verify_files"])
    assert len(held) == 6
    for i in range(3):
        assert len(files[i]) == 10
        assert not files[i] & held
        for j in range(i + 1, 3):
            assert not files[i] & files[j]
    ctl = [set(r["control_tags"]) for r in plan]
    assert all(len(c) >= 3 for c in ctl)
    assert ctl[0] != ctl[1] and not ctl[0] & ctl[1], "auto controls must rotate between rounds"
    assert set(cfg["control_tags"]) == ctl[0] | ctl[1] | ctl[2]
    de.check_config(cfg)
    prompts = de.prompt_list(cfg)
    assert all(p in prompts for p in plan[2]["base_prompts"])
    assert all(p in prompts for p in cfg["verify_prompts"])


def test_check_config_rounds_needs_caption_mode():
    cfg = de.make_config()
    cfg["rounds"] = 2
    try:
        de.check_config(cfg)
    except ValueError as e:
        assert "caption mode" in str(e)
    else:
        raise AssertionError("rounds > 1 without caption mode must be refused")


# --- the edit ---------------------------------------------------------------------------------------------------
def test_edit_delta_sign_columns_and_activation_scale():
    torch.manual_seed(1)
    w = torch.randn(16, 40)
    u, _ = torch.linalg.qr(torch.randn(16, 2))
    delta, idx = de.edit_delta(w, u, 1.0, 8)
    proj = u.t() @ w
    assert torch.allclose((u.t() @ (w + delta))[:, idx], 2 * proj[:, idx], atol=1e-5), "strength +1 amplifies"
    other = torch.ones(40, dtype=torch.bool)
    other[idx] = False
    assert float(delta[:, other].abs().max()) == 0.0
    neg, _ = de.edit_delta(w, u, -1.0, 8)
    assert torch.allclose(neg, -delta, atol=1e-6)
    assert torch.allclose((u.t() @ (w + neg))[:, idx], torch.zeros(2, 8), atol=1e-5), "strength -1 removes it"
    scale = torch.ones(40)
    scale[idx] = 0.0                                  # the weight-ranked columns never fire
    _, idx2 = de.edit_delta(w, u, 1.0, 8, scale=scale)
    assert not set(idx2.tolist()) & set(idx.tolist()), "activation scale must move the chosen columns"


def test_selection_energy_is_a_fraction():
    torch.manual_seed(2)
    w = torch.randn(16, 40)
    u, _ = torch.linalg.qr(torch.randn(16, 1))
    _, idx = de.edit_delta(w, u, 1.0, 8)
    e = de.selection_energy(w, u, idx, scale=torch.rand(40))
    assert 0 < e["projection_energy_removed"] < 1 and 0 < e["activation_energy_removed"] < 1
    _, all_idx = de.edit_delta(w, u, 1.0, 0)
    assert abs(de.selection_energy(w, u, all_idx)["projection_energy_removed"] - 1) < 1e-6


def test_input_scales_reads_the_edited_linears():
    path = tiny_path()
    enc = fake_encoder()([f"a{i}, b{i}, c{i}" for i in range(3)])
    store = {f"p{i}": {"hidden_states": enc[i]["hidden_states"], "attention_mask": None} for i in range(3)}
    names = ["txtfusion.refiner_blocks.0.attn.wo", "txtfusion.refiner_blocks.0.mlp.down"]
    sc = de.input_scales(path, store, list(store), names)
    assert set(sc) == set(names)
    assert sc[names[1]].shape[0] == path.get_submodule(names[1]).weight.shape[1]
    assert float(sc[names[0]].min()) >= 0


# --- storage ----------------------------------------------------------------------------------------------------
def test_encoded_store_spills_and_cleans_up():
    store = de.EncodedStore(budget_bytes=1000)
    hs = {f"p{i}": torch.randn(10, L, D) for i in range(6)}
    for k, h in hs.items():
        store[k] = {"hidden_states": h, "attention_mask": None}
    assert store.spilled >= 5 and len(store) == 6
    for k, h in hs.items():
        assert k in store and torch.equal(store[k]["hidden_states"], h)
    folder = store._dir
    assert os.path.isdir(folder)
    store.close()
    assert not os.path.exists(folder) and len(store) == 0


def test_encode_prompts_chunks_and_skips_done():
    calls = []
    enc = fake_encoder()

    def counting(part):
        calls.append(len(part))
        return enc(part)
    store = de.encode_prompts([f"x{i}, y" for i in range(70)], None, "cpu", encoder=counting, chunk=32)
    assert calls == [32, 32, 6] and len(store) == 70
    de.encode_prompts([f"x{i}, y" for i in range(75)], None, "cpu", encoder=counting, chunk=32, store=store)
    assert calls[3:] == [5], "prompts already stored are not encoded again"


# --- the measurement and the edit, end to end -------------------------------------------------------------------
def _store_for(cfg, enc):
    return de.encode_prompts(de.prompt_list(cfg), None, "cpu", encoder=enc)


def test_run_on_path_heldout_signed_verify_and_exact_lora():
    d = make_folder()
    cfg = caption_cfg(d, verify_count=6, columns=16, rank=4)
    de.apply_captions(cfg)
    path = tiny_path()
    store = _store_for(cfg, fake_encoder(weak={"tgt a"}))
    res = de.run_on_path(path, store, cfg)
    v = res["verify"]
    assert v["held_out"] is True and v["verify_captions"] == 6
    assert abs(v["improvement"] + v["change"]) < 1e-9 and abs(v["random_improvement"] + v["random_change"]) < 1e-9
    assert v["beats_random"] == bool(v["improvement"] > 0 and v["improvement"] > v["random_improvement"])
    assert set(res["edit"]) == {f"{de.BLOCK_PATH[b]}.{lin}" for b in cfg["edit_blocks"] for lin in cfg["edit_linears"]}
    for name, (delta, idx) in res["edit"].items():
        assert float(delta.norm()) > 0
        assert abs(res["random_delta_norm"][name] - float(delta.norm())) < 1e-4 * float(delta.norm()), \
            "the random control must be as large as the real edit"
        assert 0 < res["edit_stats"][name][0]["projection_energy_removed"] <= 1
    sd, err = de.lora_state_dict(res["edit"], 4)
    assert max(err.values()) < 1e-4, "a rank-4 LoRA must reproduce a one-direction edit exactly"
    assert any(k.startswith("lora_unet_txtfusion_refiner_blocks_0_attn_wo") for k in sd)


def test_score_stage_is_honoured():
    d = make_folder()
    cfg = caption_cfg(d, columns=16, score_stage="rf_1")
    de.apply_captions(cfg)
    res = de.run_on_path(tiny_path(), _store_for(cfg, fake_encoder(weak={"tgt a"})), cfg)
    row = res["scores"]["before"]["tgt a"]
    assert abs(row["suppression"] + row["z"]["rf_1"]) < 1e-12
    bad = dict(cfg, score_stage="nope")
    try:
        de._normalise(bad)
    except ValueError as e:
        assert "score_stage" in str(e)
    else:
        raise AssertionError("an unknown stage must be refused")


def test_strength_zero_is_a_no_op_and_cannot_beat_random():
    d = make_folder()
    cfg = caption_cfg(d, strength=0.0, columns=16)
    de.apply_captions(cfg)
    res = de.run_on_path(tiny_path(), _store_for(cfg, fake_encoder()), cfg)
    assert res["verify"]["improvement"] == 0 and res["verify"]["beats_random"] is False
    assert all(float(dl.norm()) == 0 for dl, _ in res["edit"].values())


def test_blocks_are_measured_on_the_edited_path():
    """rf_1's direction is measured after rf_0's edit: with rf_0 edited at strength 0 or >0 it must differ."""
    d = make_folder()
    store_cfg = caption_cfg(d, columns=16)
    de.apply_captions(store_cfg)
    path, enc = tiny_path(), fake_encoder(weak={"tgt a"})
    store = _store_for(store_cfg, enc)
    a = de.run_on_path(path, store, {**store_cfg, "strength": 0.0})
    b = de.run_on_path(path, store, {**store_cfg, "strength": 3.0})
    ua, ub = torch.tensor(a["directions"]["txt_rf_1"]), torch.tensor(b["directions"]["txt_rf_1"])
    assert not torch.allclose(ua, ub, atol=1e-6), "rf_1 must see the edited rf_0"
    assert torch.allclose(torch.tensor(a["directions"]["txt_rf_0"]), torch.tensor(b["directions"]["txt_rf_0"]), atol=1e-6)


def test_rounds_sum_exactly_and_raise_the_rank():
    d = make_folder()
    cfg = caption_cfg(d, rounds=2, verify_count=4, columns=16, rank=1, strength=1.0)
    de.apply_captions(cfg)
    res = de.run_on_path(tiny_path(), _store_for(cfg, fake_encoder(weak={"tgt a"})), cfg)
    assert res["rounds"] == 2
    assert {"txt_rf_0@r1", "txt_rf_0@r2", "txt_rf_1@r1", "txt_rf_1@r2"} <= set(res["reliability"])
    out = tempfile.mkdtemp(prefix="de_v2_out_")
    de.export(res, cfg, out, "r2")
    assert set(os.listdir(out)) == {"r2.safetensors"}
    from safetensors import safe_open
    with safe_open(os.path.join(out, "r2.safetensors"), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
    assert int(metadata["ss_network_dim"]) >= 2, "rank is raised to num_directions x rounds"
    assert not any("direction_edit" in key.lower() or "calibration" in key.lower() for key in metadata)


# --- the whole-dataset scan -------------------------------------------------------------------------------------
def test_scan_finds_the_weak_tags():
    d = tempfile.mkdtemp(prefix="de_v2_scan_")
    rng = random.Random(3)
    common = [f"tag{i:02d}" for i in range(30)]
    weak = {"tag03", "tag11", "tag17"}
    for i in range(300):
        open(os.path.join(d, f"c{i:03d}.txt"), "w").write(", ".join(rng.sample(common, 10)))
    cfg = de.make_config({"direction_edit_caption_folder": d, "direction_edit_scan": "1"})
    cfg.update(scan_min_count=30, scan_per_tag=6, scan_targets=3, scan_controls=6, scan_z=0.8)
    cfg = de._normalise(cfg)
    prompts, pairs, meta = de.plan_scan(cfg)
    store = de.encode_prompts(prompts, None, "cpu", encoder=fake_encoder(weak=weak))
    rows = de.scan_measure(tiny_path(), store, pairs)
    sel = de.scan_select(cfg, pairs, rows, meta)
    assert set(sel["targets"]) <= weak and len(sel["targets"]) >= 2, sel["targets"]
    assert len(sel["controls"]) >= 3 and not set(sel["controls"]) & weak


def test_run_direction_edit_end_to_end_with_scan_rounds_and_spill():
    d = make_folder(n=200)
    for i in range(200):                                     # make "tgt a" ordinary: the scan picks the targets
        p = os.path.join(d, f"img{i:03d}.txt")
        text = open(p).read()
        open(p, "w").write(text.replace("tgt a", "tag39"))
    dit = os.path.join(tempfile.mkdtemp(), "fake_dit.safetensors")
    open(dit, "wb").write(b"x")
    real_loader = de.load_fusion_path
    de.load_fusion_path = lambda *a, **k: tiny_path()
    try:
        cfg = de.make_config({"direction_edit_caption_folder": d, "direction_edit_scan": "1",
                              "direction_edit_rounds": "2", "direction_edit_caption_count": "8"})
        cfg.update(scan_min_count=30, scan_per_tag=4, scan_targets=3, scan_controls=8, scan_z=0.3, columns=16,
                   ram_budget_gb=0.00001)
        cfg = de._normalise(cfg)
        out = tempfile.mkdtemp(prefix="de_v2_e2e_")
        before = {x for x in os.listdir(tempfile.gettempdir()) if x.startswith("fizgig_direction_edit_")}
        r = de.run_direction_edit(cfg, dit_path=dit, te_path="unused", output_dir=out, output_name="e2e",
                                  device="cpu", encoder=fake_encoder(weak={"tag03", "tag11", "tag17", "tag22"}))
    finally:
        de.load_fusion_path = real_loader
    names = set(os.listdir(out))
    assert names == {"e2e.safetensors"}
    assert r["verify"]["held_out"] is True
    assert not any("direction_edit" in name.lower() or "calibration" in name.lower() for name in names)
    assert {x for x in os.listdir(tempfile.gettempdir()) if x.startswith("fizgig_direction_edit_")} == before, \
        "the spill folder must be removed"
    assert "verify" in r




def test_manual_target_is_skipped_when_not_suppressed():
    d = make_folder(n=180)
    cfg = de.make_config({"direction_edit_caption_folder": d, "direction_edit_scan": "1",
                          "direction_edit_target_tags": "tag00"})
    cfg.update(scan_min_count=8, scan_per_tag=6, scan_max_tags=500, scan_z=0.8)
    cfg = de._normalise(cfg)
    prompts, pairs, meta = de.plan_scan(cfg)
    path = tiny_path()
    neutral_rows = [{stage: 1.0 for stage in de.STAGES} for _ in pairs]
    rejected = de.scan_select(cfg, pairs, neutral_rows, meta, requested_targets=["tag00"])
    assert rejected["targets"] == []
    assert rejected["skipped_targets"] and rejected["skipped_targets"][0]["tag"] == "tag00"

    store = de.encode_prompts(prompts, None, "cpu", encoder=fake_encoder(weak={"tgt a"}))
    weak_rows = de.scan_measure(path, store, pairs)
    accepted = de.scan_select(cfg, pairs, weak_rows, meta, requested_targets=["tgt a"])
    assert accepted["targets"] == ["tgt a"]
    store.close()


def test_scan_counts_tags_beyond_context_cap():
    d = tempfile.mkdtemp(prefix="de_v2_tail_tags_")
    common = [f"common{i:02d}" for i in range(45)]
    for i in range(24):
        with open(os.path.join(d, f"caption{i:03d}.txt"), "w", encoding="utf-8") as f:
            f.write(", ".join(common + ["late_concept"]))
    cfg = de.make_config({"direction_edit_caption_folder": d, "direction_edit_scan": "1"})
    cfg.update(scan_min_count=8, scan_per_tag=6, scan_max_tags=500, caption_max_tags=40)
    cfg = de._normalise(cfg)
    prompts, pairs, meta = de.plan_scan(cfg)
    assert meta["counts"]["late concept"] == 24
    samples = [p for p in pairs if p["tag"] == "late concept"]
    assert len(samples) == 6
    for sample in samples:
        assert "late_concept" in sample["full"]
        assert "late_concept" not in sample["ablated"]
        assert len(sample["full"].split(", ")) <= 40

def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print("PASS", n)
        except Exception:
            failed += 1
            print("FAIL", n)
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
