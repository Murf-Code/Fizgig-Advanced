"""RMBG-3.0 masks (dataset/rmbgmask.py): the edge settings, the crop the pipeline makes and its way back, the folder
run with a stand-in model, the precision plan, the CLI, and the Fibo 2.0 loader reading 1038lab's single files - no
model, no GPU."""
import json
import math
import os

import numpy as np
import pytest
import torch
from PIL import Image

from fizgig.dataset import masks, rmbgmask as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def disk(h, w, cx, cy, r, soft=4.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip((r - d) / soft + 0.5, 0.0, 1.0).astype(np.float32)


# ---- edges -------------------------------------------------------------------------------------------------------
def test_soft_matte_is_kept_as_it_is():
    a = disk(64, 64, 32, 32, 16)
    assert np.allclose(R.finish(a), a)
    assert 0 < ((a > 0) & (a < 1)).sum()                     # the greys survive: masked training reads them as weight


def test_threshold_makes_it_black_and_white():
    a = disk(64, 64, 32, 32, 16)
    m = R.finish(a, threshold=0.5)
    assert set(np.unique(m)) <= {0.0, 1.0}
    assert (R.finish(a, threshold=0.1).sum() > m.sum() > R.finish(a, threshold=0.9).sum())


def test_grow_widens_and_feather_only_adds_outside():
    a = R.finish(disk(96, 96, 48, 48, 20), threshold=0.5)
    g = R.finish(a, grow=4)
    assert g.sum() > a.sum() and np.all(g >= a)
    r_a = math.sqrt(a.sum() / math.pi)
    r_g = math.sqrt(g.sum() / math.pi)
    assert 3 <= r_g - r_a <= 5                                # about 4 px further out all round
    f = R.finish(a, feather=3)
    assert np.all(f >= a - 1e-6) and f.sum() > a.sum()       # never eats into the white
    assert np.all(f[a == 1] == 1)


def test_background_is_the_opposite_after_grow():
    a = disk(64, 64, 32, 32, 12)
    s = R.finish(a, threshold=0.5, grow=2)
    b = R.finish(a, kind="background", threshold=0.5, grow=2)
    assert np.allclose(b, 1 - s)


def test_settings_are_checked():
    assert R.settings()["threshold"] is None
    assert R.settings(threshold="soft")["threshold"] is None
    assert R.settings(threshold="0.4", grow="3")["grow"] == 3
    for bad in ({"threshold": 1.5}, {"grow": 100}, {"kind": "hair"}, {"feather": -1}):
        with pytest.raises(ValueError):
            R.settings(**bad)


# ---- the crop the pipeline makes ---------------------------------------------------------------------------------
def _reference_size(aspect, area=1024 * 1024, multiple=16):
    """diffusers' BriaFibo2ImageProcessor.get_image_size, verbatim in effect."""
    lo = 4 * multiple
    sizes = [(w, h) for w in range(lo, 2 * area // lo, multiple) for h in range(lo, 2 * area // w, multiple)
             if 0.9 * area <= w * h <= 1.1 * area]
    crop = lambda s: 1 - min((s[0] / s[1]) / aspect, aspect / (s[0] / s[1]))    # noqa: E731
    return min(sizes, key=lambda s: (round(crop(s), 9), abs(s[0] * s[1] - area), -s[0]))


@pytest.mark.parametrize("w,h", [(1024, 1024), (1920, 1080), (800, 1200), (3000, 1000), (517, 1301)])
def test_work_size_is_the_pipelines(w, h):
    assert R.work_size(w, h) == _reference_size(w / h)


def test_work_size_at_half_a_megapixel():
    w, h = R.work_size(1000, 1000, megapixels=0.5)
    assert w % 16 == 0 and h % 16 == 0 and 0.45e6 < w * h * 2 < 1.2e6


def test_matte_of_the_crop_lands_back_on_the_picture():
    W, H = 900, 600                                           # 3:2 - the 1 MP size crops a little off the sides
    full = disk(H, W, 450, 300, 180)
    img = Image.fromarray((np.stack([full] * 3, -1) * 255).astype(np.uint8))
    w, h = R.work_size(W, H)
    crop, geom = R.cover_crop(img, w, h)
    assert crop.size == (w, h)
    m = np.asarray(crop.convert("L"), np.float32) / 255.0     # a perfect model: the matte is the picture
    back = R.uncrop(m, geom, (W, H))
    assert back.shape == (H, W)
    assert np.abs(back - full).mean() < 0.01
    assert np.abs(back - full)[50:-50, 50:-50].max() < 0.2


# ---- the folder run ----------------------------------------------------------------------------------------------
class FakeMatter:
    """Stands in for RMBG-3.0: a soft disk in the middle of every picture."""

    def __init__(self):
        self.seen = []

    def mattes(self, pictures):
        for p in pictures:
            self.seen.append(os.path.basename(p))
            if p.endswith("broken.png"):
                yield p, None, "can't be read (test)"
                continue
            with Image.open(p) as im:
                w, h = im.size
            if "empty" in p:
                yield p, np.zeros((h, w), np.float32), ""
            else:
                yield p, disk(h, w, w / 2, h / 2, min(w, h) / 3), ""


def _folder(tmp_path, names=("a.png", "b.jpg")):
    for n in names:
        Image.new("RGB", (120, 80), (200, 100, 50)).save(tmp_path / n)
    return str(tmp_path)


def test_make_folder_writes_masklabels_and_keeps_the_mattes(tmp_path):
    folder = _folder(tmp_path, ("a.png", "b.jpg", "empty.png"))
    c = R.make_folder(folder, "", matter=FakeMatter(), threshold=0.5, grow=2, log=lambda s: None)
    assert c == {"made": 2, "kept": 0, "skipped": 0, "no_person": 1}
    for stem in ("a", "b"):
        p = os.path.join(folder, stem + "-masklabel.png")
        assert os.path.isfile(p) and os.path.isfile(os.path.join(folder, R.MATTES_DIR, stem + ".png"))
        assert masks.mask_problem(p, (120, 80)) == ""          # training will use it
        assert set(np.unique(np.asarray(Image.open(p)))) <= {0, 255}
    assert not os.path.exists(os.path.join(folder, "empty-masklabel.png"))
    assert masks.mask_for(os.path.join(folder, "a.png")).endswith("a-masklabel.png")


def test_masks_already_there_are_kept_unless_replaced(tmp_path):
    folder = _folder(tmp_path)
    Image.new("L", (120, 80), 255).save(os.path.join(folder, "a-masklabel.png"))
    fake = FakeMatter()
    c = R.make_folder(folder, "", matter=fake, log=lambda s: None)
    assert c["kept"] == 1 and c["made"] == 1 and fake.seen == ["b.jpg"]
    assert np.asarray(Image.open(os.path.join(folder, "a-masklabel.png"))).min() == 255   # untouched
    c = R.make_folder(folder, "", matter=FakeMatter(), overwrite=True, log=lambda s: None)
    assert c["made"] == 2
    assert np.asarray(Image.open(os.path.join(folder, "a-masklabel.png"))).min() == 0


def test_unreadable_picture_is_skipped(tmp_path):
    folder = _folder(tmp_path, ("a.png", "broken.png"))
    c = R.make_folder(folder, "", matter=FakeMatter(), log=lambda s: None)
    assert c["made"] == 1 and c["skipped"] == 1


def test_refinish_remakes_from_the_saved_mattes_without_a_model(tmp_path):
    folder = _folder(tmp_path)
    R.make_folder(folder, "", matter=FakeMatter(), log=lambda s: None)          # soft
    soft = np.asarray(Image.open(os.path.join(folder, "a-masklabel.png")))
    assert len(np.unique(soft)) > 2
    c = R.refinish_folder(folder, threshold=0.5, grow=3, log=lambda s: None)
    assert c["made"] == 2
    hard = np.asarray(Image.open(os.path.join(folder, "a-masklabel.png")))
    assert set(np.unique(hard)) <= {0, 255} and (hard > 0).sum() > (soft >= 128).sum()


def test_stop_ends_the_run_after_the_current_picture(tmp_path):
    folder = _folder(tmp_path, ("a.png", "b.png", "c.png"))
    masks.STOP.set()
    try:
        c = R.make_folder(folder, "", matter=FakeMatter(), log=lambda s: None)
    finally:
        masks.STOP.clear()
    assert c["made"] == 1 and c["stopped"] == (1, 3)


def test_without_a_gpu_it_says_so_or_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "cuda_here", lambda: False)
    assert "NVIDIA" in R.why_not(str(tmp_path))
    assert "NVFP4" in R.why_not(str(tmp_path), "nvfp4")
    with pytest.raises(ValueError):
        R.matter_for(str(tmp_path))
    with pytest.raises(ValueError, match="character finder"):    # the fallback needs anime-seg downloaded
        R.matter_for(str(tmp_path), fallback=True, log=lambda s: None)


def test_files_are_found_in_the_models_folder(tmp_path):
    d = str(tmp_path)
    assert [w for w, _ in R.missing(d)] == ["RMBG-3.0 fp8", "Fibo 2.0 text encoder", "Fibo 2.0 autoencoder"]
    os.makedirs(R.model_dir(d))
    (tmp_path / R.FOLDER / "RMBG-3.0-fp8.safetensors").write_bytes(b"x")
    (tmp_path / "fibo2_text_encoder.safetensors").write_bytes(b"x")   # shared with the Fibo 2.0 family
    (tmp_path / "flux2-vae.safetensors").write_bytes(b"x")
    assert R.missing(d) == [] and R.find_model(d).endswith("RMBG-3.0-fp8.safetensors")
    info = R.info(d)
    assert info["variants"]["fp8"]["missing"] == [] and info["variants"]["bf16"]["missing"]
    json.dumps(info)                                            # the masks card reads it as JSON


# ---- the precision plan -------------------------------------------------------------------------------------------
def test_plan_fits_the_card():
    assert R.plan(40.0) == ("bf16", 0)
    assert R.plan(16.0) == ("int8", 0)
    p, swap = R.plan(11.0)                                      # a 12 GB card, the desktop taking a little
    assert p == "int8" and 0 < swap < 15
    assert R.plan(11.0, "nf4") == ("nf4", 0)
    assert R.plan(11.0, "int8", 10) == ("int8", 10)
    with pytest.raises(ValueError):
        R.plan(11.0, "fp4")


# ---- the model glue, with a stand-in driver ------------------------------------------------------------------------
class FakeDriver:
    def __init__(self):
        self.calls = []

    def load_text_encoder(self, path, device):
        self.calls.append("te")
        return object()

    def unload_text_encoder(self, te):
        self.calls.append("te off")

    def encode_text_with_references(self, te, captions, references):
        self.calls.append(("encode", captions[0], references[0][0].size))
        return [{"tokens": torch.zeros(1, 6, 3, 8), "mask": torch.ones(1, 3, dtype=torch.bool)}]

    def load_vae(self, path, device):
        self.calls.append("vae")
        return object()

    def encode_images(self, vae, images):
        a = images[0]
        return [torch.from_numpy(a[..., :1].astype(np.float32).transpose(2, 0, 1) / 255.0)]

    def load_planned(self, path, device, precision, swap):
        self.calls.append(("dit", precision, swap))
        return "dit", 0

    def generate(self, dit, cond, w, h, *, steps, seed, cfg, refs):
        self.calls.append(("gen", steps, cfg, tuple(refs[0].shape)))
        return refs[0]                                          # "the matte": the picture's own brightness

    def decode(self, vae, x, w, h):
        a = (x[0, 0].numpy() * 255).astype(np.uint8)
        return Image.fromarray(np.stack([a] * 3, -1))


def test_matter_runs_te_then_dit_with_the_fixed_instruction(tmp_path):
    W, H = 640, 400
    full = disk(H, W, 320, 200, 120)
    p = str(tmp_path / "fox.png")
    Image.fromarray((np.stack([full] * 3, -1) * 255).astype(np.uint8)).save(p)
    drv = FakeDriver()
    m = R.Matter("dit", "te", "vae", device="cpu", driver=drv, log=lambda s: None)
    out = list(m.mattes([p]))
    assert len(out) == 1 and out[0][1].shape == (H, W)
    assert np.abs(out[0][1] - full).mean() < 0.02               # the matte comes back over the picture, aligned
    enc = [c for c in drv.calls if isinstance(c, tuple) and c[0] == "encode"][0]
    assert json.loads(enc[1]) == {"edit_instruction": R.INSTRUCTION}
    assert enc[2] == R.work_size(W, H)
    gen = [c for c in drv.calls if isinstance(c, tuple) and c[0] == "gen"][0]
    assert gen[1] == 4 and gen[2] == 1.0
    order = [c if isinstance(c, str) else c[0] for c in drv.calls]
    assert order.index("te off") < order.index("dit")          # the encoder leaves before the DiT loads


def test_matter_goes_a_chunk_at_a_time(tmp_path):
    paths = []
    for i in range(5):
        p = str(tmp_path / f"p{i}.png")
        Image.new("RGB", (64, 64), (255, 255, 255)).save(p)
        paths.append(p)
    drv = FakeDriver()
    out = list(R.Matter("dit", "te", "vae", device="cpu", driver=drv, chunk=2, log=lambda s: None).mattes(paths))
    assert len(out) == 5
    assert sum(1 for c in drv.calls if c == "te") == 3          # 2 + 2 + 1
    assert sum(1 for c in drv.calls if c == "vae") == 1         # the small autoencoder stays


# ---- the CLI -------------------------------------------------------------------------------------------------------
def test_cli_arguments():
    from fizgig.scripts import rmbg3_masks
    a = rmbg3_masks.parser().parse_args(["D:/x", "--threshold", "0.4", "--grow", "3", "--variant", "bf16"])
    assert (a.folder, a.threshold, a.grow, a.variant, a.precision, a.blocks_to_swap) == \
        ("D:/x", 0.4, 3, "bf16", "auto", -1)
    with pytest.raises(SystemExit):
        rmbg3_masks.parser().parse_args(["D:/x", "--kind", "hair"])


def test_cli_refinish(tmp_path):
    from fizgig.scripts import rmbg3_masks
    folder = _folder(tmp_path)
    R.make_folder(folder, "", matter=FakeMatter(), log=lambda s: None)
    assert rmbg3_masks.main([folder, "--refinish", "--threshold", "0.5", "--models-dir", str(tmp_path)]) == 0
    assert set(np.unique(np.asarray(Image.open(os.path.join(folder, "a-masklabel.png"))))) <= {0, 255}
    assert rmbg3_masks.main([str(tmp_path / "nope"), "--models-dir", str(tmp_path)]) == 2


# ---- 1038lab's single files through the Fibo 2.0 loader ----------------------------------------------------------
def test_unpadded_ffn_splits_in_half():
    from fizgig.fibo2.driver import Fibo2Driver, HIDDEN
    d = Fibo2Driver()
    for rows in (2 * 9728, 2 * HIDDEN):                       # ComfyUI's padded build, and 1038lab's unpadded one
        t = torch.arange(rows, dtype=torch.float32)[:, None].expand(rows, 2).contiguous()
        parts = dict(d._unfuse("layers.0.feed_forward.w13.weight", t))
        w1, w3 = parts["layers.0.feed_forward.w1.weight"], parts["layers.0.feed_forward.w3.weight"]
        assert w1.shape == w3.shape == (HIDDEN, 2)
        assert w1[0, 0] == 0 and w3[0, 0] == rows // 2           # the second half starts where it should
    w2 = dict(d._unfuse("layers.0.feed_forward.w2.weight", torch.zeros(4, HIDDEN)))
    assert w2["layers.0.feed_forward.w2.weight"].shape == (4, HIDDEN)


def test_fp8_single_file_is_decoded(tmp_path):
    from safetensors.torch import save_file
    from fizgig.fibo2.driver import Fibo2Driver
    w = torch.tensor([[0.5, -1.0, 2.0], [0.25, 0.0, -0.5]])
    scale = torch.tensor(0.5)
    q = (w / scale).to(torch.float8_e4m3fn)
    blob = torch.tensor(list(b'{"format": "float8_e4m3fn"}'), dtype=torch.uint8)
    path = str(tmp_path / "rmbg.safetensors")
    save_file({"final_layer.linear.weight": q, "final_layer.linear.weight_scale": scale,
               "final_layer.linear.comfy_quant": blob,
               "final_layer.linear.bias": torch.zeros(2, dtype=torch.bfloat16)}, path)

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.final_layer = torch.nn.Module()
            self.final_layer.linear = torch.nn.Linear(3, 2)
    toy = Toy()
    Fibo2Driver()._load_fused(path, toy, "cpu")
    got = toy.final_layer.linear.weight
    assert got.dtype == torch.bfloat16 and torch.allclose(got.float(), w, atol=1e-2)


# ---- the masks card -------------------------------------------------------------------------------------------------
def test_masks_card_offers_rmbg():
    js = open(os.path.join(ROOT, "src", "fizgig", "web", "static", "app.js"), encoding="utf-8").read()
    html = open(os.path.join(ROOT, "src", "fizgig", "web", "static", "index.html"), encoding="utf-8").read()
    server = open(os.path.join(ROOT, "src", "fizgig", "web", "server.py"), encoding="utf-8").read()
    assert 'data-sam="2"' in html and 'id="maskRmbgVariant"' in html and 'id="maskRmbgRefinish"' in html
    assert "/api/masks/rmbg_fetch" in js and "/api/masks/rmbg_preview" in js and "maskRmbgRunOpts()" in js
    for route in ("/api/masks/rmbg_fetch", "/api/masks/rmbg_preview", '"rmbg": _rmbg_info()'):
        assert route in server


def test_prep_job_remakes_masks_and_refuses_without_the_model(tmp_path, monkeypatch):
    import time
    from fizgig.web import prep_jobs
    folder = _folder(tmp_path)
    R.make_folder(folder, "", matter=FakeMatter(), log=lambda s: None)
    monkeypatch.setattr(R, "cuda_here", lambda: False)
    with pytest.raises(ValueError, match="NVIDIA"):
        prep_jobs.start_masks(folder, "", rmbg={"variant": "fp8"})
    with pytest.raises(ValueError):
        prep_jobs.start_masks(folder, "", rmbg={"variant": "fp8", "refinish": True, "grow": 999})
    assert prep_jobs.start_masks(folder, "", rmbg={"refinish": True, "threshold": 0.5})["ok"]
    for _ in range(200):
        if prep_jobs._job["state"] != "running":
            break
        time.sleep(0.05)
    assert prep_jobs._job["state"] == "done", prep_jobs._job["log"]
    assert "RMBG-3.0" in prep_jobs._job["log"] and "2 mask(s) written" in prep_jobs._job["log"]
    assert set(np.unique(np.asarray(Image.open(os.path.join(folder, "a-masklabel.png"))))) <= {0, 255}


def test_preview_shows_two_pictures_and_writes_nothing(tmp_path, monkeypatch):
    folder = _folder(tmp_path, ("a.png", "b.png", "c.png"))
    shots, fresh = R.preview(folder, "", matter=FakeMatter(), megapixels=0.5, threshold=0.5, grow=2)
    assert len(shots) == 2 and all(s[3] == "character" and 0 < s[2] < 1 for s in shots)
    assert not any(n.endswith("-masklabel.png") for n in os.listdir(folder))
    monkeypatch.setattr(R, "cuda_here", lambda: False)
    with pytest.raises(ValueError, match="NVIDIA"):
        R.preview(folder, str(tmp_path), megapixels=0.5)
