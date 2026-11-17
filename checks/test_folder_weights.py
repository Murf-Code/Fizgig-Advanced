"""Per-subfolder training weight and repeats (Start tab table under Include sub-folders)."""
import os

from PIL import Image

from fizgig.dataset.config import ConfigSanitizer
from fizgig.families import form, launch
from fizgig.families.registry import get as get_family
from fizgig.families.train import folder_loss_mult


def _pic(path, caption="a photo"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (32, 32)).save(path)
    if caption is not None:
        with open(os.path.splitext(path)[0] + ".txt", "w", encoding="utf-8") as f:
            f.write(caption)


def _st(d, **kw):
    return {"image_folder": str(d), "LORA_OUTPUT_DIR": str(d / "out"), "megapixels": 0.25,
            "cache_root": str(d.parent / "cache"), "subfolders": True, **kw}


def _tree(tmp_path):
    d = tmp_path / "data"
    _pic(str(d / "a.png"))
    _pic(str(d / "Flaccid" / "a.png"))
    _pic(str(d / "Peen" / "b.png"))
    _pic(str(d / "Pussy and Ass" / "c.png"))
    return d


def test_old_config_toml_is_unchanged(tmp_path):
    d = _tree(tmp_path)
    desc = get_family("krea2")
    plain = launch.dataset_toml(desc, _st(d))
    same = launch.dataset_toml(desc, _st(d, folder_opts={}, folder_balance=False,
                                        FOLDER_OPTS={}, FOLDER_BALANCE=False))
    assert plain == same
    assert "loss_weight" not in plain
    assert plain.count("num_repeats = 1") == 1          # only the [general] default
    assert 'item_prefix = "Flaccid/"' in plain


def test_toml_writes_repeats_and_loss_weight(tmp_path):
    d = _tree(tmp_path)
    desc = get_family("krea2")
    toml = launch.dataset_toml(desc, _st(d, folder_opts={
        "": {"weight": 0.5, "repeats": 2},
        "Flaccid": {"weight": 1, "repeats": 3},
        "Peen": {"weight": 0.5},
    }))
    assert "loss_weight = 0.5" in toml
    assert "num_repeats = 2" in toml
    assert "num_repeats = 3" in toml
    # Peen: weight 0.5, default repeats; Pussy and Ass: unchanged, no extra lines
    # a small parse of each block
    blocks = toml.split("[[datasets]]")[1:]
    by_pref = {}
    for b in blocks:
        rel = ""
        if 'item_prefix = "' in b:
            rel = b.split('item_prefix = "')[1].split('"')[0].rstrip("/")
        by_pref[rel] = b
    assert "loss_weight = 0.5" in by_pref[""] and "num_repeats = 2" in by_pref[""]
    assert "num_repeats = 3" in by_pref["Flaccid"] and "loss_weight" not in by_pref["Flaccid"]
    assert "loss_weight = 0.5" in by_pref["Peen"] and "num_repeats" not in by_pref["Peen"]
    assert "loss_weight" not in by_pref["Pussy and Ass"] and "num_repeats" not in by_pref["Pussy and Ass"]
    # the smaller-size TOML uses the same writer (a cache suffix only)
    staged = launch.dataset_toml(desc, {**_st(d, folder_opts={"": {"weight": 0.5, "repeats": 2}}),
                                        "megapixels": 0.1}, "-0.1mp")
    assert "loss_weight = 0.5" in staged and "num_repeats = 2" in staged


def test_balance_caps_repeats_at_20(tmp_path):
    d = tmp_path / "data"
    _pic(str(d / "big" / "a.png"))
    for i in range(49):
        _pic(str(d / "big" / f"x{i}.png"))
    _pic(str(d / "tiny" / "one.png"))
    sched = launch.folder_schedule(get_family("krea2"), _st(d, folder_balance=True))
    assert sched["big"]["repeats"] == 1 and not sched["big"]["cap_warn"]
    assert sched["tiny"]["count"] == 1
    assert sched["tiny"]["uncapped"] == 50 and sched["tiny"]["repeats"] == 20 and sched["tiny"]["cap_warn"]
    toml = launch.dataset_toml(get_family("krea2"), _st(d, folder_balance=True))
    assert "num_repeats = 20" in toml and "num_repeats = 50" not in toml


def test_weight_zero_is_refused_at_start(tmp_path):
    d = _tree(tmp_path)
    probs = launch.start_problems(get_family("krea2"), _st(d, folder_opts={"Flaccid": {"weight": 0}}))
    assert any("Flaccid" in p and "0 is not allowed" in p for p in probs)
    probs = launch.start_problems(get_family("krea2"), _st(d, folder_opts={"Peen": {"weight": "nope"}}))
    assert any("Peen" in p and "must be a number above 0" in p for p in probs)


def test_folder_loss_mult_scales_backward_only():
    assert folder_loss_mult(["Flaccid/a.png"], {"Flaccid/a.png": 0.5}) == 0.5
    assert folder_loss_mult(["x.png"], {}) == 1.0
    assert folder_loss_mult(None, {"a": 0.5}) == 1.0
    assert folder_loss_mult(["a"], {"b": 0.5}) == 1.0


def test_schema_accepts_loss_weight():
    cfg = ConfigSanitizer().sanitize_user_config({
        "general": {"resolution": [32, 32], "batch_size": 1, "num_repeats": 1},
        "datasets": [{"image_directory": "/tmp", "loss_weight": 0.5, "num_repeats": 2}],
    })
    assert cfg["datasets"][0]["loss_weight"] == 0.5
    assert cfg["datasets"][0]["num_repeats"] == 2


def test_table_hidden_when_subfolders_are_off(tmp_path):
    d = _tree(tmp_path)
    desc = get_family("krea2")
    st = form.state(desc, {"TRAIN_SUBFOLDERS": False}, str(d))
    assert "FOLDER_OPTS" in st["hidden"] and "FOLDER_BALANCE" in st["hidden"]
    st = form.state(desc, {"TRAIN_SUBFOLDERS": True}, str(d))
    assert "FOLDER_OPTS" not in st["hidden"]
    rels = [r["rel"] for r in st["folders"]]
    assert rels == ["", "Flaccid", "Peen", "Pussy and Ass"]
    assert st["folder_items"] == 4
