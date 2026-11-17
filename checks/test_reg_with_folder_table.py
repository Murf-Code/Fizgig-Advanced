"""The upstream regularisation dataset (Subjects card, --reg_per_epoch) beside the Start tab's per-folder
Weight / Repeats / Balance table: the reg set is its own dataset block, never in the table, never given a folder
weight or repeats; the per-epoch share counts the table's repeats; and Krea 2's Global replay / Text fusion only
choices leave both alone."""
import os

from PIL import Image

from fizgig.families import form, launch
from fizgig.families.registry import get as get_family

DESC = get_family("krea2")


def _pic(path, caption="a photo"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (32, 32)).save(path)
    with open(os.path.splitext(path)[0] + ".txt", "w", encoding="utf-8") as f:
        f.write(caption)


def _tree(tmp_path, reg_count=50):
    train, reg = tmp_path / "train", tmp_path / "reg"
    for i in range(10):
        _pic(str(train / f"t{i}.png"), "zwxem, a woman")
    for i in range(4):
        _pic(str(train / "close" / f"c{i}.png"), "zwxem, a face")
    for i in range(reg_count):
        _pic(str(reg / f"r{i:02d}.png"), "a woman in a park")
    return train, reg


def _inputs(tmp_path, train, reg, **kw):
    values = {**form.DEFAULTS, "LORA_NAME": "mine", "FAMILY_REG_ON": True, "FAMILY_REG_DIR": str(reg),
              "TRAIN_SUBFOLDERS": True, "CAPTION_TRIGGER_WORD": "zwxem", **kw}
    st = form.inputs(DESC, values, models={}, python="python", repo_dir=".", image_folder=str(train),
                     cache_root=str(tmp_path / "cache"), samples_dir=str(tmp_path / "out" / "sample"))
    st.update(DATASET_CONFIG=str(tmp_path / "ds.toml"), LORA_OUTPUT_DIR=str(tmp_path / "out"))
    return st


class _Plan:
    files, console, dirs = [], [], []


def _blocks(toml):
    out = {}
    for b in toml.split("[[datasets]]")[1:]:
        pref = b.split('item_prefix = "')[1].split('"')[0] if 'item_prefix = "' in b else ""
        out[pref] = b
    return out


def test_reg_block_is_not_a_table_row_and_gets_no_weight_or_repeats(tmp_path):
    train, reg = _tree(tmp_path)
    st = _inputs(tmp_path, train, reg, FOLDER_OPTS={"": {"weight": 0.5, "repeats": 2},
                                                    "close": {"weight": 2, "repeats": 3}})
    assert [r for r in launch.folder_schedule(DESC, st)] == ["", "close"]     # no reg row in the table
    blocks = _blocks(launch.dataset_toml(DESC, st))
    assert set(blocks) == {"", "close/", "reg:"} or set(blocks) == {"", "close", "reg:"}
    reg_block = blocks["reg:"]
    assert "is_reg = true" in reg_block
    assert "loss_weight" not in reg_block and "num_repeats" not in reg_block
    close = next(b for k, b in blocks.items() if k.startswith("close"))
    assert "loss_weight = 2" in close and "num_repeats = 3" in close


def test_per_epoch_share_counts_the_tables_repeats(tmp_path):
    train, reg = _tree(tmp_path)
    plain = _inputs(tmp_path, train, reg)
    per, have, items = launch.lora_reg_budget(DESC, plain)
    assert (have, items) == (50, 14) and per == 14          # as many as the training pictures
    rep = _inputs(tmp_path, train, reg, FOLDER_OPTS={"close": {"repeats": 3}})
    per, have, items = launch.lora_reg_budget(DESC, rep)
    assert (have, items) == (50, 14 + 4 * 2) and per == 22  # the close folder trains 3 times an epoch
    cmd = launch.train_command(DESC, rep, _Plan())
    assert cmd[cmd.index("--reg_per_epoch") + 1] == "22"
    # with the table's Include sub-folders off nothing is repeated, and nothing changes
    off = _inputs(tmp_path, train, reg, TRAIN_SUBFOLDERS=False, FOLDER_OPTS={"close": {"repeats": 3}})
    assert launch.lora_reg_budget(DESC, off)[2] == 10


def test_balance_folders_counts_too(tmp_path):
    train, reg = _tree(tmp_path)
    st = _inputs(tmp_path, train, reg, FOLDER_BALANCE=True)         # close (4) is brought up to the 10 of the top folder
    assert launch.folder_schedule(DESC, st)["close"]["repeats"] == 2    # round(10 / 4)
    assert launch.lora_reg_budget(DESC, st)[2] == 14 + 4 * 1


def test_the_table_stays_with_the_reg_fold_and_goes_with_subject_modes(tmp_path):
    train, reg = _tree(tmp_path)
    vals = {"TRAIN_SUBFOLDERS": True, "FAMILY_REG_ON": True, "FAMILY_REG_DIR": str(reg)}
    st = form.state(DESC, vals, str(train))
    assert "FOLDER_OPTS" not in st["hidden"] and "TRAIN_SUBFOLDERS" not in st["hidden"]
    assert [r["rel"] for r in st["folders"]] == ["", "close"]
    st = form.state(DESC, {"TRAIN_SUBFOLDERS": True, "FAMILY_DOP": True}, str(train))
    assert {"TRAIN_SUBFOLDERS", "FOLDER_OPTS", "FOLDER_BALANCE"} <= set(st["hidden"])


def test_krea2_methods_leave_the_table_and_reg_alone(tmp_path):
    train, reg = _tree(tmp_path)
    for method in ("Standard", "Global replay", "Text fusion only"):
        st = _inputs(tmp_path, train, reg, KREA2_TRAINING_METHOD=method,
                     FOLDER_OPTS={"close": {"repeats": 2, "weight": 0.5}})
        toml = launch.dataset_toml(DESC, st)
        assert "loss_weight = 0.5" in toml and "is_reg = true" in toml


def test_training_maps_with_global_replay_is_refused_at_start(tmp_path):
    train, reg = _tree(tmp_path)
    for method, refused in (("Global replay", True), ("Standard", False), ("Text fusion only", False)):
        st = _inputs(tmp_path, train, reg, TRAINING_MAPS=True, KREA2_TRAINING_METHOD=method)
        assert st["TRAINING_MAPS"] is True
        probs = [p for p in launch.krea2_method_problems(DESC, st) if "Training maps" in p]
        assert bool(probs) == refused, (method, probs)
    st = _inputs(tmp_path, train, reg, TRAINING_MAPS=False, KREA2_TRAINING_METHOD="Global replay")
    assert launch.krea2_method_problems(DESC, st) == []
    assert launch.krea2_method_problems(get_family("anima"), {"TRAINING_MAPS": True}) == []


def test_start_lists_the_training_maps_refusal(tmp_path):
    train, reg = _tree(tmp_path)
    st = _inputs(tmp_path, train, reg, TRAINING_MAPS=True, KREA2_TRAINING_METHOD="Global replay")
    assert st["FAMILY_OPTIONS"]["KREA2_TRAINING_METHOD"] == "Global replay"
    assert any("Training maps can't be used" in p for p in launch.start_problems(DESC, st))
