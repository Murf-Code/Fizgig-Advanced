"""Validation groups (Image Validation, 1 to 3): the pictures are dealt into separate sets, each its own line on the loss
graph. One group is the check as it always was; the average (early stop, Adaptive LR) is unchanged; the log, the
snapshot, the hold-out sizing and the form / command line carry the setting."""
import os

import pytest
import torch
from PIL import Image

from fizgig.families import form, launch, train as T
from fizgig.families.registry import get as get_family
from fizgig.training import holdout
from fizgig.training.step_log import StepLog, loss_path, read
from fizgig.training.val_loss import FixedValidation, auto_pick_keys, deal_groups
from fizgig.training.val_pick import clamp_groups

ROOT = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig")
KEYS = [f"img{i:02d}" for i in range(24)]
DESC = get_family("krea2")


class _Driver:
    """A check's loss from the picture, the noise level and the picture's own seed: nothing else."""
    def batch_cond(self, batch, device):
        return None

    def training_loss(self, dit, latents, cond, g, min_t=0, max_t=1, **kw):
        return float(latents.mean()) * 0.01 + min_t * 0.1 + (g.initial_seed() % 1000) * 1e-5, {}


def _fill(v):
    for k in v.wanted:
        v.offer({"item_keys": [k], "latents": torch.full((1, 2), float(KEYS.index(k)))})


def _run(v):
    out = v.run(_Driver(), None, "cpu")
    return out


# ---- the split

def test_deal_is_round_robin_in_shape_order_and_deterministic():
    pool = [f"k{i}" for i in range(9)]
    shapes = {k: (512, 512 + 64 * (i % 3)) for i, k in enumerate(pool)}            # three shapes, interleaved
    a = deal_groups(pool, 3, shapes)
    assert a == deal_groups(pool, 3, shapes)
    assert sorted(sum(a, [])) == sorted(pool) and [len(g) for g in a] == [3, 3, 3]
    # every group holds one picture of each shape
    for g in a:
        assert sorted({shapes[k] for k in g}) == sorted(set(shapes.values()))
    # no shapes known: plain round-robin of the picked order
    assert deal_groups(pool, 2) == [pool[0::2], pool[1::2]]


def test_one_group_is_the_picked_list_and_groups_never_outnumber_pictures():
    pool = ["a", "b", "c", "d"]
    assert deal_groups(pool, 1) == [pool]
    assert deal_groups(["a", "b"], 3) == [["a"], ["b"]]
    assert [clamp_groups(x) for x in (None, "", "x", 0, "2", 3, 9, 2.0)] == [1, 1, 1, 1, 2, 3, 3, 2]


def test_automatic_pick_is_groups_times_images_and_stays_in_the_main_folder():
    keys = [f"p{i}" for i in range(30)] + [f"crops/c{i}" for i in range(30)]
    v = FixedValidation(keys, images=4, groups=3)
    assert len(v.wanted) == 12 and all("/" not in k for k in v.wanted)
    assert v.wanted == auto_pick_keys(keys, 12)
    assert [len(g) for g in v.groups] == [4, 4, 4] and sorted(sum(v.groups, [])) == sorted(v.wanted)


def test_hand_picks_are_dealt_between_the_groups():
    v = FixedValidation(KEYS, images=4, picked=["img03.png", "img07.png", "img11.png", "img15.png"], groups=2)
    assert len(v.wanted) == 4 and [len(g) for g in v.groups] == [2, 2]
    few = FixedValidation(KEYS, images=4, picked=["img03.png", "img07.png"], groups=3)
    assert [len(g) for g in few.groups] == [1, 1]            # fewer pictures than groups: fewer groups


# ---- one group == the old numbers

def test_one_group_is_the_old_check_and_groups_keep_each_pictures_own_noise():
    old = FixedValidation(KEYS, images=6, seed=11)                   # groups defaulted: the check as it was
    assert old.groups == [old.wanted] and old.last_groups == []
    _fill(old)
    mean1, per_t1, _ = _run(old)
    assert old.last_groups == []                                     # nothing extra with one group
    two = FixedValidation(KEYS, images=3, seed=11, groups=2)         # the same six pictures
    assert two.wanted == old.wanted
    _fill(two)
    mean2, per_t2, _ = _run(two)
    assert mean2 == pytest.approx(mean1) and per_t2 == pytest.approx(per_t1)     # the average is unchanged
    assert len(two.last_groups) == 2
    # each group's mean is its own pictures' mean, from the same per-picture noise
    one_each = {}
    for k in old.wanted:
        solo = FixedValidation(KEYS, images=6, seed=11)
        solo.wanted = [k]
        solo.groups = [[k]]
        solo.batches = {k: old.batches[k]}
        # keep the picture's place in the picked list: its seed index is that place
        j = old.wanted.index(k)
        pre = [k2 for k2 in old.wanted][:j]
        solo.wanted = pre + [k]
        solo.batches = {k: old.batches[k]}
        one_each[k] = _run(solo)[0]
    for g, got in zip(two.groups, two.last_groups):
        assert got == pytest.approx(sum(one_each[k] for k in g) / len(g))


def test_a_check_still_returns_its_three_things():
    v = FixedValidation(KEYS, images=2, groups=3)
    _fill(v)
    out = _run(v)
    assert isinstance(out, tuple) and len(out) == 3 and len(v.last_groups) == 3


# ---- the snapshot

def test_resume_with_other_groups_starts_a_new_section_and_old_snapshots_still_load(tmp_path):
    p = str(tmp_path / "snap.pt")
    v = FixedValidation(KEYS, images=3, seed=7, groups=2)
    v.snapshot_path = p
    _fill(v)
    same = FixedValidation(KEYS, images=3, seed=7, groups=2)
    assert same.restore(p) == "kept" and same.groups == v.groups
    other = FixedValidation(KEYS, images=6, seed=7, groups=1)          # the same six pictures, as one set
    assert other.wanted == v.wanted
    assert other.restore(p) == "changed" and other.seg == 1
    # a snapshot written before groups existed (no "groups") is the one group of its pictures
    old = FixedValidation(KEYS, images=3, seed=7)
    old.snapshot_path = str(tmp_path / "old.pt")
    _fill(old)
    snap = torch.load(old.snapshot_path, map_location="cpu", weights_only=True)
    snap.pop("groups")
    torch.save(snap, old.snapshot_path)
    assert FixedValidation(KEYS, images=3, seed=7).restore(old.snapshot_path) == "kept"
    assert FixedValidation(KEYS, images=2, seed=7, groups=2).restore(old.snapshot_path) == "changed"


# ---- the step log

def test_step_log_round_trip(tmp_path):
    lg = StepLog(str(tmp_path), "x")
    lg.add(1, 1, 0.1)
    lg.add_validation(0, 0, 0.5, [0.4, 0.6], 1.0, groups=[0.45, None, 0.55])
    lg.add_validation(2, 1, 0.4, [0.3, 0.5], 1.0, groups=[0.35, 0.4, 0.45], seg=1)
    lg.add_validation(3, 2, 0.3, [0.3, 0.3], 1.0)                      # one group, or an older run: no "vg"
    lg.flush()
    path = loss_path(str(tmp_path), "x")
    lines = open(path).read().splitlines()
    assert '"vg": [0.45, null, 0.55]' in lines[1] and '"vg"' not in lines[3]
    d = read(path)
    assert d["val"] == [[0, 0, 0.5, [0.4, 0.6]], [2, 1, 0.4, [0.3, 0.5], 1], [3, 2, 0.3, [0.3, 0.3]]]   # unchanged shape
    assert d["val_groups"] == [[0, 0, [0.45, None, 0.55], 0], [2, 1, [0.35, 0.4, 0.45], 1]]


def test_runs_without_groups_read_as_before(tmp_path):
    lg = StepLog(str(tmp_path), "y")
    lg.add(1, 1, 0.1)
    lg.add_validation(1, 1, 0.2, [0.2], 1.0)
    lg.flush()
    d = read(loss_path(str(tmp_path), "y"))
    assert d["val"] == [[1, 1, 0.2, [0.2]]] and d["val_groups"] == []


def test_the_log_line_names_each_group():
    class V:
        last_groups = [0.1234, None, 0.2]
    assert T._groups_text(V()) == "groups [g1 0.1234, g2 -, g3 0.2000]  "

    class One:
        last_groups = []
    assert T._groups_text(One()) == "" and T._groups_text(object()) == ""


# ---- the hold-out

def _cands(n):
    return [("", [f"/d/p{i:03d}.png" for i in range(n)])]


def test_holdout_covers_groups_times_images_up_to_a_fifth():
    paths, _names, how, why = holdout.choose(_cands(100), 4)
    assert how == "auto" and len(paths) == 4 and why == ""            # as ever
    paths, _names, how, why = holdout.choose(_cands(100), 4, groups=3)
    assert len(paths) == 12 and why == ""                              # 3 groups of 4: room for all of them
    paths, _names, how, why = holdout.choose(_cands(100), 4, [], 1)
    assert len(paths) == 4
    # too few pictures for full groups: capped at about a fifth, and the note says so
    paths, _names, how, why = holdout.choose(_cands(40), 4, groups=3)
    assert len(paths) == 8 and "3 validation groups would need 12" in why and "8 are held out" in why
    paths, _names, how, why = holdout.choose(_cands(10), 4, groups=3)
    assert len(paths) == 2 and "in 2 group(s)" in why
    assert holdout.choose(_cands(100), 4, groups=3) == holdout.choose(_cands(100), 4, groups=3)   # deterministic


def test_holdout_picks_you_made_are_all_kept_with_groups():
    paths, names, how, why = holdout.choose(_cands(60), 4, ["p001.png", "p005.png", "p009.png"], groups=2)
    assert how == "picked" and len(paths) == 3


# ---- the form, the command line and the trainer

def _inputs(tmp_path, **kw):
    d = tmp_path / "train"
    for i in range(30):
        os.makedirs(d, exist_ok=True)
        Image.new("RGB", (32, 32)).save(str(d / f"t{i}.png"))
        (d / f"t{i}.txt").write_text("a photo")
    values = {**form.DEFAULTS, "LORA_NAME": "mine", **kw}
    st = form.inputs(DESC, values, models={}, python="python", repo_dir=".", image_folder=str(d),
                     cache_root=str(tmp_path / "cache"), samples_dir=str(tmp_path / "out" / "sample"))
    st.update(DATASET_CONFIG=str(tmp_path / "ds.toml"), LORA_OUTPUT_DIR=str(tmp_path / "out"))
    return st


class _Plan:
    files, console, dirs = [], [], []


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1] if name in cmd else None


def test_form_row_and_default():
    assert form.DEFAULTS["VALIDATION_GROUPS"] == "1"
    rows = {r["key"]: r for s in form.form(DESC)["sections"] for r in s["rows"]}
    row = rows["VALIDATION_GROUPS"]
    assert row["kind"] == "choice" and row["choices"] == ["1", "2", "3"] and row["default"] == "1"
    assert "colour" not in row["hint"].lower() and "test" not in row["hint"].lower()


def test_form_notes_say_what_the_groups_make():
    st = form.state(DESC, {"VALIDATION_GROUPS": "3", "VALIDATION_IMAGES": "4"}, "")
    assert "3 groups of 4 images: 12 in all" in st["notes"]["VALIDATION_GROUPS"]
    st = form.state(DESC, {"VALIDATION_GROUPS": "2", "VALIDATION_PICKED": ["a.png", "b.png", "c.png"]}, "")
    assert "3 picked" in st["notes"]["VALIDATION_GROUPS"]
    assert "VALIDATION_GROUPS" not in form.state(DESC, {"VALIDATION_GROUPS": "1"}, "")["notes"]


def test_command_line_carries_the_groups(tmp_path):
    for val, want in (("3", "3"), ("2", "2"), ("1", "1"), ("", "1"), ("9", "3"), ("zero", "1")):
        st = _inputs(tmp_path, VALIDATION_LOSS=True, VALIDATION_GROUPS=val)
        cmd = launch.train_command(DESC, st, _Plan())
        assert _flag(cmd, "--validation_groups") == want, (val, want)
        assert _flag(cmd, "--validation_images") == "4"
    st = _inputs(tmp_path, VALIDATION_LOSS=True)                       # nothing set: the old default
    assert _flag(launch.train_command(DESC, st, _Plan()), "--validation_groups") == "1"


def test_holdout_pick_uses_the_groups(tmp_path):
    d = tmp_path / "train"
    toml = f'[general]\n[[datasets]]\nimage_directory = "{d.as_posix()}"\n'
    st = _inputs(tmp_path, VALIDATION_GROUPS="3", VALIDATION_IMAGES="4")
    paths, names, how, why = launch.holdout_pick(DESC, st, toml)
    assert how == "auto" and len(paths) == 6 and "3 validation groups would need 12" in why      # 30 pictures: a fifth
    st = _inputs(tmp_path, VALIDATION_GROUPS="1", VALIDATION_IMAGES="4")
    assert len(launch.holdout_pick(DESC, st, toml)[0]) == 3                                          # a tenth, as ever


def test_trainer_takes_the_flag_and_hands_it_to_the_check():
    src = open(os.path.join(ROOT, "families", "train.py"), encoding="utf-8").read()
    assert '"--validation_groups"' in src and "validation_groups=a.validation_groups" in src
    assert "groups=validation_groups, shapes=_val_shapes" in src


# ---- the graph's styles and data

def test_graph_colours_have_colour_blind_safe_overrides():
    css = open(os.path.join(ROOT, "web", "static", "style.css"), encoding="utf-8").read()
    for n in (1, 2, 3):
        assert f"--vg-{n}:" in css
    safe = css[css.index(':root[data-theme="colour-safe"] {'):]
    safe = safe[:safe.index("}")]
    assert "--vg-1:" in safe and "--vg-2:" in safe and "--vg-3:" in safe
    # each group also has its own dash pattern, so colour is never the only cue
    assert ".lg-vg-2 { stroke: var(--vg-2); stroke-dasharray:" in css and ".lg-vg-3 { stroke: var(--vg-3); stroke-dasharray:" in css


def test_graph_reads_the_groups_without_breaking_old_runs():
    js = open(os.path.join(ROOT, "web", "static", "app.js"), encoding="utf-8").read()
    assert "data.val_groups" in js and "valGroupParts" in js and "vgMark" in js
    assert "const hasGroups = hasVal && nGroups > 1" in js          # no val_groups (an older run): nothing extra is drawn
    assert "gvals" in js and "vgOn" in js and "vgOverlayOffsets" in js
    assert "bandOn" not in js and "lg-vg-band" not in js and "Group range" not in js
    css = open(os.path.join(ROOT, "web", "static", "style.css"), encoding="utf-8").read()
    assert "lg-vg-band" not in css and "lg-sw-vgband" not in css


def test_group_lines_shift_onto_the_average_and_keep_the_real_loss():
    """The group toggle draws each line offset onto the average so the scale stays tight;
    tooltips still show the real loss. Offsets are one a group a section (shared checks)."""
    import json, shutil, subprocess
    if not shutil.which("node"):
        pytest.skip("needs node")
    js = open(os.path.join(ROOT, "web", "static", "app.js"), encoding="utf-8").read()
    assert "shifted onto the " in js and "average to compare their shape" in js
    assert "offset to the average" in js
    # the form help and the training notes agree
    form_src = open(os.path.join(ROOT, "families", "form.py"), encoding="utf-8").read()
    assert "shifted onto the average so you can compare their shape" in form_src
    assert "under the average" not in form_src
    start = js.index("function valSections(val)")
    end = js.index("function vgMark(")
    fns = js[start:end]
    script = fns + r"""
const val = [
  [10, 1, 0.160, null, 0],
  [20, 2, 0.161, null, 0],
  [30, 3, 0.158, null, 0],
  [40, 4, 0.160, null, 0],
  [50, 5, 0.200, null, 1],
  [60, 6, 0.220, null, 1],
];
const groups = [
  [10, 1, [0.140, 0.180], 0],
  [20, 2, [0.141, 0.181], 0],
  [30, 3, [0.138, 0.178], 0],
  [40, 4, [0.140, 0.180], 0],
  [50, 5, [0.300, 0.100], 1],
  [60, 6, [0.320, 0.120], 1],
];
const off = vgOverlayOffsets(val, groups, 2);
const shifted = (seg, gi, x) => x + off.get(`${seg}:${gi}`);
const mean = (xs) => xs.reduce((s, x) => s + x, 0) / xs.length;
const g1s0 = [0.140, 0.141, 0.138, 0.140].map((x) => shifted(0, 0, x));
const g2s0 = [0.180, 0.181, 0.178, 0.180].map((x) => shifted(0, 1, x));
const val0 = [0.160, 0.161, 0.158, 0.160];
const g1s1 = [0.300, 0.320].map((x) => shifted(1, 0, x));
const g2s1 = [0.100, 0.120].map((x) => shifted(1, 1, x));
const val1 = [0.200, 0.220];
const raw = groups.flatMap((g) => g[2]);
const out = {
  off00: off.get("0:0"), off01: off.get("0:1"), off10: off.get("1:0"), off11: off.get("1:1"),
  g1s0, g2s0, g1s1, g2s1,
  meanG1s0: mean(g1s0), meanVal0: mean(val0),
  meanG1s1: mean(g1s1), meanVal1: mean(val1),
  dipG1: g1s0[2] - g1s0[0], dipVal: val0[2] - val0[0],
  rawSpan: Math.max(...raw) - Math.min(...raw),
  shiftSpan: Math.max(...g1s0, ...g2s0) - Math.min(...g1s0, ...g2s0),
  valSpan: Math.max(...val0) - Math.min(...val0),
};
console.log(JSON.stringify(out));
"""
    out = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    assert abs(out["off00"] - 0.02) < 1e-12 and abs(out["off01"] - -0.02) < 1e-12
    assert abs(out["off10"] - -0.10) < 1e-12 and abs(out["off11"] - 0.10) < 1e-12
    assert abs(out["meanG1s0"] - out["meanVal0"]) < 1e-12
    assert abs(out["meanG1s1"] - out["meanVal1"]) < 1e-12
    want0 = [0.160, 0.161, 0.158, 0.160]
    assert all(abs(a - b) < 1e-12 for a, b in zip(out["g1s0"], want0))
    assert all(abs(a - b) < 1e-12 for a, b in zip(out["g2s0"], want0))
    assert abs(out["dipG1"] - out["dipVal"]) < 1e-12
    assert abs(out["shiftSpan"] - out["valSpan"]) < 1e-12
    assert out["rawSpan"] > 0.04 and out["shiftSpan"] < 0.005
