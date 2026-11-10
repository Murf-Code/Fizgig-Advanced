"""PoLoRA's Magnitude / Curl / Compile boxes (each one its Optimizer Args word, shown with PoLoRA only, kept in step
both ways) and every family's built-in "PoLoRA (experimental)" preset."""
import json
import os
import re
import shutil
import subprocess

import pytest

from fizgig.families import form
from fizgig.families.description import (POLORA_CHOICE_LABEL, POLORA_EXPERIMENTAL_NOTE, POLORA_PRESET,
                                         POLORA_PRESET_ALIASES)
from fizgig.families.registry import FAMILIES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_boxes_show_with_polora_only_and_read_the_args():
    desc = FAMILIES["krea2"]
    rows = {r["key"]: r for s in form.form(desc)["sections"] for r in s["rows"] if r}
    assert [rows[k]["label"] for k in form.POLORA_BOXES] == ["Magnitude", "Curl", "Compile"]
    assert all(rows[k]["hint"] for k in form.POLORA_BOXES) and rows["POLORA_COMPILE"]["kind"] == "check"
    opt = rows["OPTIMIZER_TYPE"]
    assert "polora" in opt["choices"]
    assert opt["choice_labels"]["polora"] == POLORA_CHOICE_LABEL == "PoLoRA (experimental)"
    assert opt["choice_tips"]["polora"] == POLORA_EXPERIMENTAL_NOTE
    st = form.state(desc, {**form.DEFAULTS})
    assert "OPTIMIZER_TYPE" not in st.get("notes", {})          # the note only shows when PoLoRA is picked
    assert set(form.POLORA_BOXES) <= set(st["hidden"])
    v = {**form.DEFAULTS, "OPTIMIZER_TYPE": "polora", "OPTIMIZER_ARGS": "magnitude_scale=2.5 compile=True ns_steps=6"}
    st = form.state(desc, v)
    assert not set(form.POLORA_BOXES) & set(st["hidden"])
    assert st["notes"]["OPTIMIZER_TYPE"] == POLORA_EXPERIMENTAL_NOTE
    assert st["force"]["POLORA_MAGNITUDE"] == "2.5" and "POLORA_COMPILE" not in st["force"]   # ticked already
    assert "POLORA_CURVATURE" not in st["force"]                     # not in the args: empty (PoLoRA's 0.99)
    assert not set(form.POLORA_BOXES) & set(form.state(desc, {**v, **st["force"]})["force"])   # in step: nothing
    # a preset holds the args only (no duplicates)
    assert not set(form.POLORA_BOXES) & set(form.to_preset(desc, {**v, **st["force"]}))


def test_compile_is_on_unless_the_args_turn_it_off():
    import inspect
    from fizgig.training.polora import Polora
    assert inspect.signature(Polora.__init__).parameters["compile"].default is True
    desc = FAMILIES["krea2"]
    assert form.DEFAULTS["POLORA_COMPILE"] is True
    for args, want in (("magnitude_scale=3", True), ("", True), ("compile=True", True), ("compile=False", False),
                       ("compile=false", False), ("compile=0", False)):
        v = {**form.DEFAULTS, "OPTIMIZER_TYPE": "polora", "OPTIMIZER_ARGS": args}
        st = form.state(desc, v)
        assert st["force"].get("POLORA_COMPILE", v["POLORA_COMPILE"]) is want, args
    # an older form that had the box unticked by default and no compile word: it shows ticked now
    st = form.state(desc, {**form.DEFAULTS, "POLORA_COMPILE": False, "OPTIMIZER_TYPE": "polora",
                           "OPTIMIZER_ARGS": "magnitude_scale=3"})
    assert st["force"]["POLORA_COMPILE"] is True
    from fizgig.training.optimizers import parse_optimizer_args
    assert parse_optimizer_args("compile=False") == {"compile": False}
    src = open(os.path.join(ROOT, "src/fizgig/web/static/app.js"), encoding="utf-8").read()
    assert 'const POLORA_DEFAULT_ARGS = {magnitude_scale: "3", compile: "True"};' in src      # filled when picked
    assert '(val ? "True" : "False")' in src                                                  # unticked: compile=False


@pytest.mark.skipif(not shutil.which("node"), reason="needs node")
def test_a_box_writes_its_word_into_the_args():
    src = open(os.path.join(ROOT, "src/fizgig/web/static/app.js"), encoding="utf-8").read()
    fn = re.search(r"function setOptArg\(raw, key, val\) \{.*?\n\}\n", src, re.S).group()
    cases = [["magnitude_scale=3 ns_steps=6", "magnitude_scale", "2.5"], ["magnitude_scale=3", "compile", "True"],
             ["a=1 compile=True b=2", "compile", ""], ["", "curvature_beta", "0.95"]]
    js = fn + f"console.log(JSON.stringify({json.dumps(cases)}.map((c) => setOptArg(...c))));"
    out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    assert out == ["magnitude_scale=2.5 ns_steps=6", "magnitude_scale=3 compile=True", "a=1 b=2",
                   "curvature_beta=0.95"]
    assert 'POLORA_BOXES = {POLORA_MAGNITUDE: "magnitude_scale", POLORA_CURVATURE: "curvature_beta", ' \
           'POLORA_COMPILE: "compile"}' in src
    assert "function choiceText(r, c)" in src and "function choiceTip(r, c)" in src
    assert "text: choiceText(r, c)" in src


def _standard_lora(desc):
    """The family's standard LoRA preset: the first one a first visit applies (not PoLoRA, an edit, slider or fine-tune)."""
    return next((n, dict(v)) for n, v in desc.presets
                if n not in POLORA_PRESET_ALIASES and str(v.get("OPTIMIZER_TYPE") or "").lower() != "polora"
                and not (v.get("FAMILY_EDIT") or v.get("FAMILY_SLIDER") or v.get("FAMILY_FT")))


@pytest.mark.parametrize("key", FAMILIES)
def test_every_family_has_the_polora_preset(key):
    desc = FAMILIES[key]
    names = [n for n, _v in desc.presets]
    assert names[-1] == POLORA_PRESET and names.count(POLORA_PRESET) == 1 and names[0] != POLORA_PRESET
    assert "PoLoRA" not in names                                      # the old default name is not kept beside it
    p = dict(desc.presets)[POLORA_PRESET]
    assert (p["OPTIMIZER_TYPE"], p["ADAPTIVE_LR"], p["ADAPTIVE_LR_MIN"], p["ADAPTIVE_LR_MAX"]) == \
        ("polora", True, "1e-2", "4e-2")
    assert p["LEARNING_RATE"] == "2e-2"
    assert form._arg_words(p["OPTIMIZER_ARGS"]) == {"magnitude_scale": "3", "curvature_beta": "0.99",
                                                    "compile": "True"}
    assert not (p.get("FAMILY_SLIDER") or p.get("FAMILY_FT") or p.get("FAMILY_EDIT"))
    std_name, std = _standard_lora(desc)
    assert names[0] == std_name and str(std.get("OPTIMIZER_TYPE") or "").lower() != "polora"
    assert p.get("NETWORK_DIM") == std.get("NETWORK_DIM")                # the rest: the family's own
    assert float(desc.lr) < 1e-2                                         # the family's usual rate isn't PoLoRA's
    out = form.apply_preset(desc, dict(form.DEFAULTS), p)
    assert out["OPTIMIZER_TYPE"] == "polora" and out["LEARNING_RATE"] == "2e-2"
    assert "compile=True" in out["OPTIMIZER_ARGS"] and out["ADAPTIVE_LR"] is True
    # a first visit applies the standard preset, not PoLoRA
    first = form.apply_preset(desc, dict(form.DEFAULTS), desc.presets[0][1])
    assert str(first.get("OPTIMIZER_TYPE") or form.DEFAULTS["OPTIMIZER_TYPE"]).lower() != "polora"
    old = form.apply_preset(desc, dict(form.DEFAULTS), std)
    assert str(old.get("OPTIMIZER_TYPE") or form.DEFAULTS["OPTIMIZER_TYPE"]).lower() != "polora"


def test_a_saved_run_keeps_its_optimizer_and_a_first_visit_is_the_standard_preset():
    """Last Train / a saved form keeps its optimizer; a first visit uses the standard preset, not PoLoRA.
    Start and the form don't flag PoLoRA's usual 2e-2 when you do pick it."""
    from fizgig.families.launch import start_checks
    desc = FAMILIES["krea2"]
    saved = {**form.DEFAULTS, "OPTIMIZER_TYPE": "muon", "LEARNING_RATE": "1e-4", "ADAPTIVE_LR": True,
             "ADAPTIVE_LR_MIN": "2e-4", "ADAPTIVE_LR_MAX": "4e-4"}
    st = form.state(desc, saved)
    assert st["force"].get("OPTIMIZER_TYPE", saved["OPTIMIZER_TYPE"]) == "muon"
    assert st["force"].get("LEARNING_RATE", saved["LEARNING_RATE"]) == "1e-4"
    assert desc.presets[0][0] != POLORA_PRESET
    first = form.apply_preset(desc, dict(form.DEFAULTS), desc.presets[0][1])
    assert str(first.get("OPTIMIZER_TYPE") or form.DEFAULTS["OPTIMIZER_TYPE"]).lower() != "polora"
    picked = form.apply_preset(desc, dict(form.DEFAULTS), dict(desc.presets)[POLORA_PRESET])
    assert picked["OPTIMIZER_TYPE"] == "polora" and picked["LEARNING_RATE"] == "2e-2"
    st = form.state(desc, picked)
    assert st["force"].get("LEARNING_RATE", picked["LEARNING_RATE"]) == "2e-2"
    assert st["notes"]["OPTIMIZER_TYPE"] == POLORA_EXPERIMENTAL_NOTE
    assert start_checks(desc, picked, "") == []
    # a slider / edit / fine-tune preset keeps its own optimizer (not switched to PoLoRA)
    other = next((v for _n, v in desc.presets if v.get("FAMILY_FT") or v.get("FAMILY_SLIDER") or v.get("FAMILY_EDIT")), None)
    assert other is not None
    applied = form.apply_preset(desc, dict(form.DEFAULTS), other)
    assert str(applied.get("OPTIMIZER_TYPE") or form.DEFAULTS["OPTIMIZER_TYPE"]).lower() != "polora"
