"""Optimizer section options have a short hover tip (the label's ⓘ), and the visible help under
the boxes is short enough to show without a click."""
import json
import os
import shutil
import subprocess

import pytest

from fizgig.families import form
from fizgig.families.registry import get as get_family

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "..", "src", "fizgig", "web", "static")

# every option in the Optimizer card, including PoLoRA's boxes (shown with PoLoRA only)
OPT_KEYS = (
    "OPTIMIZER_TYPE", "OPTIMIZER_ARGS", "POLORA_MAGNITUDE", "POLORA_CURVATURE", "POLORA_COMPILE",
    "LR_SCHEDULER", "LR_WARMUP_STEPS", "CAPTION_DROPOUT", "GRADIENT_ACCUMULATION", "MAX_GRAD_NORM",
)


def _opt_rows(desc):
    sec = next(s for s in form.form(desc)["sections"] if s["id"] == "optimizer")
    return {r["key"]: r for r in sec["rows"] if r}


def test_every_optimizer_option_has_a_short_hover_tip():
    rows = _opt_rows(get_family("krea2"))
    assert set(OPT_KEYS) <= set(rows)
    for key in OPT_KEYS:
        r = rows[key]
        tip = r.get("tip") or ""
        assert tip, key
        # 1-2 short sentences: no "experimental", and not a wall of text
        assert "experimental" not in tip.lower()
        assert len(tip) < 280, (key, tip)
    assert rows["POLORA_COMPILE"]["hint"] == (
        "Speeds up PoLoRA after a 30-second warm-up. Training results are the same. Untick if it "
        "errors or runs out of VRAM.")
    assert "compile=False" in rows["POLORA_COMPILE"]["tip"] and "triton" in rows["POLORA_COMPILE"]["tip"]
    # visible help is short (the long detail lives in the tip)
    for key in ("POLORA_MAGNITUDE", "POLORA_CURVATURE", "CAPTION_DROPOUT", "GRADIENT_ACCUMULATION"):
        assert len(rows[key]["hint"]) < 120 and "..." not in rows[key]["hint"]
    # each optimizer and scheduler in the lists has its own hover, including ones with no extra boxes
    for name in ("prodigy", "muon", "automagic3", "automagic4", "polora", "soap", "lora_tsd", "adamw8bit"):
        assert name in rows["OPTIMIZER_TYPE"]["choice_tips"]
    for name in ("constant", "cosine", "linear", "polynomial"):
        assert name in rows["LR_SCHEDULER"]["choice_tips"]
    # Prodigy, Muon and Automagic have no extra boxes of their own; their settings stay in Optimizer Args
    assert not any(k.startswith(("PRODIGY_", "MUON_", "AUTOMAGIC_")) for k in rows)


JS = r"""
const {JSDOM} = require("jsdom"); const fs = require("fs");
const dom = new JSDOM("<body><div id='trainingForm'></div></body>");
global.window = dom.window; global.document = dom.window.document;
const src = fs.readFileSync(process.env.STATIC + "/app.js", "utf8");
const fn = (name) => { const i = src.indexOf("function " + name + "("); const a = src.lastIndexOf("\n", i) + 1;
  return src.slice(a, src.indexOf("\n}\n", i) + 3); };
const code = src.match(/const \$ = \(id\)[^\n]*\n/)[0] + src.match(/const el = \(tag[\s\S]*?\n};\n/)[0] +
  src.match(/function choiceAlias[\s\S]*?\n\}\n/)[0] + src.match(/function choiceText[\s\S]*?\n\}\n/)[0] +
  src.match(/function choiceTip[\s\S]*?\n\}\n/)[0] + fn("renderSections") + fn("renderRow");
const FORM = JSON.parse(process.env.FORM);
const S = {fam: {form: FORM}, values: Object.fromEntries(FORM.sections[0].rows.filter(Boolean).map((r) =>
  [r.key, r.kind === "check" ? !!r.default : (r.default ?? "")])), family: "krea2", secShut: {}};
const onValue = (r, v) => { S.values[r.key] = v; };
eval(code + "; global.renderSections = renderSections;");
renderSections("training", document.getElementById("trainingForm"));
const out = {};
for (const key of JSON.parse(process.env.KEYS)) {
  const row = document.querySelector(`.form-row[data-key="${key}"]`);
  const host = row && (row.querySelector("label.checkbox-label") || row.querySelector(".field-label"));
  const hint = row && [...row.querySelectorAll(".field-hint")].find((h) => !h.classList.contains("row-note"));
  out[key] = row ? {
    title: host ? host.title : "",
    mark: !!(host && host.querySelector(".tip-mark")),
    hint: hint ? hint.textContent : "",
    options: [...(row.querySelectorAll("select option") || [])].map((o) => [o.value, o.title || ""]),
  } : null;
}
console.log(JSON.stringify(out));
"""


def test_page_draws_a_hover_mark_on_each_optimizer_option():
    from checks.test_val_picker import JSDOM_DIRS
    if not shutil.which("node"):
        pytest.skip("node not installed")
    d = next((p for p in JSDOM_DIRS if os.path.isdir(os.path.join(p, "jsdom"))), None)
    if d is None:
        pytest.skip("jsdom not installed")
    fm = form.form(get_family("krea2"))
    sec = next(s for s in fm["sections"] if s["id"] == "optimizer")
    fm = {"sections": [sec]}
    out = json.loads(subprocess.check_output(["node", "-e", JS], text=True, env={
        **os.environ, "NODE_PATH": d, "STATIC": os.path.abspath(STATIC),
        "FORM": json.dumps(fm), "KEYS": json.dumps(list(OPT_KEYS))}))
    rows = _opt_rows(get_family("krea2"))
    for key in OPT_KEYS:
        got = out[key]
        assert got and got["mark"] and got["title"] == rows[key]["tip"], key
        if key == "POLORA_COMPILE":
            assert got["hint"] == form.POLORA_COMPILE_HINT
        if key == "OPTIMIZER_TYPE":
            tips = dict(got["options"])
            assert tips["prodigy"] == form.OPTIMIZER_TYPE_TIPS["prodigy"]
            assert tips["polora"] == form.OPTIMIZER_TYPE_TIPS["polora"]
        if key == "LR_SCHEDULER":
            tips = dict(got["options"])
            assert tips["cosine"] == form.LR_SCHEDULER_TIPS["cosine"]
