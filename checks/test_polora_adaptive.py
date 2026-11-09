"""PoLoRA's Adaptive LR works like Muon's: Min/Max are real learning rates (the run starts at their midpoint, the LR
box is greyed), magnitude_scale stays as set; its factor sizes go to the graph and the log, and the graph's
"PoLoRA factor size" switch shows or hides both. A tiny CPU family trains for real (test_save_stop's)."""
import json
import math
import os
import re

from test_save_stop import _losses, _run, data, tiny  # noqa: F401 - the fixtures

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_polora_adaptive_lr_moves_the_real_rate(tiny, tmp_path, caplog):
    T, cfg = tiny
    caplog.set_level("INFO")
    out = tmp_path / "pol"
    assert _run(T, cfg, out, optimizer_type="polora", optimizer_args="magnitude_scale=3 log_norms_every=2",
                learning_rate=2e-2, lr_scheduler="constant", adaptive_lr=True, adaptive_lr_min=1e-2,
                adaptive_lr_max=9e-2, max_train_epochs=2, save_state=False) == "done"
    log = _losses(out)
    assert math.isclose(log[0]["lr"], 3e-2, rel_tol=1e-6)            # sqrt(1e-2 * 9e-2): Min/Max are real rates
    assert all(1e-2 - 1e-12 <= r["lr"] <= 9e-2 + 1e-12 for r in log)
    mags = [float(m) for m in re.findall(r"\[polora\] step \d+: .*?magnitude ([\d.]+)", caplog.text)]
    assert mags and all(abs(m - 3.0) < 1e-6 for m in mags)            # magnitude_scale stays as set
    assert "moves the learning rate between Min and Max" in caplog.text
    assert "magnitude factor" not in caplog.text
    with open(os.path.join(str(out), ".fizgig_loss_t.jsonl"), encoding="utf-8") as f:
        assert any('"k": "pf"' in ln for ln in f)                   # the graph's factor size line


def test_form_greys_the_lr_box_and_the_page_links_graph_and_log():
    from fizgig.families import form
    from fizgig.families.registry import FAMILIES
    desc = FAMILIES["zimage"]
    v = {**form.DEFAULTS, "OPTIMIZER_TYPE": "polora", "LEARNING_RATE": "2e-2", "ADAPTIVE_LR": True,
         "ADAPTIVE_LR_MIN": "1e-2", "ADAPTIVE_LR_MAX": "4e-2"}
    st = form.state(desc, v, "")
    assert "LEARNING_RATE" in st["disabled"]
    assert not {"ADAPTIVE_LR_MIN", "ADAPTIVE_LR_MAX"} & set(st["hidden"])
    assert "real learning rates" in st["notes"]["ADAPTIVE_LR"]
    js = open(os.path.join(ROOT, "src", "fizgig", "web", "static", "app.js"), encoding="utf-8").read()
    assert 'POLORA_ADAPTIVE = {ADAPTIVE_LR_MIN: "1e-2", ADAPTIVE_LR_MAX: "4e-2"}' in js
    assert "if (polFactorHidden(line)) return;" in js and 'line.includes("[polora] step ")' in js
    json.dumps(st)
