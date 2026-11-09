"""Where a tab opens: the tabs share one scrolling pane, so Preferences goes back to its top instead of opening
wherever the last tab was scrolled to."""
import json
import os
import re
import shutil
import subprocess

import pytest

APP = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static", "app.js")


def _src():
    with open(APP, encoding="utf-8") as f:
        return f.read()


def _fn(src, name):
    return re.search(r"(?:async )?function " + name + r"\(.*?\n\}\n", src, re.S).group(0)


def _node(js):
    if not shutil.which("node"):
        pytest.skip("node not installed")
    return json.loads(subprocess.check_output(["node", "-e", js], text=True))


def test_pane_top_scrolls_the_pane_and_the_page():
    js = ("const pane = {scrollTop: 900}; let win = null;"
          "const document = {querySelector: (s) => s === '.notebook-pane' ? pane : null};"
          "const window = {scrollTo: (x, y) => { win = [x, y]; }};"
          + _fn(_src(), "paneTop") +
          "paneTop(); console.log(JSON.stringify({pane: pane.scrollTop, win}));")
    assert _node(js) == {"pane": 0, "win": [0, 0]}


def test_preferences_opens_at_its_top():
    show = _fn(_src(), "showTab")
    assert re.search(r'if \(name === "prefs"\) paneTop\(\);', show)
