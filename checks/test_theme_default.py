"""Deep Space is Fizgig Advanced's default theme for a device that never picked one; a picked theme (the classic
Fizgig one included, kept as "fizgig") stays. index.html's first-paint script run in node with a fake page."""
import json
import os
import re
import shutil
import subprocess

import pytest

STATIC = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static")


def _read(name):
    with open(os.path.join(STATIC, name), encoding="utf-8") as f:
        return f.read()


@pytest.mark.skipif(not shutil.which("node"), reason="needs node")
def test_first_paint_theme():
    html = _read("index.html")
    script = re.search(r"<script>\s*(// the theme this device picked.*?)</script>", html, re.S).group(1)
    cases = [None, "fizgig", "graphite", "deep-space", "THROW"]
    js = ("const out = [];\nfor (const saved of %s) {\n"
          "  const document = {documentElement: {dataset: {}}};\n"
          "  const localStorage = {getItem: () => { if (saved === 'THROW') throw new Error('blocked'); return saved; }};\n"
          "  %s\n  out.push(document.documentElement.dataset.theme || '');\n}\n"
          "console.log(JSON.stringify(out));") % (json.dumps(cases), script)
    out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    assert out == ["deep-space", "", "graphite", "deep-space", "deep-space"]


def test_picking_the_classic_theme_is_remembered():
    js = _read("app.js")
    fn = js[js.index("function applyTheme(key)"):js.index("function wireTheme()")]
    assert 'localStorage.setItem("fizgig.theme", key || "fizgig")' in fn and "removeItem" not in fn
    assert 'const DEFAULT_THEME = "deep-space"' in js and '{key: "deep-space"' in js
    assert '<meta name="theme-color" content="#07070D">' in _read("index.html")    # Deep Space's --bg-deep
