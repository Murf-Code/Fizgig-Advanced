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


def test_preferences_opens_at_its_top():
    # each tab returns to its own scroll place (upstream's showTab), except Preferences: always its top
    show = _fn(_src(), "showTab")
    assert re.search(r'const \[py, wy\] = name === "prefs" \? \[0, 0\] : S\.tabScroll\["tab-" \+ name\]', show)
    assert "put();" in show


def _rows_js(results):
    rows = ", ".join("{result: %s, scrolled: false, scrollIntoView() { this.scrolled = true; }}" % json.dumps(r)
                     for r in results)
    return ("const rows = [" + rows + "];"
            "const host = {querySelector: (sel) => rows.find((r) => sel === "
            "`.run-row[data-result=\"${r.result}\"]`) || null};"
            "const requestAnimationFrame = (f) => f();"
            + _fn(_src(), "runsScrollToCurrent") +
            "const hit = runsScrollToCurrent(host);"
            "console.log(JSON.stringify({hit, scrolled: rows.map((r) => r.scrolled)}));")


def test_runs_and_outputs_goes_to_the_run_training_now():
    assert _node(_rows_js(["running", "finished", "paused"])) == {"hit": True, "scrolled": [True, False, False]}
    assert _node(_rows_js(["finished", "paused"])) == {"hit": False, "scrolled": [False, False]}  # not training: stays


def test_only_the_tabs_own_click_asks_for_it():
    src = _src()
    assert 'if (name === "outputs") { S.runsToCurrent = true; return false; }' in _fn(src, "tabRunScroll")
    runs = _fn(src, "loadRuns")
    # read after the fetch, since the tab's click sets it just after showTab has started loadRuns
    assert runs.index('await api("/api/runs")') < runs.index("const toCurrent = S.runsToCurrent")
    assert '"data-result": run.result' in runs and "if (toCurrent) runsScrollToCurrent(host);" in runs
