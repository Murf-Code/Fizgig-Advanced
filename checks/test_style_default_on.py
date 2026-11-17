"""The gallery's Style score is on by default: a run (and its training folder) where it was never switched on or off
scores and shows it; switched off, it stays off (and back on when switched on). Per run and per training folder since
upstream's 11 Nov 2026 change (style_<run>.json, <dataset>/fizgig_style.json)."""
import json

from fizgig.web import gallery_style as gs


def test_style_score_on_by_default_and_off_stays_off(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(gs, "_start", lambda s, r, d: started.append((s, r)))
    samples, data = tmp_path / "sample", tmp_path / "data"
    samples.mkdir(), data.mkdir()
    assert gs.state(str(samples), "run", str(data))["on"] is True and started == [(str(samples), "run")]   # never chosen: on
    gs.switch(str(samples), "run", str(data), False)
    started.clear()
    assert gs.state(str(samples), "run", str(data))["on"] is False and started == []          # off is remembered
    assert json.loads((samples / "style_run.json").read_text())["on"] is False
    assert gs.state(str(samples), "other_run", str(data))["on"] is False                      # ...for the folder too
    gs.switch(str(samples), "run", str(data), True)
    assert gs.state(str(samples), "run", str(data))["on"] is True
    gs.switch(str(samples), "run", str(data), False)


def test_no_training_folder_shows_on_but_scores_nothing(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(gs, "_start", lambda s, r, d: started.append(s))
    (tmp_path / "sample").mkdir()
    assert gs.state(str(tmp_path / "sample"), "run", "")["on"] is True and started == []
