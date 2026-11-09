"""The gallery's Style score is on by default: a samples folder where it was never switched on or off scores and
shows it; switched off, it stays off (and back on when switched on)."""
import json

from fizgig.web import gallery_style as gs


def test_style_score_on_by_default_and_off_stays_off(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(gs, "_start", lambda s, d: started.append(s))
    samples, data = tmp_path / "sample", tmp_path / "data"
    samples.mkdir(), data.mkdir()
    assert gs.state(str(samples), str(data))["on"] is True and started == [str(samples)]   # never chosen: on
    gs.switch(str(samples), str(data), False)
    started.clear()
    assert gs.state(str(samples), str(data))["on"] is False and started == []              # off is remembered
    assert json.loads((samples / "style.json").read_text())["on"] is False
    gs.switch(str(samples), str(data), True)
    assert gs.state(str(samples), str(data))["on"] is True
    gs.switch(str(samples), str(data), False)


def test_no_training_folder_shows_on_but_scores_nothing(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(gs, "_start", lambda s, d: started.append(s))
    (tmp_path / "sample").mkdir()
    assert gs.state(str(tmp_path / "sample"), "")["on"] is True and started == []
