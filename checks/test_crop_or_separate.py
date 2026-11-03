"""Crop to Other…'s Find this: 'ass OR pussy' crops whichever is found (the first choice wins), and Separate crop for
each word gives every word found its own close-up (with Fit all words on too: both). Also the page bits that go
with it: the tick, the hint, and the Training / Samples tabs opening at the run's graph and previews."""
import os

import numpy as np
from PIL import Image

from fizgig.dataset import detectors as D
from fizgig.dataset import prep
from test_detectors import Stub

IMG = Image.new("RGB", (1000, 800))
AIO = Stub({1: "pussy", 2: "anus"}, [(300, 300, 360, 360, 0.9, 1), (700, 600, 740, 640, 0.8, 2)], "aio")
ASS = Stub({0: "ass"}, [(100, 100, 400, 400, 0.9, 0)], "assdetailer")
NOASS = Stub({0: "ass"}, [], "assdetailer")


def test_split_or():
    assert prep.split_or("ass OR pussy") == ["ass", "pussy"] and prep.split_or("ass or butt | rear") == \
        ["ass", "butt", "rear"]
    assert prep.split_or("orange hat") == ["orange hat"]          # 'or' inside a word is just a word


def _picks(words, stages, fit=True, sep=False):
    f = D.chain_finder(words, stages, fit_all=fit, log=lambda _t: None)
    f.separate = sep
    p = prep.Prep(lambda _t: None, None, f)
    p._padding = 0
    return [s.gender for s in p._picks(IMG, "", single=True)]


def test_or_takes_the_first_choice_found():
    assert _picks(["ass OR pussy"], [("aio", AIO), ("ass", ASS)]) == ["ass"]
    assert _picks(["ass OR pussy"], [("aio", AIO), ("ass", NOASS)]) == ["pussy"]
    # with another word: one crop around the choice found and that word
    assert _picks(["ass OR pussy", "anus"], [("aio", AIO), ("ass", NOASS)]) == ["pussy_anus"]


def test_separate_crop_for_each_word():
    st = [("aio", AIO), ("ass", ASS)]
    assert _picks(["pussy", "anus", "ass"], st, fit=False, sep=True) == ["pussy", "anus", "ass"]
    assert _picks(["pussy", "anus", "ass"], st, fit=True, sep=True) == ["pussy_anus_ass", "pussy", "anus", "ass"]
    assert _picks(["pussy", "anus"], st, fit=True, sep=False) == ["pussy_anus"]


def test_or_and_separate_in_a_prep_run(tmp_path):
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (800, 1000, 3), dtype=np.uint8)).save(tmp_path / "a.png")
    f = D.chain_finder(["ass OR pussy", "anus"], [("aio", AIO), ("ass", NOASS)], fit_all=True, log=lambda _t: None)
    f.separate = True
    prep.Prep(lambda _t: None, None, f).run("Face Crop Only", str(tmp_path), 0.05, prep.OTHER_TARGET, 20.0, False)
    assert sorted(os.listdir(tmp_path / "crops")) == ["Crop_anus_001.png", "Crop_pussy_001.png",
                                                     "Crop_pussy_anus_001.png"]


def test_find_words_routes_each_choice(monkeypatch):
    from fizgig.web import prep_jobs
    monkeypatch.setattr(D, "stages", lambda log, b: [("aio", AIO), ("ass", ASS)])
    words, sam = prep_jobs.find_words(prep.OTHER_TARGET, "ass OR pussy, anus", backend=object())
    assert words == ["ass OR pussy", "anus"]                # no SAM 3.1 needed: both choices have a model


def test_page():
    here = os.path.join(os.path.dirname(prep.__file__), "..", "web", "static")
    html = open(os.path.join(here, "index.html"), encoding="utf-8").read()
    js = open(os.path.join(here, "app.js"), encoding="utf-8").read()
    assert 'id="prepSeparate"' in html and "Separate crop for each word" in html
    assert 'separate: $("prepSeparate").checked' in js and "use OR for either one" in js
    from fizgig.web import server
    src = open(server.__file__, encoding="utf-8").read()
    assert 'bool(b.get("separate"))' in src
    # while a run trains, the tabs open at the run's graph / previews
    assert 'TAB_RUN_SPOT = {training: "#tab-training .run-card", samples: "#galleryCard"}' in js
    assert "showTab(b.dataset.tab); tabRunScroll(b.dataset.tab);" in js
    assert 'id="galleryCard"' in html and 'class="section-card accent-card run-card"' in html
