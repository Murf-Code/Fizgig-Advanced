"""Image Prep's detector chain (Other…): each word goes to the first model that knows it (Fdetailer, booru_yolo,
All-in-One, assDetailer), SAM 3.1 last; a detector's boxes are trusted however big (only SAM's oversized ones are
dropped); a missing model is skipped; the log names the model behind each box. Watermark masks are made by Make masks
(beside the picture), never by a prep run, and a watermark box over a tenth of the picture is dropped. Stub models
stand in for the ONNX ones."""
import os

import numpy as np
from PIL import Image

from fizgig.dataset import detectors as D
from fizgig.dataset import prep
from fizgig.dataset.masks import MASK_SUFFIX, mask_for


class Stub:
    def __init__(self, names, boxes, label="stub"):
        self.names, self.boxes, self.label, self.asked = names, boxes, label, []

    def all_classes(self):
        return dict(self.names)

    def find_classes(self, arr, ids):
        self.asked.append(list(ids))
        return [b for b in self.boxes if b[5] in ids]


class Seg:
    def __init__(self, hits):
        self.hits, self.asked = hits, []

    def masks(self, image, prompts, threshold=0.5, refine=2, max_each=None):
        self.asked += list(prompts)
        return [{"prompt": p, "score": 0.9, "box": list(self.hits[p]), "mask": None} for p in prompts if p in self.hits]


IMG = Image.new("RGB", (1000, 800))
FDET = Stub({0: "face", 1: "pussy", 2: "anus"}, [(300, 300, 360, 360, 0.9, 1)], "fdetailer")
BOORU = Stub({7: "butt", 8: "ass", 1: "bust"}, [(0, 0, 1000, 790, 0.8, 8), (400, 400, 600, 600, 0.7, 7)], "booru_yolo")


def test_synonyms_route_words_to_classes():
    assert D.classes_for("ass", BOORU.all_classes()) == [7, 8]
    assert D.classes_for("Breasts", {2: "boob"}) == [2] and D.classes_for("hat", BOORU.all_classes()) == []


def test_chain_order_big_detector_box_trusted_and_named_in_the_log():
    log = []
    f = D.chain_finder(["pussy", "ass"], [("fdetailer", FDET), ("booru_yolo", BOORU)], log=log.append)
    hits = f.find(IMG)
    assert [(h.gender, tuple(h.bbox)) for h in hits] == [("pussy", (300, 300, 360, 360)), ("ass", (0, 0, 1000, 790)),
                                                         ("ass", (400, 400, 600, 600))]
    assert "pussy: fdetailer" in f.describe() and "ass: booru_yolo" in f.describe()
    text = "".join(log)
    assert "pussy: found by fdetailer (score 0.90" in text and "ass: found by booru_yolo (score 0.80, 99% of" in text


def test_a_big_detector_box_is_not_handed_to_sam():
    big = Stub({8: "ass"}, [(0, 0, 1000, 800, 0.9, 8)], "booru_yolo")
    made = []
    f = D.chain_finder(["ass"], [("booru_yolo", big)], seg_factory=lambda: made.append(1))
    hits = f.find(IMG)
    assert [(h.source, tuple(h.bbox)) for h in hits] == [("yolo", (0, 0, 1000, 800))] and made == []
    # Prep keeps it (only SAM's big boxes are ignored) ...
    p = prep.Prep(lambda _t: None, None, f)
    assert [h.source for h in p._sane(hits, IMG.size)] == ["yolo"]


def test_sams_oversized_box_still_ignored_and_said():
    log = []
    seg = Seg({"hat": (0, 0, 1000, 790)})
    f = D.chain_finder(["hat"], [], seg=seg, log=log.append)
    p = prep.Prep(log.append, None, f)
    import pytest
    with pytest.raises(prep._NothingToAdd):
        p._sane(f.find(IMG), IMG.size)
    text = "".join(log)
    assert "hat: found by SAM 3.1" in text and "hat: SAM 3.1's box is too big (99%), ignored" in text


def test_sam_not_loaded_when_a_model_finds_the_word():
    made = []
    f = D.chain_finder(["pussy"], [("fdetailer", FDET)], seg_factory=lambda: made.append(1))
    assert len(f.find(IMG)) == 1 and made == []


def test_unknown_word_goes_to_sam_and_yolo_beats_sam():
    seg = Seg({"hat": (50, 50, 150, 150), "pussy": (0, 0, 900, 700)})
    f = D.chain_finder(["pussy", "hat"], [("fdetailer", FDET)], seg=seg)
    hits = f.find(IMG)
    assert [(h.gender, h.source) for h in hits] == [("pussy", "yolo"), ("hat", "sam3")] and seg.asked == ["hat"]


def test_missing_models_skipped_and_said_once(monkeypatch):
    D._said.clear()
    log = []
    st = D.stages(log.append, FDET, loader=lambda k: None)
    D.stages(log.append, FDET, loader=lambda k: None)
    assert [s[0] for s in st] == ["fdetailer"]
    assert len(log) == 3 and all("isn't installed" in t for t in log)


def test_chris_scenario_in_a_prep_run(tmp_path):
    """A rear shot: the detector's ass box is 49% of the picture. With 20% padding, squared, the crop would be the whole
    picture - before, skipped; now the padding is left off and the crop (which still trims) is kept."""
    for i in range(6):
        Image.fromarray(np.random.default_rng(i).integers(0, 255, (800, 1000, 3), dtype=np.uint8)).save(
            tmp_path / f"p{i}.png")
    booru = Stub({8: "ass"}, [(150, 100, 850, 660, 0.8, 8)], "booru_yolo")
    log = []
    f = D.chain_finder(["pussy", "anus", "ass"], [("fdetailer", FDET), ("booru", booru)], fit_all=True, log=log.append)
    prep.Prep(log.append, None, f).run("Face Crop Only", str(tmp_path), 0.05, prep.OTHER_TARGET, 20.0, False)
    assert len([c for c in os.listdir(tmp_path / "crops") if c.endswith(".png")]) == 6
    assert not (tmp_path / "skipped").exists() and "padding left off" in "".join(log)


def test_a_detector_crop_that_is_the_whole_picture_is_still_skipped(tmp_path):
    Image.fromarray(np.zeros((800, 1000, 3), np.uint8)).save(tmp_path / "p.png")
    f = D.chain_finder(["pussy", "ass"], [("fdetailer", FDET), ("booru_yolo", BOORU)], fit_all=True)
    prep.Prep(lambda _t: None, None, f).run("Face Crop Only", str(tmp_path), 0.05, prep.OTHER_TARGET, 20.0, False)
    assert "cover nearly the whole picture" in (tmp_path / "skipped" / "skipped.txt").read_text()


def test_detector_model_takes_every_class():
    m = D._backend_class()("x.onnx")
    m.names = {0: "a", 1: "b"}
    assert m._pick_classes() is None


# ---- watermark masks ----
def test_watermark_mask_merged_beside_the_picture_and_found(tmp_path):
    p = tmp_path / "a.png"
    Image.new("RGB", (200, 100), "white").save(p)
    old = np.full((100, 200), 255, np.uint8)
    old[:, :20] = 0
    Image.fromarray(old, "L").save(tmp_path / f"a{MASK_SUFFIX}.png")
    wm = Stub({0: "watermark"}, [(150, 70, 190, 90, 0.5, 0), (10, 10, 30, 30, 0.2, 0)])
    share = D.watermark_mask(str(p), Image.open(p), [wm])
    out = tmp_path / f"a{MASK_SUFFIX}.png"
    m = np.asarray(Image.open(out))
    assert m[80, 170] == 0 and m[50, 5] == 0 and m[50, 100] == 255 and m[20, 20] == 255   # 0.2 score ignored
    assert m[65, 145] == 0                                           # padded 8 px
    assert 0 < share < 0.2 and mask_for(str(p)) == str(out)


def test_a_watermark_box_over_a_tenth_of_the_picture_is_dropped(tmp_path):
    """watermark_s fires on big plain areas (35-60% boxes at 0.3-0.7 on Chris's Puss set); real ones are small."""
    p = tmp_path / "a.png"
    Image.new("RGB", (600, 400), "white").save(p)
    wm = Stub({0: "watermark"}, [(0, 0, 500, 300, 0.9, 0), (10, 360, 130, 390, 0.4, 0)])
    boxes = D.watermark_boxes(np.asarray(Image.open(p)), [wm])
    assert boxes == [(2, 352, 138, 398)]


def test_watermarks_only_with_make_masks_never_in_a_prep_run(tmp_path):
    import inspect
    assert "watermarks" not in inspect.signature(prep.Prep.__init__).parameters
    from fizgig.web import prep_jobs
    assert "watermarks" in inspect.signature(prep_jobs.start_masks).parameters
    assert "watermarks" not in inspect.signature(prep_jobs.start).parameters
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (400, 600, 3), dtype=np.uint8)).save(tmp_path / "a.png")
    log = []
    wm = Stub({0: "watermark"}, [(10, 360, 130, 390, 0.9, 0)])
    found = D.watermark_folder(str(tmp_path), [wm], log=log.append)
    assert [n for n, _s in found] == ["a.png"] and (tmp_path / f"a{MASK_SUFFIX}.png").exists()
    assert "Watermarks masked: 1" in "".join(log) and not (tmp_path / "masks").exists()


def test_paths_saved_and_checked(tmp_path, monkeypatch):
    from fizgig.web import prep_jobs
    f = tmp_path / "m.onnx"
    f.write_bytes(b"x")
    assert prep_jobs.save_face_settings({"detectors": {"aio": str(f)}}) == {"detector_aio": str(f)}
    import pytest
    with pytest.raises(ValueError):
        prep_jobs.save_face_settings({"detectors": {"aio": str(tmp_path / "m.pt")}})
    assert D.model_path("aio", {"detector_aio": str(f)}) == str(f)
    assert D.model_path("aio", {}) == "" or os.path.isfile(D.model_path("aio", {}))


def test_page_texts_are_current():
    here = os.path.join(os.path.dirname(prep.__file__), "..", "web", "static")
    html = open(os.path.join(here, "index.html"), encoding="utf-8").read()
    js = open(os.path.join(here, "app.js"), encoding="utf-8").read()
    assert "nothing is\n                  cropped" not in html and "Body-part and watermark detectors" in html
    assert "Pictures where nothing is found are left whole" not in js and "copied to 'skipped'" in js
    assert 'id="prepWatermarks"' not in html and 'id="maskWatermarks"' in html      # in the Masks card now
    assert html.index('id="maskWatermarks"') > html.index('id="maskCard"')
    assert 'watermarks: $("maskWatermarks").checked' in js


def test_detectors_section_is_one_compact_table():
    here = os.path.join(os.path.dirname(prep.__file__), "..", "web", "static")
    js = open(os.path.join(here, "app.js"), encoding="utf-8").read()
    body = js[js.index("function detectorsShow("):js.index("async function faceSettingsSave(")]
    assert 'el("table", {class: "det-table"})' in body and "✓ Installed" in body and "– Missing" in body
    assert "Use files from somewhere else" in body and "Choose…" in body and "uploadButton" in body
    assert "Not installed: " not in body                     # no long note per model any more
    css = open(os.path.join(here, "style.css"), encoding="utf-8").read()
    rules = css[css.index(".det-table"):]
    assert "#" not in rules.split("/*")[0] or "var(--border" in rules     # theme colours only


def test_converter_message_is_not_face_only():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cv", os.path.join(os.path.dirname(prep.__file__), "..", "scripts",
                                                                     "convert_yolo_to_onnx.py"))
    cv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cv)
    aio = {0: "nipples", 1: "pussy", 2: "anus"}
    assert "no 'face'" not in cv.note(aio, face_model=False).lower() and "pussy" in cv.note(aio, face_model=False)
    assert "Body-part" in cv.note(aio) and "'face'" in cv.note({0: "face", 1: "penis"})
    tool = open(os.path.join(os.path.dirname(prep.__file__), "..", "..", "..", "tools", "convert_detector_to_onnx.py"),
                encoding="utf-8").read()
    assert '"--detector"' in tool
