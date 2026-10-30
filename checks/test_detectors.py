"""Image Prep's detector chain (Other…): each word goes to the first model that knows it (Fdetailer, booru_yolo,
All-in-One, assDetailer), SAM 3.1 last; a model with only oversized boxes hands the word on; a missing model is
skipped. Watermark masks go to masks/ and masked training finds them. Stub models stand in for the ONNX ones."""
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


def test_chain_order_and_oversized_box_handed_on():
    f = D.chain_finder(["pussy", "ass"], [("fdetailer", FDET), ("booru_yolo", BOORU)])
    hits = f.find(IMG)
    assert [(h.gender, tuple(h.bbox)) for h in hits] == [("pussy", (300, 300, 360, 360)), ("ass", (400, 400, 600, 600))]
    assert "pussy: fdetailer" in f.describe() and "ass: booru_yolo" in f.describe()


def test_only_oversized_goes_to_the_next_model_then_sam():
    big = Stub({8: "ass"}, [(0, 0, 1000, 800, 0.9, 8)], "booru_yolo")
    aio = Stub({3: "butt"}, [(10, 10, 990, 790, 0.9, 3)], "aio")       # too big as well
    seg = Seg({"ass": (200, 200, 400, 400)})
    f = D.chain_finder(["ass"], [("booru_yolo", big), ("aio", aio)], seg_factory=lambda: seg)
    hits = f.find(IMG)
    assert [(h.source, tuple(h.bbox)) for h in hits] == [("sam3", (200, 200, 400, 400))] and seg.asked == ["ass"]


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
    for i in range(6):
        Image.fromarray(np.random.default_rng(i).integers(0, 255, (800, 1000, 3), dtype=np.uint8)).save(
            tmp_path / f"p{i}.png")
    f = D.chain_finder(["pussy", "anus", "ass"], [("fdetailer", FDET), ("booru_yolo", BOORU)], fit_all=True)
    prep.Prep(lambda _t: None, None, f).run("Face Crop Only", str(tmp_path), 0.05, prep.OTHER_TARGET, 20.0, False)
    assert len([c for c in os.listdir(tmp_path / "crops") if c.endswith(".png")]) == 6
    assert not (tmp_path / "skipped").exists()


def test_detector_model_takes_every_class():
    m = D._backend_class()("x.onnx")
    m.names = {0: "a", 1: "b"}
    assert m._pick_classes() is None


# ---- watermark masks ----
def test_watermark_mask_merged_into_masks_folder_and_found(tmp_path):
    p = tmp_path / "a.png"
    Image.new("RGB", (200, 100), "white").save(p)
    old = np.full((100, 200), 255, np.uint8)
    old[:, :20] = 0
    Image.fromarray(old, "L").save(tmp_path / f"a{MASK_SUFFIX}.png")
    wm = Stub({0: "watermark"}, [(150, 70, 190, 90, 0.5, 0), (10, 10, 30, 30, 0.2, 0)])
    share = D.watermark_mask(str(p), Image.open(p), [wm])
    out = tmp_path / "masks" / f"a{MASK_SUFFIX}.png"
    m = np.asarray(Image.open(out))
    assert m[80, 170] == 0 and m[50, 5] == 0 and m[50, 100] == 255 and m[20, 20] == 255   # 0.2 score ignored
    assert m[65, 145] == 0                                           # padded 8 px
    assert 0 < share < 0.2 and mask_for(str(p)) == str(out)


def test_prep_masks_watermarks_and_the_resized_copy_gets_it(tmp_path):
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (400, 600, 3), dtype=np.uint8)).save(tmp_path / "a.jpg")
    wm = Stub({0: "watermark"}, [(0, 0, 500, 300, 0.9, 0)])
    log = []
    prep.Prep(log.append, watermarks=[wm]).run("Resize Only", str(tmp_path), 0.05)
    text = "".join(log)
    assert "Watermarks masked: 1" in text and "worth a look: a.jpg" in text
    assert (tmp_path / "resized" / f"a{MASK_SUFFIX}.png").exists()


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
    assert 'id="prepWatermarks"' in html and "$(\"prepWatermarks\").checked = !!saved.watermarks" in js
