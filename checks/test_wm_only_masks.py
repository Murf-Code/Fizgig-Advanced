"""Make masks' Just hide the watermarks: all white with the watermark boxes black; no watermark -> no mask written;
masks already there kept unless Replace. Stub detector stands in for the ONNX ones."""
import numpy as np
from PIL import Image

from fizgig.dataset import detectors as D
from fizgig.dataset.masks import mask_for


class Stub:
    def __init__(self, boxes):
        self.boxes = boxes

    def all_classes(self):
        return {0: "watermark"}

    def find_classes(self, arr, ids):
        return self.boxes


def test_compose():
    m = D.watermarks_only((10, 20, 3), [(2, 3, 5, 6)])
    assert m.shape == (10, 20) and m.dtype == np.uint8
    assert (m[3:6, 2:5] == 0).all() and (m == 0).sum() == 9 and (m == 255).sum() == 191
    assert D.watermarks_only((10, 20), []) is None


def test_merge():
    base = np.full((10, 20), 255, np.uint8); base[:, :10] = 0
    m = D.watermarks_only((10, 20, 3), [(12, 0, 15, 2)], base)
    assert (m[:, :10] == 0).all() and (m[0:2, 12:15] == 0).all() and m[5, 18] == 255 and base[0, 12] == 255


def test_folder(tmp_path):
    for n in ("a", "b"):
        Image.new("RGB", (400, 400), "red").save(tmp_path / f"{n}.png")
    wm, none = [Stub([(10, 10, 40, 40, 0.9, 0)])], [Stub([])]
    q = dict(log=lambda m: None)
    c = D.watermarks_only_folder(str(tmp_path), wm, **q)
    assert c["made"] == 2
    m = np.asarray(Image.open(mask_for(str(tmp_path / "a.png"))))
    assert m[200, 200] == 255 and m[20, 20] == 0
    # a character mask already there: merged, then no watermark keeps it
    char = np.full((400, 400), 255, np.uint8); char[:, 300:] = 0
    Image.fromarray(char).save(mask_for(str(tmp_path / "a.png")))
    c = D.watermarks_only_folder(str(tmp_path), wm, **q)
    assert c["merged"] == 2
    m = np.asarray(Image.open(mask_for(str(tmp_path / "a.png"))))
    assert m[20, 20] == 0 and m[200, 350] == 0 and m[200, 200] == 255
    c = D.watermarks_only_folder(str(tmp_path), none, **q)
    assert c["kept"] == 2 and mask_for(str(tmp_path / "a.png"))
    # Replace: old ignored (fresh), then no watermark removes it
    c = D.watermarks_only_folder(str(tmp_path), wm, overwrite=True, **q)
    m = np.asarray(Image.open(mask_for(str(tmp_path / "a.png"))))
    assert c["made"] == 2 and m[200, 350] == 255
    c = D.watermarks_only_folder(str(tmp_path), none, overwrite=True, **q)
    assert c["removed"] == 2 and not mask_for(str(tmp_path / "a.png"))
    assert sorted(f.name for f in tmp_path.iterdir()) == ["a.png", "b.png"]
    c = D.watermarks_only_folder(str(tmp_path), none, overwrite=True, **q)
    assert c["no_person"] == 2


def test_trace():
    arr = np.zeros((200, 200, 3), np.uint8)
    box = (60, 80, 120, 100)                       # 60 x 20: padded 18 sideways, 12 up and down
    assert D.trace_box(arr, box, None).sum() == 60 * 20
    pb = D.padded_box(box, 200, 200)
    assert pb == (42, 68, 138, 112)
    full = np.zeros((200, 200), bool); full[68:112, 42:138] = True

    def stroke(crop):                              # a signature running past the YOLO box's right end
        t = np.zeros(crop.shape[:2], bool); t[18:24, 10:90] = True; return t
    m = D.trace_box(arr, box, stroke)
    assert 0 < m.sum() < full.sum() and m[90, 125] and not m[70, 60] and not (m & ~full).any()

    rng = np.random.default_rng(0)
    speckle = lambda c: rng.random(c.shape[:2]) > 0.93          # SAM picked up texture
    blob = lambda c: np.ones(c.shape[:2], bool)                 # SAM took the whole area
    def boom(c): raise RuntimeError
    for t in (lambda c: None, lambda c: np.zeros(c.shape[:2], bool), lambda c: np.ones((3, 3), bool), boom,
              speckle, blob):
        assert (D.trace_box(arr, box, t) == full).all()

    def stroke_specks(crop):                       # a good stroke with a few specks: specks dropped
        t = stroke(crop); t[2, 2] = t[40, 5] = t[3, 80] = True; return t
    m = D.trace_box(arr, box, stroke_specks)
    assert not m[68 + 2, 42 + 2] and m[90, 100]

    cov = D.watermark_cover(arr, [box], stroke)
    out = D.watermarks_only(arr.shape, [box], None, cov)
    assert (out == 0).sum() == cov.sum()


def test_merge_boxes():
    # stacked (two detectors), overlapping, and a signature found in two parts with a small gap -> one box
    assert D.merge_boxes([(10, 10, 50, 30), (12, 11, 49, 31)]) == [(10, 10, 50, 31)]
    assert sorted(D.merge_boxes([(10, 10, 50, 30), (55, 12, 90, 30), (300, 300, 320, 310)])) == \
        [(10, 10, 90, 30), (300, 300, 320, 310)]
    # chained: a joins b, then the union reaches c
    assert D.merge_boxes([(0, 0, 20, 20), (25, 0, 45, 20), (50, 0, 70, 20)]) == [(0, 0, 70, 20)]
    # too far apart: kept separate
    assert len(D.merge_boxes([(0, 0, 20, 20), (40, 0, 60, 20)])) == 2


def test_stop(tmp_path):
    from fizgig.dataset.masks import STOP
    for n in "abc":
        Image.new("RGB", (100, 100), "red").save(tmp_path / f"{n}.png")
    calls = []

    class Once(Stub):
        def find_classes(self, arr, ids):
            calls.append(1); STOP.set()           # Stop pressed while the first picture is checked
            return [(10, 10, 40, 40, 0.9, 0)]
    try:
        c = D.watermarks_only_folder(str(tmp_path), [Once([])], log=lambda m: None)
    finally:
        STOP.clear()
    assert c["stopped"] == (1, 3) and c["made"] == 1 and len(calls) == 1


def test_stroke_fallback():
    # SAM misses a faint grey signature on cream: the stroke trace finds it instead of the whole box
    arr = np.full((200, 400, 3), (245, 238, 220), np.uint8)
    arr[95:105, 120:280] = (200, 195, 185)          # one faint stroke of "text"
    box = (100, 80, 300, 120)
    m = D.trace_box(arr, box, lambda c: None)
    x1, y1, x2, y2 = D.padded_box(box, 400, 200)
    assert m[100, 200] and m.sum() < 0.6 * (x2 - x1) * (y2 - y1) and not m[y1 + 1, x1 + 1]
    # a flat crop: nothing to trace, the padded box
    flat = np.full((200, 400, 3), 128, np.uint8)
    assert D.trace_box(flat, box, lambda c: None).sum() == (x2 - x1) * (y2 - y1)


def test_wm_score():
    old = D.WM_SCORE
    try:
        D.set_wm_score("0.7"); assert D.WM_SCORE == 0.7
        D.set_wm_score(5); assert D.WM_SCORE == 0.9
        D.set_wm_score("x"); assert D.WM_SCORE == 0.9
    finally:
        D.WM_SCORE = old
