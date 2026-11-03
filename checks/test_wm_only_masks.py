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
