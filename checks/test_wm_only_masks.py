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


def test_folder(tmp_path):
    for n in ("a", "b"):
        Image.new("RGB", (400, 400), "red").save(tmp_path / f"{n}.png")
    c = D.watermarks_only_folder(str(tmp_path), [Stub([(10, 10, 40, 40, 0.9, 0)])], log=lambda m: None)
    assert c["made"] == 2 and mask_for(str(tmp_path / "a.png"))
    m = np.asarray(Image.open(mask_for(str(tmp_path / "a.png"))))
    assert m[200, 200] == 255 and m[20, 20] == 0
    c = D.watermarks_only_folder(str(tmp_path), [Stub([])], log=lambda m: None)
    assert c["kept"] == 2 and c["made"] == 0
    c = D.watermarks_only_folder(str(tmp_path), [Stub([])], overwrite=True, log=lambda m: None)
    assert c["removed"] == 2 and c["made"] == 0 and not mask_for(str(tmp_path / "a.png"))
    assert sorted(f.name for f in tmp_path.iterdir()) == ["a.png", "b.png"]    # only the masks went
    c = D.watermarks_only_folder(str(tmp_path), [Stub([])], overwrite=True, log=lambda m: None)
    assert c["no_person"] == 2 and c["removed"] == 0
