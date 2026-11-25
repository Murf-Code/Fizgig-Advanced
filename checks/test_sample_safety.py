"""Samples-only local classifier regression (no actual explicit imagery required)."""
from PIL import Image
from fizgig.web import sample_safety as safety


class FakeModel:
    def __init__(self, hit=False):
        self.hit = hit
    def all_classes(self):
        return {0:"penis",1:"pussy",2:"nipples",3:"boob"}
    def find_classes(self, rgb, ids):
        assert rgb.shape[-1] == 3
        return [(0,0,1,1,.8,0)] if self.hit else []


class FakeDetectors:
    def __init__(self, hit=False):
        self.hit = hit
    def load(self, key, **kw):
        return FakeModel(self.hit)
    def classes_for(self, word, names):
        return [k for k,v in names.items() if v == word]


def test_missing_or_partial_model_stays_obscured(tmp_path):
    p = tmp_path / "sample.png"
    Image.new("RGB",(10,10)).save(p)
    assert safety._classify(p,FakeDetectors(),[])["state"] == "unknown"
    assert safety._classify(p,FakeDetectors(),[("booru","x",0,0)])["state"] == "unknown"


def test_detection_result(tmp_path):
    p = tmp_path / "sample.png"
    Image.new("RGB",(10,10)).save(p)
    models = [("aio","x",0,0)]
    assert safety._classify(p,FakeDetectors(),models)["state"] == "clear"
    assert safety._classify(p,FakeDetectors(True),models)["state"] == "flagged"


def test_off_by_default():
    from fizgig.web import server
    assert server.PREF_DEFAULTS["blur_nsfw_samples"] == "0"
    assert "blur_nsfw_samples" in server.SETTING_PREFS
