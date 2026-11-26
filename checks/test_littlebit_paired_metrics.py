import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research/krea2_littlebit"))
from paired_metrics import paired
import pytest


def scores(a,b,c):
    return {name:{"per_image":{str(i):[v,v] for i,v in enumerate(vals)}} for name,vals in
            zip(("no_adapter","stock_trained","compressed_trained"),(a,b,c))}


def test_paired_positive():
    x=paired(scores([4]*5,[3]*5,[3.1]*5))
    assert x["retention_descriptive"]==pytest.approx(.9)
    assert x["retention_interpretable"]
    assert x["gain_difference_interval"]==pytest.approx([-.1,-.1])


def test_unstable_and_small():
    assert not paired(scores([4]*4,[3,5,3,5],[3]*4))["retention_interpretable"]
    assert paired(scores([4],[3],[3]))["retention_interval"] is None


def test_mismatched():
    with pytest.raises(ValueError):
        paired(scores([4]*5,[3]*4,[3]*5))
