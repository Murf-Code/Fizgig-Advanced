import pytest
from fizgig.training.replay_lr import reference_rate

@pytest.mark.parametrize("key", ["adamw","adamw8bit","pagedadamw8bit"])
def test_adamw_identity(key):
    assert reference_rate(key,.00025)==(.00025,.00025,1.)

@pytest.mark.parametrize("ref,native", [(.0001,.00018),(.0002,.00036),(.00025,.00045),(.0003,.00054)])
def test_muon_reference(ref,native):
    assert reference_rate("muon",ref)==(ref,native,1.8)

@pytest.mark.parametrize("rate", [0,-1,float("nan"),float("inf"),"invalid"])
def test_invalid_rate(rate):
    with pytest.raises(ValueError):
        reference_rate("muon",rate)

def test_unknown_optimizer():
    with pytest.raises(ValueError):
        reference_rate("polora",.00025)
