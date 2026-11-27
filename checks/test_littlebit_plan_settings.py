"""Regression for float alpha emitted by the normal native launch planner."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fizgig.web.littlebit_benchmark import require_resolved


@pytest.mark.parametrize("value", ["32", "32.0", "3.2e1"])
def test_equivalent_alpha(value):
    require_resolved(["--network_alpha",value],"--network_alpha","32")


@pytest.mark.parametrize("value", ["16","32.01","nan","inf","wrong"])
def test_wrong_alpha(value):
    with pytest.raises(ValueError,match="Unexpected resolved trainer setting"):
        require_resolved(["--network_alpha",value],"--network_alpha","32")


@pytest.mark.parametrize("flag,value", [("--network_dim","32.0"),("--optimizer_type","adamw"),("--seed","43")])
def test_other_settings_stay_strict(flag,value):
    with pytest.raises(ValueError,match="Unexpected resolved trainer setting"):
        require_resolved([flag,value],flag,{"--network_dim":"32","--optimizer_type":"muon","--seed":"42"}[flag])


def test_missing_and_duplicate_alpha():
    for cmd in ([],["--network_alpha","32","--network_alpha","32"]):
        with pytest.raises(ValueError,match="Missing or repeated"):
            require_resolved(cmd,"--network_alpha","32")
