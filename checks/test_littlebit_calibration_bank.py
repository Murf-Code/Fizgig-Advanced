from pathlib import Path
import sys
import pytest
import torch
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts/research/krea2_littlebit"))
sys.path.insert(0,str(ROOT/"src"))
from calibration_bank import CaptureBank, COUNT, partition
from fizgig.web import littlebit_benchmark as bench


def test_split_is_disjoint_and_stable():
    records=[{"step":i+1} for i in range(COUNT)]
    steps=[{"step":i+1,"items":[f"image-{i}"]} for i in range(COUNT)]
    split=partition(records,steps)
    assert [len(split[k]) for k in ("fit","development","confirmation")]==[64,8,8]
    assert len(set(sum(split.values(),[])))==80
    assert partition(records[::-1],steps[::-1])==split
    steps[-1]["items"]=steps[0]["items"]
    with pytest.raises(ValueError,match="distinct"):
        partition(records,steps)


def test_capture_only_queue_does_not_change_smoke(tmp_path):
    normal=bench.settings(tmp_path)
    bank=bench.settings(tmp_path,capture_bank=True)
    assert [v["__littlebit_case"] for v in normal]==list(bench.SMOKE_CASES)
    assert all("__littlebit_capture_bank" not in v for v in normal)
    assert len(bank)==1 and bank[0]["__littlebit_case"]=="stock"
    assert bank[0]["__littlebit_capture_bank"] is True
    assert all(bank[0][k]==v for k,v in bench.COMMON.items())


def test_capture_bank_stops_only_after_requested_unique_ids(tmp_path):
    bank=CaptureBank()
    bank.out=tmp_path
    gen=torch.Generator().manual_seed(42)
    x=torch.zeros(1)
    for i in range(2):
        bank.begin({"item_keys":["same"]},x,{},gen)
        if i==0:
            assert bank.active
            bank.seen.add("same")
        else:
            assert not bank.active
        bank.active=False
        assert not bank.end({"t":.5},x)
    with pytest.raises(RuntimeError,match="distinct"):
        bank.finish()
