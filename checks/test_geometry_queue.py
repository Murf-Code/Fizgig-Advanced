"""Fixed geometry queue controls and report lifecycle (no GPU)."""
from pathlib import Path
import json
import zipfile
from fizgig.web import geometry_benchmark as bench, geometry_report as report


def test_fixed_pilot(tmp_path):
    configs=bench.settings("pilot", str(tmp_path))
    assert len(configs)==5
    assert len({v["LORA_OUTPUT_DIR"] for v in configs})==5
    assert {v["OPTIMIZER_TYPE"] for v in configs}=={"adamw8bit","muon","polora","lora_tsd"}
    for v in configs:
        assert v["NETWORK_DIM"]==v["NETWORK_ALPHA"]=="64"
        assert v["MAX_TRAIN_EPOCHS"]=="5"
        assert v["FAMILY_EMA"]=="Off" and v["KREA2_TRAINING_METHOD"]=="Standard"
        assert v["VALIDATION_HOLDOUT"] and v["SAMPLE_AT_FIRST"]
        assert not v["ADAPTIVE_LR"] and not v["KREA2_SELECTIVE_BLOCK_CHECK"]
    root=tmp_path/"pilot"
    report.create(root,"dataset",configs)
    for i,v in enumerate(configs):
        out=Path(v["LORA_OUTPUT_DIR"])
        out.mkdir()
        log=out/"run.log"
        log.write_text("[validation] step 0 (before): loss 1.0\n[validation] step 5 (epoch): loss 0.5\n",encoding="utf-8")
        report.update(out,v["__geometry_compare"],"finished" if i!=3 else "failed",log_path=log)
    data=json.loads((root/report.REPORT).read_text())
    assert data["status"]=="partial"
    assert len(data["runs"])==5
    assert data["runs"][0]["validation"][-1]["loss"]==0.5
    with zipfile.ZipFile(root/report.BUNDLE) as z:
        assert report.REPORT in z.namelist() and report.COMBINED in z.namelist()
