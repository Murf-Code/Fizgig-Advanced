import json
import math
from types import SimpleNamespace
import pytest
from fizgig.web import selective_optimizer_smoke as smoke


def test_fixed_queue(tmp_path):
    group = tmp_path / "new-group"
    cases = smoke.settings(group)
    assert [v["OPTIMIZER_TYPE"] for v in cases] == list(smoke.CASES)
    for v in cases:
        assert v["MAX_TRAIN_EPOCHS"] == "1"
        assert v["LEARNING_RATE"] == ".00025" or v["LEARNING_RATE"] == "0.00025"
        assert v["KREA2_SELECTIVE_COMPENSATION"] is True
        p=SimpleNamespace(problems=[],stages=[SimpleNamespace(name="Training",cmd=[])])
        smoke.attach(p,v,"krea2")
        assert p.stages[0].cmd == ["--family_option","selective_optimizer_smoke="+v["OPTIMIZER_TYPE"]]
        with pytest.raises(ValueError):
            smoke.attach(p,{**v,"MAX_TRAIN_EPOCHS":"2"},"krea2")
    assert smoke.initialize(group,"test dataset",cases)["status"] == "incomplete"
    rows=[{"update":i,"active_blocks":list(range(28 if i<=8 else 14)),
           "multiplier":1. if i<=8 else math.sqrt(2),"groups_restored":True} for i in range(1,14)]
    for name in smoke.CASES:
        dtypes={} if name=="adamw" else {"torch.uint8":448}
        result=smoke.finish(group/name,name,name,rows,[{"step":13,"loss":.18}],dtypes,13)
        assert result["status"]=="passed" and result["compensated_updates"]==5
    assert json.loads((group/smoke.REPORT).read_text())["status"]=="passed"


def test_no_reduced_mask_is_not_pass():
    rows=[{"update":i,"active_blocks":list(range(28)),"multiplier":1.,"groups_restored":True}
          for i in range(1,14)]
    assert smoke.assess("adamw",rows,[{"step":13,"loss":.18}],{})=="compensation_not_exercised"
    assert smoke.assess("adamw8bit",rows,[{"step":13,"loss":.18}],{})=="8bit_state_not_exercised"
