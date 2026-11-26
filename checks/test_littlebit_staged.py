import sys
from pathlib import Path
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"scripts/research/krea2_littlebit"))
from mixed_selector import choose
from research_trace import fingerprint
from task_diagnostics import movement,vector_compare
from wddm_memory import counters,values


def test_mixed_budget_and_pareto():
    stock=lambda: {"codec":"int8","bytes":100,"error":0.}
    x=choose({"a":[stock(),{"codec":"q6","bytes":80,"error":.001},
                  {"codec":"q4","bytes":50,"error":.1}],
              "b":[stock(),{"codec":"q6","bytes":80,"error":.002},
                  {"codec":"q4","bytes":50,"error":.003}]},.70)
    assert x["budget_met"]
    assert x["chosen_bytes"]==130
    assert x["families"]["a"]["codec"]=="q6"
    assert x["families"]["b"]["codec"]=="q4"


def test_mixed_unreachable():
    x=choose({"a":[{"codec":"int8","bytes":100,"error":0.}]},.7)
    assert not x["budget_met"] and x["chosen_bytes"]==100


def test_trace_fingerprint():
    a=torch.arange(8,dtype=torch.bfloat16)
    assert fingerprint({"a":a,"b":[1,2]})==fingerprint({"b":[1,2],"a":a.clone()})
    assert fingerprint(a)!=fingerprint(a.reshape(2,4))
    assert fingerprint(a)!=fingerprint(a+1)


def test_effective_update():
    torch.manual_seed(4)
    a,b=torch.randn(3,7),torch.randn(9,3)
    c,d=a+0.1,b-0.2
    result=movement({"x":("lora",a,b,0.5)},{"x":("lora",c,d,0.6)})["x"]
    x=b@a*.5; y=d@c*.6
    assert result["stock_norm"]==pytest.approx(float(x.norm()),rel=1e-5)
    assert result["candidate_norm"]==pytest.approx(float(y.norm()),rel=1e-5)
    assert result["difference_norm"]==pytest.approx(float((x-y).norm()),rel=1e-5)
    assert result["effective_update_cosine"]==pytest.approx(vector_compare(x,y)["cosine"],rel=1e-5)


def test_wddm_parser():
    h=["timestamp",r"\\host\GPU Process Memory(pid_42_luid_a_phys_0)\Dedicated Usage",
       r"\\host\GPU Process Memory(pid_42_luid_a_phys_0)\Shared Usage",
       r"\\host\GPU Process Memory(pid_142_luid_a_phys_0)\Shared Usage",
       r"\\host\GPU Process Memory(pid_42_luid_b_phys_1)\Shared Usage"]
    assert values(["time","100","200","900","50"],counters(h,42))=={"dedicated":100,"shared":250}
    assert values(["time","bad","bad","900","bad"],counters(h,42))=={}
