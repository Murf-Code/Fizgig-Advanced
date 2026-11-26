import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from fizgig.training.replay_forks import Pair, MovementCurve
from fizgig.training.replay_history import core, evaluate_history
from fizgig.training.replay_diagnostics import digest
from fizgig.training.replay_evaluator import Case


def test_historical_curve_matches_dense_effective_weight():
    g = torch.Generator().manual_seed(5)
    a,b,c,d = [torch.randn(3,3,generator=g) for _ in range(4)]
    pair = Pair("block",0,a,b,.75)
    curve = MovementCurve([core(pair,(a,b),(c,d))],{0:1.})
    for t in (0.,.25,.5,1.):
        aa,bb = torch.lerp(a.double(),c.double(),t),torch.lerp(b.double(),d.double(),t)
        want = float(torch.linalg.vector_norm(bb@aa-b.double()@a.double()))*.75
        assert curve.norm(t) == pytest.approx(want,abs=1e-12)


def test_history_all_arms_and_restoration():
    # Small CPU tensors exercise real geometry, candidate matching and report
    # wiring. Mocked predictions do not establish CUDA/streaming parity.
    pairs = [Pair(str(b),b,torch.ones(1,1),torch.tensor([[float(b+1)]]),1.)
             for b in range(28)]
    initial = {p.name:(p.A.clone(),p.B.clone()) for p in pairs}
    factors = {p.name:(p.A.clone(),p.B.clone()*.75,p.block,p.scale) for p in pairs}
    history = {"step":3,"factors":factors,"hash":digest(factors)}
    inputs = {"x":torch.ones(1)}
    panel = [(Case("target","unit","original",.5,1,digest(inputs)),inputs,torch.zeros(1))]
    report = {"step":12}
    net = SimpleNamespace(wrapped={})
    def predict(_):
        return torch.tensor([sum(float(p.B@p.A) for p in pairs)])
    evaluate_history(pairs=pairs,net=net,history=history,panel=panel,predict=predict,
                     guard=lambda f:f(),report=report,save=lambda:None,
                     precision_context=lambda *args:nullcontext())
    assert report["integrity_passed"]
    assert len(report["geometry"]) == 38
    assert len(report["matching"]) == 6
    assert all(m["matched"] for m in report["matching"].values())
    assert all(len(v["restored"]) == 38 for v in report["evaluators"].values())
    assert digest({p.name:(p.A,p.B) for p in pairs}) == digest(initial)
    assert report["evaluators"]["native"]["arms"]["block26"]["rows"][0]["target_gain"] > 0
