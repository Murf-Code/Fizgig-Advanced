"""Chris's PoLoRA run (Nov 2026, base LR 0.02): Auto LR walked 0.02 -> 0.039 and validation stalled e9-e11. Raises are
now capped at 1.5x the start, and a raise that makes validation and train loss both worse is undone at once."""
import torch

from fizgig.training.adaptive_lr import AdaptiveLR

VAL = [.1516, .1503, .1493, .1485, .1478, .1477, .1479, .1472, .1462, .1468, .1466, .1470, .1456]  # [0] = step 0
LOSS = [.1167, .1193, .117, .1272, .1267, .1287, .1248, .1157, .1209, .132, .128, .1132]
START = 0.02


def _replay(a=None, val=VAL, loss=LOSS):
    net = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(net.parameters(), lr=START)
    a = a or AdaptiveLR(START * 0.05, START * 10)
    if val:
        a.set_baseline(val[0])
    lrs = []
    for e, t in enumerate(loss):
        a.epoch_boundary(e, t, net, opt, val_loss=val[e + 1] if val else None)
        lrs.append(opt.param_groups[0]["lr"])
    return a, lrs


def test_chris_run_never_passes_the_cap_and_backs_off_after_a_bad_raise():
    _a, lrs = _replay()
    print(" ".join(f"{x:.4f}" for x in lrs))
    assert max(lrs) <= 0.03 + 1e-12
    for i in range(1, len(lrs) - 1):
        if lrs[i] > lrs[i - 1]:       # a raise at epoch i: judged at i+1
            v_worse = VAL[i + 2] > VAL[i + 1]
            t_up = LOSS[i + 1] > LOSS[i] * 1.01
            if v_worse and t_up:
                assert lrs[i + 1] < lrs[i]


def test_cap_applies_without_validation():
    _a, lrs = _replay(val=None, loss=[.2 - .01 * i for i in range(14)])
    assert max(lrs) <= 0.03 + 1e-12 and max(lrs) > START


def test_old_state_without_new_keys_loads():
    a = AdaptiveLR(1e-4, 1)
    a.load_state_dict({"best_loss": .1, "good_streak": 1, "best_val": .14, "cooldown": 0})
    assert a.base_lr is None and a.raised_from is None
    _replay(a)
