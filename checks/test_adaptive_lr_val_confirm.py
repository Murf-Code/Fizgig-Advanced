"""Chris's PoLoRA run (Nov 2026, base LR 0.02): with validation on, train loss now confirms it - Auto LR only raises
while validation improves and the epoch train loss isn't rising 1%+, and holds when only validation looks better."""
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


def test_chris_run_raises_only_when_train_loss_agrees():
    _a, lrs = _replay()
    print(" ".join(f"{x:.4f}" for x in lrs))
    for i in range(1, len(lrs)):
        if lrs[i] > lrs[i - 1]:
            assert VAL[i + 1] < VAL[i] and LOSS[i] <= LOSS[i - 1] * 1.01
    assert max(lrs) < 0.039        # the old run reached 0.039 by e9


def test_old_state_without_prev_raw_loads_and_new_keys_are_ignored():
    a = AdaptiveLR(1e-4, 1)
    a.load_state_dict({"best_loss": .1, "good_streak": 1, "best_val": .14, "cooldown": 0,
                       "base_lr": .02, "raised_from": .02, "prev_val": .14})
    assert a.prev_raw is None
    _replay(a)
