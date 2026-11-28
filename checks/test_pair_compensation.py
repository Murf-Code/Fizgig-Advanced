import pytest
from fizgig.training.pair_compensation import pair_multiplier, pair_step


def test_multiplier_bounds():
    for name, cap in (("polora", 1.5), ("lora_tsd", 1.25)):
        assert pair_multiplier(name,28,28,enabled=True)==1.
        assert pair_multiplier(name,1,28,enabled=True)==cap
        assert pair_multiplier(name,1,28,enabled=False)==1.
        assert pair_multiplier(name,27,28,enabled=True)==pytest.approx((28/27)**.5)
    with pytest.raises(ValueError):
        pair_multiplier("lora_tsd",0,28,enabled=True)


class Fake:
    magnitude_scale = 3.
    def __init__(self, fail=False):
        self.param_groups = [{"lr":.005,"max_delta_norm":.1},{"lr":.004,"max_delta_norm":.1}]
        self.fail=fail
    def step(self):
        self.seen=[g.copy() for g in self.param_groups]
        if self.fail:
            raise RuntimeError("test failure")


def test_native_rate_restored():
    opt=Fake()
    report=pair_step(opt,"lora_tsd",1.25)
    assert [g["lr"] for g in opt.seen]==pytest.approx([.00625,.005])
    assert [g["lr"] for g in opt.param_groups]==[.005,.004]
    assert all(g["max_delta_norm"]==.1 for g in opt.seen+opt.param_groups)
    assert opt.magnitude_scale==3.
    assert report["pair_compensation_lr_restored"]
    assert report["pair_compensation_applied_lrs"]==pytest.approx([.00625,.005])


def test_restore_on_error():
    opt=Fake(fail=True)
    with pytest.raises(RuntimeError, match="test failure"):
        pair_step(opt,"lora_tsd",1.25)
    assert [g["lr"] for g in opt.param_groups]==[.005,.004]


def test_reject_out_of_range():
    opt=Fake()
    with pytest.raises(ValueError):
        pair_step(opt,"lora_tsd",1.5)
    assert not hasattr(opt,"seen")
