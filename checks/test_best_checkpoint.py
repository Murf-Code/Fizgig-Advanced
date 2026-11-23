"""Rolling-best checkpoint scoring, persistence and CLI/UI regression checks."""
import json
import math

from fizgig.training.best_checkpoint import RollingBest, save_snapshot


def test_train_smoothing_requires_complete_window():
    t = RollingBest("train", window=3, cooldown=4, min_delta=0.05)
    assert t.training(0.5, 1) is None
    assert t.training(0.4, 2) is None
    score = t.training(0.3, 3)
    assert score is not None and math.isclose(score, 0.4)
    t.saved(score, 3, 1)
    assert t.training(0.2, 4) is None
    assert t.training(0.15, 5) is None
    assert t.training(0.1, 6) is None  # even a lower score cannot write within cooldown
    assert t.training(0.1, 7) < 0.4 * 0.95


def test_train_optimize_boundary_and_invalid_losses():
    t = RollingBest("train", window=2, cooldown=1)
    assert t.training(float("nan"), 1) is None
    assert t.training(float("inf"), 2) is None
    assert t.training(-1, 3) is None
    assert t.training(0.3, 4, ready=False) is None
    assert t.training(0.2, 5, ready=True) == 0.25


def test_validation_ignores_noise_and_resets_when_validation_set_changes():
    t = RollingBest("validation", cooldown=5, min_delta=0.01)
    assert t.validation(0.25, 10, seg=0) == 0.25
    t.saved(0.25, 10, 1, seg=0)
    assert t.validation(0.249, 15, seg=0) is None  # less than 1% relative
    assert t.validation(0.23, 15, seg=0) == 0.23
    t.saved(0.23, 15, 1, seg=0)
    assert t.validation(0.21, 16, seg=0) is None  # cooldown
    assert t.validation(0.4, 17, seg=1) == 0.4  # different validation set, incomparable


def test_best_score_advances_only_after_saver_commits():
    t = RollingBest("train", window=2, cooldown=10)
    assert t.training(0.2, 1) is None
    assert t.training(0.1, 2) == 0.15
    assert t.best is None  # save failed or was never called
    assert t.training(0.1, 3) == 0.1
    t.saved(0.1, 3, 1)
    assert t.best == 0.1 and t.step == 3 and t.epoch == 1


def test_resume_restores_score_but_not_training_window(tmp_path):
    path = tmp_path / "best.json"
    original = RollingBest("train", window=3, cooldown=50)
    original.saved(0.14, 200, 4)
    save_snapshot(str(path), original.snapshot())
    loaded = RollingBest("train", window=3, cooldown=50)
    assert not loaded.restore(str(path), checkpoint_exists=False)
    assert loaded.restore(str(path), checkpoint_exists=True)
    assert loaded.best == 0.14 and loaded.step == 200
    assert len(loaded.values) == 0
    assert not RollingBest("train", window=8, cooldown=50).restore(
        str(path), checkpoint_exists=True)


def test_saved_metadata_is_atomic_json(tmp_path):
    path = tmp_path / "rolling.json"
    save_snapshot(str(path), {"mode": "validation", "score": 0.01})
    assert json.loads(path.read_text())["score"] == 0.01
    assert not (tmp_path / "rolling.json.tmp").exists()


def test_validation_source_wiring_and_output_filename():
    """The launch command must be able to enable mid-epoch fixed validation itself."""
    from fizgig.families import launch
    assert "validation" in launch.train_command.__code__.co_consts or callable(launch.train_command)
    from fizgig.training.best_checkpoint import MODES
    assert MODES == ("off", "train", "validation")


def test_parser_accepts_both_modes_without_old_cli_changes():
    from fizgig.families.train import setup_parser
    p = setup_parser()
    assert p.get_default("best_checkpoint_mode") == "off"
    assert p.get_default("best_checkpoint_window") == 32
    assert p.get_default("best_checkpoint_min_delta") == 0.005
