"""Regression checks for stuck Step / Steps left during slow Krea 2 Global Replay."""
from fizgig.training.progress import TrainingProgressTracker, parse_tqdm_progress_line


def test_slow_replay_step_without_numeric_tqdm_speed_still_advances():
    # Global Replay can be slow enough for tqdm to have no usable rate.
    t = TrainingProgressTracker(2)
    start = t.consume("[training-progress] epoch 1/2 start step 0/10, 0/5 in epoch")
    assert start == {"kind": "epoch_start", "epoch": 1, "total_epochs": 2,
                     "step": 0, "total_steps": 10, "epoch_step": 0, "steps_per_epoch": 5}
    update = t.consume("steps:  10%|██       | 1/10 [00:27<?, ?it/s, avr_loss=0.1245]")
    assert update["kind"] == "training" and update["step"] == 1
    assert update["epoch_step"] == 1 and update["steps_per_epoch"] == 5
    assert update["average_loss_text"] == "0.1245"
    assert update["speed_text"] == ""


def test_true_epoch_start_resets_steps_left_after_boundary():
    t = TrainingProgressTracker(2)
    t.consume("[training-progress] epoch 1/2 start step 0/10, 0/5 in epoch")
    full = t.consume("steps:  50%|█████     | 5/10 [01:00<01:00, 12.0s/it]")
    assert full["epoch_step"] == 5 and full["epoch"] == 1

    start = t.consume("[training-progress] epoch 2/2 start step 5/10, 0/5 in epoch")
    assert start["epoch_step"] == 0 and start["epoch"] == 2
    mid = t.consume("steps: 60%|██████    | 6/10 [01:10<00:40, 10.0s/it]")
    assert mid["epoch_step"] == 1 and mid["epoch"] == 2


def test_resume_uses_actual_step_offset_within_epoch():
    t = TrainingProgressTracker(4)
    start = t.consume("[training-progress] epoch 3/4 start step 12/20, 2/5 in epoch")
    assert start["epoch_step"] == 2 and start["steps_per_epoch"] == 5
    update = t.consume("steps: 65%|██████▌   | 13/20 [02:20<?, ?it/s]")
    assert update["epoch_step"] == 3 and update["epoch"] == 3


def test_legacy_tqdm_works_without_epoch_start():
    t = TrainingProgressTracker(4)
    p = t.consume("steps: 35%|███▌      | 7/20 [01:20<01:40, 12.50s/it, avr_loss=0.120]")
    assert p["step"] == 7 and p["epoch_step"] == 2 and p["steps_per_epoch"] == 5
    assert p["speed_text"] == "12.50 s/it" and p["average_loss_text"] == "0.120"


def test_missing_rate_does_not_accept_unrelated_progress_bars():
    assert parse_tqdm_progress_line("sampling: 5/8 [00:10<?, ?it/s]") is None
    assert parse_tqdm_progress_line("steps: 1/10 [?]")["step"] == 1
    assert parse_tqdm_progress_line("steps: 1/10 [00:10<?, ?it/s]")["step"] == 1
