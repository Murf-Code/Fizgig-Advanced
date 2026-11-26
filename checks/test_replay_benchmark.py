"""Replay comparison queue recipe and launch regression checks."""
from types import SimpleNamespace

from fizgig.web import replay_benchmark as rb


def test_recipe_is_eight_independent_runs(tmp_path):
    runs = rb.settings("test", str(tmp_path), 3, "a test prompt")
    assert [v["__replay_compare"] for v in runs] == [c[0] for c in rb.CASES]
    assert len({v["LORA_NAME"] for v in runs}) == len({v["LORA_OUTPUT_DIR"] for v in runs}) == 8
    assert [v["KREA2_SELECTIVE_BLOCK_CHECK"] for v in runs] == [False] * 6 + [True, True]
    assert [v["KREA2_TRAINING_METHOD"] for v in runs] == ["Global replay"] * 8
    assert [v["LEARNING_RATE"] for v in runs] == ["0.00020", "0.00025", "0.000264", "0.00030", "0.00035", "0.00045", "0.00045", "0.00045"]
    varying = {"__replay_compare", "LORA_NAME", "LORA_OUTPUT_DIR", "KREA2_TRAINING_METHOD", "KREA2_SELECTIVE_BLOCK_CHECK", "LEARNING_RATE"}
    assert all({k: v for k, v in row.items() if k not in varying} ==
               {k: v for k, v in runs[0].items() if k not in varying} for row in runs)
    assert all(v["VALIDATION_HOLDOUT"] and v["SAMPLE_ENABLED"] for v in runs)
    assert all(v["MAX_TRAIN_EPOCHS"] == "5" for v in runs)
    assert [v["__replay_compare"] for v in runs] == [
        "calibration-unrestricted-lr200", "calibration-unrestricted-lr250",
        "calibration-unrestricted-lr264", "calibration-unrestricted-lr300",
        "calibration-unrestricted-lr350", "calibration-unrestricted-lr450",
        "calibration-random-lr450", "calibration-gradient-lr450"]
    assert all(v["SEED"] == "42" and v["VALIDATION_SEED"] == "42" for v in runs)


def test_comparison_preserves_normal_adapter_targets(tmp_path):
    desc = SimpleNamespace(key="krea2", n_blocks=28)
    for values in rb.settings("test", str(tmp_path)):
        cmd = ["python", "train.py", "--sample_prompts", "prompts.txt"]
        plan = SimpleNamespace(problems=[], stages=[SimpleNamespace(name="Training", cmd=cmd)],
                               dataset_toml='holdout_file = "held.json"', console=[])
        rb.adjust_plan(desc, values, plan)
        assert not plan.problems
        expected = ["python", "train.py", "--sample_prompts", "prompts.txt"]
        if values["KREA2_SELECTIVE_BLOCK_CHECK"]:
            expected += ["--family_option", "selective_strategy=" + rb.case_strategy(values["__replay_compare"])]
        if rb.case_strategy(values["__replay_compare"]) == "random":
            expected += ["--family_option", "selective_budget=14"]
        expected += ["--family_option", "replay_research=calibration", "--family_option",
                     f"replay_calibration_lr={values['LEARNING_RATE']}"]
        assert cmd == expected


def test_no_holdout_is_rejected(tmp_path):
    desc = SimpleNamespace(key="krea2", n_blocks=28)
    values = rb.settings("test", str(tmp_path))[0]
    plan = SimpleNamespace(problems=[], stages=[SimpleNamespace(name="Training", cmd=[])],
                           dataset_toml="", console=[])
    rb.adjust_plan(desc, values, plan)
    assert plan.problems


def test_samples_on_with_blank_prompt(tmp_path):
    runs = rb.settings("test", str(tmp_path))
    for v in runs:
        assert v["SAMPLE_ENABLED"] and v["SAMPLE_AT_FIRST"]
        assert v["SAMPLE_EVERY_N_EPOCHS"] == "1"
        assert v["SAMPLE_SEED"] == "1234"
        assert v["SAMPLE_PROMPT"] == rb.DEFAULT_PREVIEW_PROMPTS
        assert len(v["SAMPLE_PROMPT"].splitlines()) == 4
        assert (v["SAMPLE_WIDTH"], v["SAMPLE_HEIGHT"]) == ("512", "512")


def test_fixed_prompts_ignore_stale_browser_input(tmp_path):
    runs = rb.settings("test", str(tmp_path), 3, "watermark")
    assert all(v["SAMPLE_PROMPT"] == rb.DEFAULT_PREVIEW_PROMPTS for v in runs)
    prompts = rb.DEFAULT_PREVIEW_PROMPTS.splitlines()
    assert len(prompts) == 4
    assert all("watermark" not in p.lower() for p in prompts)
    assert all("Ferwildir" in p for p in prompts[:2])
    assert all("Ferwildir" not in p for p in prompts[2:])


def test_missing_sample_flags_rejected(tmp_path):
    desc = SimpleNamespace(key="krea2", n_blocks=28)
    values = rb.settings("test", str(tmp_path))[0]
    plan = SimpleNamespace(problems=[], stages=[SimpleNamespace(name="Training", cmd=[])],
                           dataset_toml='holdout_file = "held.json"', console=[])
    rb.adjust_plan(desc, values, plan)
    assert any("sample prompt" in p for p in plan.problems)


def test_previous_controls_remain_launchable_without_requeuing():
    desc = SimpleNamespace(key="krea2", n_blocks=28)
    for case in ("replay", "replay-lr450", "standard", "standard-lr450",
                 "high_gradient-lr450", "random-lr450", "uphill-lr450", "static_early-lr450"):
        cmd = ["python", "train.py", "--sample_prompts", "prompts.txt"]
        plan = SimpleNamespace(problems=[], stages=[SimpleNamespace(name="Training", cmd=cmd)],
                               dataset_toml='holdout_file = "held.json"', console=[])
        rb.adjust_plan(desc, {"__replay_compare": case, "SAMPLE_ENABLED": True}, plan)
        assert not plan.problems


def test_calibration_controls_keep_all_blocks_until_selective_arms(tmp_path):
    runs = rb.settings("test", str(tmp_path))
    assert all(not v["KREA2_SELECTIVE_BLOCK_CHECK"] for v in runs[:6])
    assert all(rb.case_strategy(v["__replay_compare"]) == "replay" for v in runs[:6])
    assert [rb.case_strategy(v["__replay_compare"]) for v in runs[6:]] == ["random", "adaptive_gradient"]
    # Historical 5B queues retain their exact route and decay mappings.
    assert rb.CAUSAL_DECAYS["causal-decay10-lr450"] == 10.0
    assert rb.CAUSAL_DECAYS["causal-decay30-lr450"] == 30.0


def test_movement_pilot_single_run(tmp_path):
    runs = rb.pilot_settings("pilot", str(tmp_path))
    assert len(runs) == 1
    v = runs[0]
    assert v["__replay_compare"] == rb.MOVEMENT_PILOT_CASE
    assert v["MAX_TRAIN_EPOCHS"] == "1" and v["LEARNING_RATE"] == "0.00045"
    assert v["VALIDATION_HOLDOUT"] and not v["SAMPLE_ENABLED"]
    assert not v["KREA2_SELECTIVE_BLOCK_CHECK"]
    cmd = ["python", "train.py"]
    plan = SimpleNamespace(problems=[], stages=[SimpleNamespace(name="Training", cmd=cmd)],
                           dataset_toml='holdout_file = "held.json"', console=[])
    rb.adjust_plan(SimpleNamespace(key="krea2", n_blocks=28), v, plan)
    assert not plan.problems
    assert cmd[-6:] == ["--family_option", "replay_research=calibration",
                        "--family_option", "replay_calibration_lr=0.00045",
                        "--family_option", "movement_pilot_step=8"]


def test_isolation_queue_recipe(tmp_path):
    v = rb.isolation_settings("isolate", str(tmp_path))[0]
    assert v["__replay_compare"] == rb.MOVEMENT_ISOLATION_CASE
    assert v["MAX_TRAIN_EPOCHS"] == "1" and not v["SAMPLE_ENABLED"]
    assert not v["KREA2_SELECTIVE_BLOCK_CHECK"]
    cmd = ["python", "train.py"]
    plan = SimpleNamespace(problems=[], stages=[SimpleNamespace(name="Training", cmd=cmd)],
                           dataset_toml='holdout_file = "held.json"', console=[])
    rb.adjust_plan(SimpleNamespace(key="krea2", n_blocks=28), v, plan)
    assert not plan.problems
    assert cmd[-4:] == ["--family_option", "movement_pilot_step=8",
                        "--family_option", "movement_isolation=1"]


def test_dual_queue_recipe(tmp_path):
    v = rb.dual_settings("dual", str(tmp_path))[0]
    assert v["__replay_compare"] == rb.MOVEMENT_DUAL_CASE
    assert v["MAX_TRAIN_EPOCHS"] == "1" and not v["SAMPLE_ENABLED"]
    assert not v["KREA2_SELECTIVE_BLOCK_CHECK"]
    cmd = ["python", "train.py"]
    plan = SimpleNamespace(problems=[], stages=[SimpleNamespace(name="Training", cmd=cmd)],
                           dataset_toml='holdout_file = "held.json"', console=[])
    rb.adjust_plan(SimpleNamespace(key="krea2", n_blocks=28), v, plan)
    assert not plan.problems
    assert cmd[-4:] == ["--family_option", "movement_pilot_step=8",
                        "--family_option", "movement_dual=1"]
