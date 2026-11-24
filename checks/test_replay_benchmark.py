"""Replay comparison queue recipe and launch regression checks."""
from types import SimpleNamespace

from fizgig.web import replay_benchmark as rb


def test_recipe_is_three_independent_runs(tmp_path):
    runs = rb.settings("test", str(tmp_path), 3, "a test prompt")
    assert [v["__replay_compare"] for v in runs] == ["replay", "selective", "standard"]
    assert len({v["LORA_NAME"] for v in runs}) == len({v["LORA_OUTPUT_DIR"] for v in runs}) == 3
    assert [v["KREA2_SELECTIVE_BLOCK_CHECK"] for v in runs] == [False, True, False]
    assert [v["KREA2_TRAINING_METHOD"] for v in runs] == ["Global replay", "Global replay", "Standard"]
    varying = {"__replay_compare", "LORA_NAME", "LORA_OUTPUT_DIR", "KREA2_TRAINING_METHOD", "KREA2_SELECTIVE_BLOCK_CHECK"}
    assert all({k: v for k, v in row.items() if k not in varying} ==
               {k: v for k, v in runs[0].items() if k not in varying} for row in runs)
    assert all(v["VALIDATION_HOLDOUT"] and v["SAMPLE_ENABLED"] for v in runs)
    assert all(v["MAX_TRAIN_EPOCHS"] == "3" for v in runs)


def test_comparison_preserves_normal_adapter_targets(tmp_path):
    desc = SimpleNamespace(key="krea2", n_blocks=28)
    for values in rb.settings("test", str(tmp_path)):
        cmd = ["python", "train.py"]
        plan = SimpleNamespace(problems=[], stages=[SimpleNamespace(name="Training", cmd=cmd)],
                               dataset_toml='holdout_file = "held.json"', console=[])
        rb.adjust_plan(desc, values, plan)
        assert not plan.problems
        assert cmd == ["python", "train.py"]


def test_no_holdout_is_rejected(tmp_path):
    desc = SimpleNamespace(key="krea2", n_blocks=28)
    values = rb.settings("test", str(tmp_path))[0]
    plan = SimpleNamespace(problems=[], stages=[SimpleNamespace(name="Training", cmd=[])],
                           dataset_toml="", console=[])
    rb.adjust_plan(desc, values, plan)
    assert plan.problems
