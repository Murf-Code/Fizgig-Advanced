"""Replay comparison queue recipe and launch regression checks."""
from types import SimpleNamespace

from fizgig.web import replay_benchmark as rb


def test_recipe_is_eight_independent_runs(tmp_path):
    runs = rb.settings("test", str(tmp_path), 3, "a test prompt")
    assert [v["__replay_compare"] for v in runs] == [c[0] for c in rb.CASES]
    assert len({v["LORA_NAME"] for v in runs}) == len({v["LORA_OUTPUT_DIR"] for v in runs}) == 8
    assert [v["KREA2_SELECTIVE_BLOCK_CHECK"] for v in runs] == [False] * 4 + [True] * 4
    assert [v["KREA2_TRAINING_METHOD"] for v in runs] == ["Global replay"] * 2 + ["Standard"] * 2 + ["Global replay"] * 4
    assert [v["LEARNING_RATE"] for v in runs] == ["0.00025", "0.00045"] * 2 + ["0.00045"] * 4
    varying = {"__replay_compare", "LORA_NAME", "LORA_OUTPUT_DIR", "KREA2_TRAINING_METHOD", "KREA2_SELECTIVE_BLOCK_CHECK", "LEARNING_RATE"}
    assert all({k: v for k, v in row.items() if k not in varying} ==
               {k: v for k, v in runs[0].items() if k not in varying} for row in runs)
    assert all(v["VALIDATION_HOLDOUT"] and v["SAMPLE_ENABLED"] for v in runs)
    assert all(v["MAX_TRAIN_EPOCHS"] == "5" for v in runs)


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
            expected += ["--family_option", "selective_strategy=" + values["__replay_compare"].removesuffix("-lr450"),
                         "--family_option", "selective_budget=14"]
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
