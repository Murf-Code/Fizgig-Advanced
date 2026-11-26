"""Fixed geometry queue controls and report lifecycle (no GPU)."""
from pathlib import Path
import json
import zipfile
from fizgig.web import geometry_benchmark as bench, geometry_report as report


def test_fixed_pilot(tmp_path):
    configs=bench.settings("pilot", str(tmp_path))
    assert len(configs)==5
    assert len({v["LORA_OUTPUT_DIR"] for v in configs})==5
    assert {v["OPTIMIZER_TYPE"] for v in configs}=={"adamw8bit","muon","polora","lora_tsd"}
    for v in configs:
        assert v["NETWORK_DIM"]==v["NETWORK_ALPHA"]=="32"
        assert v["MAX_TRAIN_EPOCHS"]=="5"
        assert v["FAMILY_EMA"]=="Off" and v["KREA2_TRAINING_METHOD"]=="Standard"
        assert v["VALIDATION_HOLDOUT"] and v["SAMPLE_AT_FIRST"]
        assert not v["ADAPTIVE_LR"] and not v["KREA2_SELECTIVE_BLOCK_CHECK"]
    root=tmp_path/"pilot"
    report.create(root,"dataset",configs)
    for i,v in enumerate(configs):
        out=Path(v["LORA_OUTPUT_DIR"])
        out.mkdir()
        log=out/"run.log"
        log.write_text("[validation] step 0 (before): loss 1.0\n[validation] step 5 (epoch): loss 0.5\n",encoding="utf-8")
        report.update(out,v["__geometry_compare"],"finished" if i!=3 else "failed",log_path=log)
    data=json.loads((root/report.REPORT).read_text())
    assert data["status"]=="partial"
    assert len(data["runs"])==5
    assert data["runs"][0]["validation"][-1]["loss"]==0.5
    with zipfile.ZipFile(root/report.BUNDLE) as z:
        assert report.REPORT in z.namelist() and report.COMBINED in z.namelist()

def test_rank64_feasibility_is_separate(tmp_path):
    v, = bench.feasibility_settings("fit", str(tmp_path))
    assert v["NETWORK_DIM"] == v["NETWORK_ALPHA"] == "64"
    assert v["OPTIMIZER_TYPE"] == "polora"
    assert v["MAX_TRAIN_EPOCHS"] == "1"
    assert not v["SAMPLE_ENABLED"] and not v["SAVE_STATE_ON_TRAIN_END"]
    assert all(x["NETWORK_DIM"] == "32" for x in bench.settings("pilot", str(tmp_path)))
    parsed = report.parse_log("[geometry-fit] complete updates=50 allocated_bytes=1024 reserved_bytes=2048 wall_seconds=12.5")
    assert parsed["feasibility"]["updates"] == 50
    assert parsed["feasibility"]["cuda_peak_reserved_bytes"] == 2048

def test_resolution_queue(tmp_path):
    configs = bench.resolution_settings("resolution", str(tmp_path))
    assert len(configs) == 10
    assert len({v["LORA_OUTPUT_DIR"] for v in configs}) == 10
    assert {(v["NETWORK_DIM"], v["DATASET_MEGAPIXELS"]) for v in configs[:8]} == {
        (rank, mp) for rank in ("16", "32") for mp in ("0.25", "0.50", "0.75", "1.00")}
    for v in configs:
        assert bench.is_feasibility(v["__geometry_compare"])
        assert v["NETWORK_DIM"] == v["NETWORK_ALPHA"]
        assert v["MAX_TRAIN_EPOCHS"] == "1" and not v["SAMPLE_ENABLED"]
    assert configs[-2]["OPTIMIZER_ARGS"].endswith("chunk_memory_mib=128")
    assert configs[-1]["OPTIMIZER_ARGS"].endswith("chunk_memory_mib=0")
    assert configs[-2]["DATASET_MEGAPIXELS"] == configs[-1]["DATASET_MEGAPIXELS"] == "0.25"
    root = tmp_path / "resolution"
    report.create(root, "dataset", configs)
    v = configs[0]
    out = Path(v["LORA_OUTPUT_DIR"])
    out.mkdir()
    log = out / "run.log"
    log.write_text("torch.OutOfMemoryError: CUDA out of memory.", encoding="utf-8")
    report.update(out, v["__geometry_compare"], "failed", log_path=log)
    data = json.loads((root / report.REPORT).read_text())
    assert data["runs"][0]["failure_kind"] == "cuda_oom"
    assert data["runs"][1]["status"] == "queued"
    assert report.parse_log("bucket: (496, 496), count: 29")["reported_buckets"][0]["width"] == 496


def test_optimizer_comparison(tmp_path):
    quality = bench.optimizer_settings("quality", str(tmp_path))
    short = bench.optimizer_settings("fit", str(tmp_path), short=True)
    assert len(quality) == 6 and len(short) == 2
    for v in quality + short:
        assert v["NETWORK_DIM"] == v["NETWORK_ALPHA"] == "16"
        assert v["BLOCKS_SWAP"] == "18" and v["DATASET_MEGAPIXELS"] == "0.25"
        assert v["VALIDATION_HOLDOUT"] and not v["KREA2_SELECTIVE_BLOCK_CHECK"]
    assert all(v["SAMPLE_ENABLED"] and v["MAX_TRAIN_EPOCHS"] == "5" for v in quality)
    assert all(not v["SAMPLE_ENABLED"] and bench.is_feasibility(v["__geometry_compare"])
               and v["MAX_TRAIN_EPOCHS"] == "1" and v["OPTIMIZER_TYPE"] == "lora_tsd" for v in short)
    assert quality[2]["OPTIMIZER_ARGS"].replace("research_state_views=False","research_state_views=True") == quality[3]["OPTIMIZER_ARGS"]
    parsed = report.parse_log("[geometry-opt] update=4 ms=12.5 allocated_bytes=1024 reserved_bytes=2048 alloc_retries=1\n"
                              "[geometry-tsd] update=4 max_change=0.299 warm=2 unchanged=0 shrunk=1 pairs=264")
    assert parsed["optimizer_ms_mean_after_warmup"] == 12.5
    assert parsed["tsd_stats"][0]["warm"] == 2


def test_round10_queue(tmp_path):
    configs = bench.round10_settings("r10", str(tmp_path))
    assert len(configs) == 7 and len({v["LORA_OUTPUT_DIR"] for v in configs}) == 7
    exclude = {"__geometry_compare", "LORA_NAME", "LORA_OUTPUT_DIR"}
    assert {k:v for k,v in configs[0].items() if k not in exclude} == {k:v for k,v in configs[1].items() if k not in exclude}
    assert all(v["NETWORK_DIM"] == v["NETWORK_ALPHA"] == "16" and v["BLOCKS_SWAP"] == "18"
               and v["DATASET_MEGAPIXELS"] == "0.25" and v["MAX_TRAIN_EPOCHS"] == "5"
               and v["VALIDATION_HOLDOUT"] and v["SAMPLE_ENABLED"] for v in configs)
    assert all("ball_iters=1" in v["OPTIMIZER_ARGS"] and "balance=False" in v["OPTIMIZER_ARGS"] for v in configs[2:])
    assert configs[6]["OPTIMIZER_ARGS"] == configs[5]["OPTIMIZER_ARGS"] + " polar=svd"
    parsed = report.parse_log("[geometry-tangent] update=31 pairs=264 budget_min=0.05 budget_mean=0.2 budget_max=0.9 lr_factor_mean=0.4 change_sum=1.25")
    assert parsed["tangent_stats"][0]["budget_mean"] == 0.2
    assert parsed["tangent_stats"][0]["change_sum_all_pairs"] == 1.25

def test_muon_me_queue(tmp_path):
    from types import SimpleNamespace
    configs = bench.muon_me_settings("me", str(tmp_path))
    assert len(configs) == 6 and len({v["LORA_OUTPUT_DIR"] for v in configs}) == 6
    exclude = {"__geometry_compare", "LORA_NAME", "LORA_OUTPUT_DIR", "OPTIMIZER_TYPE", "OPTIMIZER_ARGS"}
    assert all({k:v for k,v in x.items() if k not in exclude} == {k:v for k,v in configs[0].items() if k not in exclude} for x in configs[:4])
    assert all(v["NETWORK_DIM"] == v["NETWORK_ALPHA"] == "64" and v["DATASET_MEGAPIXELS"] == "0.25" and v["VALIDATION_HOLDOUT"] and v["SAMPLE_ENABLED"] for v in configs)
    assert [v["BLOCKS_SWAP"] for v in configs] == ["22"]*4 + ["16"]*2
    assert [v["MAX_TRAIN_EPOCHS"] for v in configs] == ["5"]*4 + ["1"]*2
    assert configs[2]["OPTIMIZER_ARGS"].endswith("chunk_memory_mib=0")
    assert configs[3]["OPTIMIZER_ARGS"].endswith("chunk_memory_mib=128")
    for v in configs:
        plan = SimpleNamespace(problems=[], stages=[SimpleNamespace(name="Training",cmd=["--holdout_file","holdout","--sample_prompts","prompts"])],dataset_toml="",console=[])
        bench.adjust_plan(SimpleNamespace(key="krea2"),v,plan)
        assert not plan.problems and "geometry_optimizer_timing=1" in plan.stages[0].cmd
