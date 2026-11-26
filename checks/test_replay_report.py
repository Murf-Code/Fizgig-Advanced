from fizgig.web import replay_report as rr


def test_report_preserves_runs_and_partial_failure(tmp_path):
    root = tmp_path / "comparison"
    configs = [{"__replay_compare": k, "LORA_NAME": k, "LEARNING_RATE": "0.00025"} for k in ("a", "b")]
    rr.create(root, "dataset", configs)
    log = tmp_path / "original.log"
    log.write_text("epoch 1/3 avr_loss=0.1584 step=29 3.09s/step\n[validation] step 0 (baseline): loss 0.19972\n[validation] step 29 (epoch 1): loss 0.19952\n[selective-replay] probe 8: 14/28 blocks active\n[selective-replay] active_blocks=[0, 2] scores=[0.1, None] strategy=downhill\n", encoding="utf-8")
    rr.update(root / "a", "a", "running")
    rr.update(root / "a", "a", "finished", log_path=log, wall_seconds=120)
    log.unlink()
    rr.update(root / "b", "b", "failed", reason="example failure")
    report = rr._read_json(root / rr.REPORT)
    assert report["status"] == "partial"
    a, b = report["runs"]
    assert a["epochs"][0]["seconds_per_step"] == 3.09
    assert a["validation"][-1]["loss"] == 0.19952
    assert a["probes"][0]["active_blocks"] == [0, 2]
    assert b["reason"] == "example failure"
    assert "avr_loss=0.1584" in (root / rr.COMBINED).read_text(encoding="utf-8")


def test_all_finished_and_removed(tmp_path):
    root = tmp_path / "comparison"
    rr.create(root, "dataset", [{"__replay_compare": "a", "LORA_NAME": "a", "LEARNING_RATE": "0.00045"}])
    rr.update(root / "a", "a", "finished")
    assert rr._read_json(root / rr.REPORT)["status"] == "complete"
    rr.update(root / "a", "a", "removed")
    assert rr._read_json(root / rr.REPORT)["status"] == "failed"


def test_probe_details_with_explicit_steps_and_missing_header():
    log = ("steps: 9%|...| INFO:__main__:[selective-replay] active_blocks=[0] scores=[0.1, 0.2] strategy=uphill step=8\n"
           "[selective-replay] probe 16: 1/2 blocks active\n"
           "[selective-replay] active_blocks=[1] scores=[0.3, 0.4] strategy=uphill step=16\n")
    probes = rr.parse_log(log)["probes"]
    assert [p["step"] for p in probes] == [8, 16]
    assert [p["active_blocks"] for p in probes] == [[0], [1]]


def test_final_bundle_excludes_weights(tmp_path):
    import zipfile
    root = tmp_path / "comparison"
    rr.create(root, "dataset", [{"__replay_compare": "a", "LORA_NAME": "a", "LEARNING_RATE": "0.00045"}])
    out = root / "a"
    (out / "sample").mkdir(parents=True)
    (out / "sample" / "sample.png").write_bytes(b"sample")
    (out / "weights.safetensors").write_bytes(b"weights")
    rr.update(out, "a", "finished")
    with zipfile.ZipFile(root / rr.BUNDLE) as bundle:
        assert set(bundle.namelist()) == {rr.REPORT, rr.COMBINED, "a/sample/sample.png"}


def test_walking_rotation_records_are_separate_from_probes():
    log = ("[selective-replay] probe 8: 2/4 blocks active\n"
           "[selective-replay] active_blocks=[0, 1] scores=[4, 3, 2, 1] strategy=walking_coverage step=8\n"
           "steps: 9%|...| INFO:[selective-replay] rotation next_step=10 active_blocks=[0, 2] strategy=walking_coverage\n"
           "[selective-replay] rotation next_step=11 active_blocks=[0, 3] strategy=walking_coverage\n")
    result = rr.parse_log(log)
    assert len(result["probes"]) == 1
    assert result["rotations"] == [
        {"next_step": 10, "active_blocks": [0, 2], "strategy": "walking_coverage"},
        {"next_step": 11, "active_blocks": [0, 3], "strategy": "walking_coverage"},
    ]


def test_gradient_update_counts_are_parsed_separately():
    result = rr.parse_log("INFO:[selective-replay] gradient_updates step=29 counts=[29, 12]\n")
    assert result["gradient_updates"] == [{"step": 29, "counts": [29, 12]}]
    assert not result["probes"] and not result["rotations"]


def test_failed_pilot_bundle_includes_diagnostics(tmp_path):
    import zipfile
    root = tmp_path / "comparison"
    rr.create(root, "dataset", [{"__replay_compare": "movement-isolation-lr450",
              "LORA_NAME": "isolate", "LEARNING_RATE": "0.00045"}])
    out = root / "movement-isolation-lr450"
    out.mkdir()
    (out / "movement-isolation.json").write_text('{"status":"failed"}')
    (out / "movement-pilot.json").write_text('{}')
    (out / "weights.safetensors").write_bytes(b"not exported")
    rr.update(out, "movement-isolation-lr450", "failed")
    with zipfile.ZipFile(root / rr.BUNDLE) as bundle:
        assert set(bundle.namelist()) == {rr.REPORT, rr.COMBINED,
            "movement-isolation-lr450/movement-isolation.json",
            "movement-isolation-lr450/movement-pilot.json"}


def test_dual_failed_bundle_includes_report(tmp_path):
    import zipfile
    root = tmp_path / "comparison"
    rr.create(root, "dataset", [{"__replay_compare": "movement-dual-lr450",
              "LORA_NAME": "dual", "LEARNING_RATE": "0.00045"}])
    out = root / "movement-dual-lr450"
    out.mkdir()
    (out / "movement-dual.json").write_text('{"status":"failed"}')
    rr.update(out, "movement-dual-lr450", "failed")
    with zipfile.ZipFile(root / rr.BUNDLE) as bundle:
        assert set(bundle.namelist()) == {rr.REPORT, rr.COMBINED,
            "movement-dual-lr450/movement-dual.json"}
