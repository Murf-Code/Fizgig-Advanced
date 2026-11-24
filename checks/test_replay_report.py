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
