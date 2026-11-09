"""The CPU / GPU / disk readout (training/perf_watch.py): its sampler thread reads the data-wait share from background
loading's timing and writes the live readout and, at the end, a summary with a plain hint, which /api/run hands the
run card; the hint's wording at each level; the run card draws it with theme status colours."""
import json
import os
import time

from fizgig.training import perf_watch
from fizgig.training.prefetch import Stats


def test_sampler_writes_live_then_summary(tmp_path, caplog):
    st = Stats()
    caplog.set_level("INFO")
    w = perf_watch.start(str(tmp_path), "run", st, background_loading=True, interval=0.05, gpu=False)
    for _ in range(10):                      # 10 steps: 30 ms waiting, 70 ms training each
        st.wait_s += 0.03
        st.busy_s += 0.07
        st.batches += 1
        time.sleep(0.02)
    time.sleep(0.15)
    live = json.load(open(perf_watch.file_for(str(tmp_path), "run"), encoding="utf-8"))["live"]
    assert "wait_share_run" in live and "cpu" in live and w.samples >= 2
    sm = perf_watch.finish()
    data = json.load(open(perf_watch.file_for(str(tmp_path), "run"), encoding="utf-8"))
    assert data["summary"] == sm and sm["steps"] == 10 and abs(sm["wait_share"] - 0.3) < 1e-6
    assert round(sm["wait_ms"]) == 30 and round(sm["step_ms"]) == 70
    assert sm["hint"].startswith("GPU waited on data 30% of the time - the cache drive may be slow")
    assert "[perf] data wait 30%" in caplog.text and "cache drive may be slow" in caplog.text
    assert perf_watch.finish() is None                       # ended once


def test_hints():
    h = perf_watch.hint
    assert "Background loading is off" in h({"wait_share": 0.4, "background_loading": False})
    assert "faster drive" in h({"wait_share": 0.4, "background_loading": True})
    assert "a little" in h({"wait_share": 0.12})
    assert "busy only 40%" in h({"wait_share": 0.02, "gpu_util": 40})
    assert h({"wait_share": 0.02, "gpu_util": 95}).startswith("Data kept up")
    assert h({}) == ""


def test_off_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FIZGIG_PERF_READOUT", "0")
    assert perf_watch.start(str(tmp_path), "x", Stats()) is None
    assert not os.path.exists(perf_watch.file_for(str(tmp_path), "x"))


def test_api_run_hands_it_to_the_run_card(tmp_path):
    from fizgig.web import server
    path = perf_watch.file_for(str(tmp_path), "lora")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"live": {"cpu": 12.0}}, f)
    r = {"output_dir": str(tmp_path), "lora": "lora", "started_at": time.time()}
    assert server._perf_status(r) == {"live": {"cpu": 12.0}}
    assert server._perf_status({**r, "started_at": time.time() + 60}) is None     # an earlier run's file
    assert server._perf_status({**r, "lora": ""}) is None
    root = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static")
    js = open(os.path.join(root, "app.js"), encoding="utf-8").read()
    css = open(os.path.join(root, "style.css"), encoding="utf-8").read()
    assert "runPerf(r.perf" in js and 'id="runPerf"' in open(os.path.join(root, "index.html"), encoding="utf-8").read()
    block = css[css.index(".run-perf {"):css.index(".perf-hint.bad")]
    assert "#" not in block.replace("#runPerf", "")           # theme tokens only: the colour-safe theme swaps them
