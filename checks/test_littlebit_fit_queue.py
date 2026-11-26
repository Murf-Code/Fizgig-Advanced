import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import tomllib
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research/krea2_littlebit"))
import fit_queue as fit


def setup_case(tmp_path, monkeypatch):
    training = ModuleType("fizgig.training")
    training.holdout = SimpleNamespace(give_back=lambda _: None)
    monkeypatch.setitem(sys.modules,"fizgig",ModuleType("fizgig"))
    monkeypatch.setitem(sys.modules,"fizgig.training",training)
    hold = tmp_path / "original.json"
    hold.write_text(json.dumps({"paths":["one.png","two.png"]}))
    config = tmp_path / "data.toml"
    config.write_text('[general]\nholdout_file = '+json.dumps(hold.as_posix())+'\n[[datasets]]\nimage_directory = "data"\n')
    out = tmp_path / "out"
    cmd = ["fake-worker","--","--output_dir",str(out),"--output_name","demo",
           "--dataset_config",str(config),"--blocks_to_swap","22"]
    return out, cmd, hold


@pytest.mark.parametrize("statuses,expected,ret", [
    ([(0,"finished")],0,0),
    ([(86,"cuda_oom"),(86,"cuda_oom"),(0,"finished")],8,0),
    ([(1,"error")],None,1),
    ([(86,"error")],None,86),
    ([(86,"cuda_oom")]*len(fit.SWAP_STEPS),None,86),
])
def test_sweep(tmp_path, monkeypatch, statuses, expected, ret):
    out, cmd, hold = setup_case(tmp_path,monkeypatch)
    original = hold.read_bytes()
    seen = []
    def launch(args):
        count = int(fit.arg(args,"--blocks_to_swap"))
        folder = Path(fit.arg(args,"--output_dir"))
        cfg = tomllib.loads(Path(fit.arg(args,"--dataset_config")).read_text())
        retry_hold = Path(cfg["general"]["holdout_file"])
        assert retry_hold.read_bytes() == original
        assert hold.read_bytes() == original
        assert cfg["datasets"][0]["image_directory"] == "data"
        retry_hold.write_text('{}')
        code, status = statuses[len(seen)]
        seen.append(count)
        Path(fit.arg(args,"--attempt-result")).write_text(json.dumps({"status":status}))
        (folder/"demo.safetensors").write_text(str(count))
        if status == "finished":
            (folder/".fizgig_val_demo.pt").write_text("snapshot")
        return SimpleNamespace(returncode=code)
    assert fit.run_sweep(cmd,out,launch)==ret
    report=json.loads((out/"fit-report.json").read_text())
    assert report["selected_swaps"]==expected
    assert seen==list(fit.SWAP_STEPS[:len(statuses)])
    if expected is None:
        assert not (out/"demo.safetensors").exists()
    else:
        assert (out/"demo.safetensors").read_text()==str(expected)
        assert report["status"]=="finished"
