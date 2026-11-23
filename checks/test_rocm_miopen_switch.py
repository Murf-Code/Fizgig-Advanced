"""Upstream #389: FIZGIG_MIOPEN=0 is opt-in and AMD-only at process import."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
torch = pytest.importorskip("torch")

from fizgig.utils import gpu_backend


def test_explicit_disable(monkeypatch):
    monkeypatch.setenv("FIZGIG_MIOPEN", "0")
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)
    gpu_backend.miopen_switch()
    assert torch.backends.cudnn.enabled is False


def test_unset_is_noop(monkeypatch):
    monkeypatch.delenv("FIZGIG_MIOPEN", raising=False)
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)
    gpu_backend.miopen_switch()
    assert torch.backends.cudnn.enabled is True


def test_standard_cuda_import_does_not_disable_cudnn(tmp_path):
    # Even if a user left the variable set, NVIDIA must not take AMD's path.
    root = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ, FIZGIG_MIOPEN="0", FIZGIG_GPU_BACKEND="cuda",
               PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-c",
                             "import fizgig, torch; print(torch.backends.cudnn.enabled)"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "True"


def test_windows_launcher_prefers_actual_database_when_present():
    launcher = (Path(__file__).resolve().parents[1] / "Launch Fizgig AMD.bat").read_text()
    bin_fallback = 'if exist "!ROCM_DEVEL_BIN!" set "MIOPEN_SYSTEM_DB_PATH=!ROCM_DEVEL_BIN!"'
    devel_share = 'MIOPEN_SYSTEM_DB_PATH=!ROCM_SP!\\_rocm_sdk_devel\\share\\miopen\\db'
    libraries_share = 'MIOPEN_SYSTEM_DB_PATH=!ROCM_SP!\\_rocm_sdk_libraries\\share\\miopen\\db'
    assert launcher.index(bin_fallback) < launcher.index(devel_share) < launcher.index(libraries_share)
    assert 'if /I not "!GPU_ARCH!"=="gfx1100" (' in launcher
