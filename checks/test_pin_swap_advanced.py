"""Upstream opt-in pinning and preference regression adapted to Advanced."""
import json
import pytest
from fizgig import user_data
from fizgig.families import quant, launch


def test_opt_in_and_override(tmp_path, monkeypatch):
    path = tmp_path / "prefs.json"
    monkeypatch.setattr(user_data, "PREFS_FILE", str(path))
    monkeypatch.delenv("FIZGIG_PIN_SWAP", raising=False)
    assert not quant.pin_swap_ok(1, 100)
    path.write_text(json.dumps({"pin_block_swap":"1"}))
    assert quant.pin_swap_ok(1, 5)
    assert not quant.pin_swap_ok(1, 4.9)
    monkeypatch.setenv("FIZGIG_PIN_SWAP", "0")
    assert not quant.pin_swap_ok(1, 100)
    monkeypatch.setenv("FIZGIG_PIN_SWAP", "1")
    assert not quant.pin_swap_ok(7, 10)
    assert quant.pin_swap_ok(7, 11)


def test_preferences(tmp_path, monkeypatch):
    from fizgig.web import server
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({"send_only_block_swap":"True","pin_block_swap":False}))
    monkeypatch.setattr(server, "PREFS_FILE", str(p))
    assert server.load_prefs()["send_only_block_swap"] == "1"
    assert server.load_prefs()["pin_block_swap"] == "0"
    assert launch.clean_reclaim_prefs({"pin_block_swap":True})["pin_block_swap"] == "1"
    assert launch.RECLAIM_PREF_DEFAULTS["pin_block_swap"] == "0"
