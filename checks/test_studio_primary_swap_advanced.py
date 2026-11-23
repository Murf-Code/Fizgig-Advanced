"""Advanced Studio regression: primary LoRA swaps (#394), ownership and render metadata."""
import base64
import os

import pytest

W = pytest.importorskip("fizgig.scripts.studio_worker")
from fizgig.web import studio


class _Engine:
    primary_block_ids = donor_block_ids = ()
    primary_hash = ""

    def __init__(self):
        self.primary_network = object()
        self.primary_path = "old.safetensors"
        self.calls = []

    def unload_primary(self):
        self.calls.append("unload")
        self.primary_network = None

    def load_primary(self, path):
        self.calls.append(("load", path))
        if os.path.basename(path) == "invalid.safetensors":
            raise ValueError("Wrong model")
        self.primary_path = path
        self.primary_network = object()


def test_primary_swap_and_rollback(monkeypatch, tmp_path):
    eng, events = _Engine(), []
    monkeypatch.setitem(W.S, "engine", eng)
    monkeypatch.setitem(W.S, "desc", type("Desc", (), {"video_workbench": False})())
    monkeypatch.setitem(W.S, "frank_active", False)
    monkeypatch.setattr(W, "emit", lambda **event: events.append(event))
    fresh, invalid = tmp_path / "fresh.safetensors", tmp_path / "invalid.safetensors"
    fresh.touch()
    invalid.touch()
    W.cmd_primary({"path": str(fresh)})
    assert eng.calls == ["unload", ("load", str(fresh))]
    assert events[-1]["primary"] == str(fresh)
    eng.calls.clear()
    W.cmd_primary({"path": str(invalid)})
    assert eng.calls == ["unload", ("load", str(invalid)), ("load", str(fresh))]
    assert eng.primary_path == str(fresh)
    assert events[-1]["primary"] == str(fresh) and "Wrong model" in events[-1]["error"]


def test_primary_swap_rejects_frankenstein(monkeypatch, tmp_path):
    eng = _Engine()
    monkeypatch.setitem(W.S, "engine", eng)
    monkeypatch.setitem(W.S, "frank_active", True)
    with pytest.raises(ValueError, match="Frankenstein"):
        W.cmd_primary({"path": str(tmp_path / "not-used.safetensors")})
    assert eng.calls == []


def test_owner_guard_busy_and_acknowledgement(monkeypatch, tmp_path):
    state = dict(studio._S, state="ready", rendering=False, primary="old.safetensors",
                 primary_busy=False, primary_seq=0, load_id=5, load_session="tab-a",
                 frankenstein="", nolora_busy=False)
    monkeypatch.setattr(studio, "_S", state)
    commands = []
    monkeypatch.setattr(studio, "_send", lambda command: commands.append(command))
    file = tmp_path / "new.safetensors"
    file.touch()
    with pytest.raises(ValueError, match="Another Studio tab"):
        studio.require_studio_session("tab-b")
    assert studio.primary(str(file)) == 6
    assert state["primary_busy"] and commands[-1]["cmd"] == "primary"
    with pytest.raises(ValueError, match="busy"):
        studio.primary(str(file))
    studio._event({"ev": "primary", "primary": str(file), "primary_blocks": [],
                   "donor_blocks": [], "hash": ""})
    assert state["primary_seq"] == 1 and state["primary"] == str(file)
    assert not state["primary_busy"]


def test_render_metadata_follows_pending_generation(monkeypatch):
    state = dict(studio._S, state="ready", rendering=False, primary_busy=False,
                 nolora_busy=False, gen=0, shown=0, render_setups={})
    monkeypatch.setattr(studio, "_S", state)
    sent = []
    monkeypatch.setattr(studio, "_send", lambda command: sent.append(command))
    first = {"fizgig_repair_setup": 1, "values": {"seed": 1}}
    second = {"fizgig_repair_setup": 1, "values": {"seed": 2}}
    studio.render({"seed": 1}, {"_setup": first})
    assert state["render_setups"][1] == first and "_setup" not in sent[-1]
    studio.render({"seed": 2}, {"_setup": second})
    pending = state["pending"]
    assert pending[1]["_setup"] == second
    state["rendering"] = False
    studio.render(*pending)
    state["shown"] = 2
    assert studio.render_setup_for(2) == second
    assert studio.render_setup_for(1) is None


def test_png_metadata_is_embedded(tmp_path):
    # A valid tiny PNG; the setup is encoded as an iTXt chunk before IEND.
    pixel = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLttAAAAABJRU5ErkJggg=="
    )
    src = tmp_path / "preview.png"
    src.write_bytes(pixel)
    result = studio.png_with_setup(src, {"fizgig_repair_setup": 1, "values": {"seed": 42}})
    assert result.startswith(bytes([137, 80, 78, 71]))
    assert b"fizgig_studio_setup" in result
    assert b"seed" in result


def test_ui_swaps_before_falling_back_to_full_load():
    from pathlib import Path
    source = (Path(studio.__file__).parent / "static" / "app.js").read_text(encoding="utf-8")
    start = source[source.index("async function studioStart()"):source.index("async function studioPoll()")]
    assert "studioSwapPrimary(key," in start
    assert start.index("studioSwapPrimary(key,") < start.index('api("/api/studio/load"')
    assert 'st.load_session === studioSession()' in start
    assert 'L.kind !== "video"' in start
