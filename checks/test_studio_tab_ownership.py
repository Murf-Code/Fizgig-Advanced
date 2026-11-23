"""Advanced Studio: a web tab must not silently act on another tab's load (#395).

The worker is intentionally shared; the session is a UI ownership guard and
is not a security/authentication credential.
"""
import pytest

from fizgig.web import studio


def test_current_session_can_act_but_stale_session_cannot():
    with studio._lock:
        prior = studio._S.copy()
        studio._S.update(load_session="tab-new", load_id=42, state="ready")
    try:
        studio.require_studio_session("tab-new")
        # Backward-compatible calls from internal tools don't require a web tab.
        studio.require_studio_session(None)
        with pytest.raises(ValueError, match="Another Studio tab"):
            studio.require_studio_session("tab-old")
        status = studio.status()
        assert status["load_id"] == 42
        assert status["load_session"] == "tab-new"
    finally:
        with studio._lock:
            studio._S.clear()
            studio._S.update(prior)


def test_fresh_idle_studio_does_not_reject_unload():
    with studio._lock:
        prior = studio._S.copy()
        studio._S.update(load_session="", load_id=0, state="idle")
    try:
        studio.require_studio_session("first-tab")  # no model is owned yet
        with studio._lock:
            studio._S.update(state="ready")
        with pytest.raises(ValueError, match="Another Studio tab"):
            studio.require_studio_session("first-tab")
    finally:
        with studio._lock:
            studio._S.clear()
            studio._S.update(prior)


def test_unload_does_not_mark_its_own_tab_as_foreign(monkeypatch):
    monkeypatch.setattr(studio, "stop_worker", lambda: None)
    with studio._lock:
        prior = studio._S.copy()
        studio._S.update(load_session="my-tab", load_id=7, state="ready")
    try:
        studio.reset(studio_session="my-tab")
        st = studio.status()
        assert st["load_session"] == "my-tab"
        assert st["load_id"] == 8
        assert st["state"] == "idle"
        studio.require_studio_session("my-tab")
        with pytest.raises(ValueError, match="Another Studio tab"):
            studio.require_studio_session("other-tab")
    finally:
        with studio._lock:
            studio._S.clear()
            studio._S.update(prior)


def test_server_scopes_studio_mutations():
    from pathlib import Path
    server = (Path(__file__).resolve().parents[1] / "src/fizgig/web/server.py").read_text()
    assert 'studio.require_studio_session(b.get("studio_session"))' in server
    for action in ('"render"', '"cancel"', '"donor"', '"settings"', '"save"', '"reset"', '"frank_export"'):
        assert action in server[server.index('if what in ("keep", "map"'):server.index('if what == "download_setup"')]
    assert 'studio_session=b.get("studio_session") or ""' in server
