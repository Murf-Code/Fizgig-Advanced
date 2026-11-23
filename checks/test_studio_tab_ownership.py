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


def test_server_scopes_studio_mutations():
    from pathlib import Path
    server = (Path(__file__).resolve().parents[1] / "src/fizgig/web/server.py").read_text()
    assert 'studio.require_studio_session(b.get("studio_session"))' in server
    for action in ('"render"', '"cancel"', '"donor"', '"settings"', '"save"', '"reset"', '"frank_export"'):
        assert action in server[server.index('if what in ("keep", "map"'):server.index('if what == "download_setup"')]
    assert 'studio_session=b.get("studio_session") or ""' in server
