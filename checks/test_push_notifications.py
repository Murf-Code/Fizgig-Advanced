"""Push notification privacy, opt-in and transport tests (no actual network)."""
import json
from unittest.mock import MagicMock
import pytest
from fizgig.web import push_notifications as push


@pytest.fixture
def local(monkeypatch, tmp_path):
    monkeypatch.setattr(push, "CONFIG", str(tmp_path / "push.json"))
    return tmp_path


def test_default_off(local, monkeypatch):
    sender = MagicMock()
    monkeypatch.setattr(push, "_send", sender)
    assert not push.status()["enabled"]
    push.emit("finished")
    with pytest.raises(ValueError):
        push.test()
    sender.assert_not_called()
    assert not (local / "push.json").exists()


def test_config_validation_and_token(local):
    with pytest.raises(ValueError):
        push.configure({"enabled": True, "topic": "../bad"})
    cfg = push.configure({"enabled": True, "topic": "fizgig-test-1234567890", "token": "tk_secret"})
    assert cfg["token_saved"] and "token" not in cfg
    assert cfg["events"]["finished"] and not cfg["events"]["preview"]
    assert not cfg["names"]
    push.configure({"enabled": False})
    assert not push.status()["enabled"]
    assert json.loads((local / "push.json").read_text())["token"] == "tk_secret"
    push.configure({"token": ""})
    assert not push.status()["token_saved"]


def test_test_notification_transport(local, monkeypatch):
    push.configure({"enabled": True, "topic": "fizgig-test-1234567890", "token": "tk_secret"})
    opener = MagicMock()
    opener.open.return_value.__enter__.return_value.status = 200
    monkeypatch.setattr(push.urllib.request, "build_opener", lambda *args: opener)
    push.test()
    request = opener.open.call_args.args[0]
    assert request.full_url == "https://ntfy.sh/fizgig-test-1234567890"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer tk_secret"
    assert opener.open.call_args.kwargs["timeout"] == 5
    push._send(push._settings(), "finished", "private-model")
    assert b"private-model" not in opener.open.call_args.args[0].data
    push.configure({"names": True})
    push._send(push._settings(), "finished", "private-model")
    assert b"private-model" in opener.open.call_args.args[0].data


def test_event_filter(local, monkeypatch):
    push.configure({"enabled": True, "topic": "fizgig-test-1234567890"})
    class Worker:
        def is_alive(self):
            return True
    monkeypatch.setattr(push, "_worker", Worker())
    pending = MagicMock()
    monkeypatch.setattr(push, "_jobs", pending)
    push.emit("preview")
    pending.put_nowait.assert_not_called()
    push.emit("finished", "run")
    assert pending.put_nowait.call_args.args[0][1:] == ("finished", "run")
    push.configure({"enabled": False})
    push.emit("failed")
    assert pending.put_nowait.call_count == 1
