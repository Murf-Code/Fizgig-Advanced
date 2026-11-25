"""Opt-in phone/desktop push via ntfy.sh. No network activity until explicitly enabled."""
import json
import logging
import os
import queue
import re
import tempfile
import threading
import urllib.error
import urllib.request

from fizgig.user_data import user_file

CONFIG = user_file("push_notifications.json")
EVENTS = {
    "finished": ("Training completed", True),
    "failed": ("Training failed", True),
    "oom": ("Out-of-memory recovery exhausted", True),
    "queue_done": ("Training queue drained", True),
    "paused": ("Training paused / restore point saved", False),
    "stopped": ("Training manually stopped", False),
    "started": ("Run started", False),
    "cache": ("Cache stage completed", False),
    "preview": ("Sample preview written", False),
}
_lock = threading.RLock()
_jobs = queue.Queue(maxsize=32)
_worker = None
_log = logging.getLogger(__name__)


def _read():
    try:
        with open(CONFIG, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _settings():
    raw = _read()
    chosen = raw.get("events")
    return {"enabled": raw.get("enabled") is True, "topic": str(raw.get("topic") or ""),
            "token": str(raw.get("token") or ""), "names": raw.get("names") is True,
            "events": {key: chosen.get(key) is True if isinstance(chosen, dict) and key in chosen else default
                       for key, (_, default) in EVENTS.items()}}


def status():
    with _lock:
        cfg = _settings()
        cfg["token_saved"] = bool(cfg.pop("token"))
        cfg["labels"] = {key: label for key, (label, _) in EVENTS.items()}
        return cfg


def configure(data):
    if not isinstance(data, dict):
        raise ValueError("Invalid notification settings.")
    with _lock:
        cfg = _settings()
        if "enabled" in data:
            cfg["enabled"] = data["enabled"] is True
        if "names" in data:
            cfg["names"] = data["names"] is True
        if "topic" in data:
            cfg["topic"] = str(data["topic"] or "").strip()
        if cfg["topic"] and not re.fullmatch(r"[A-Za-z0-9_-]{16,64}", cfg["topic"]):
            raise ValueError("Use a 16–64 character topic (letters, digits, hyphens or underscores).")
        if "token" in data:
            token = str(data["token"] or "").strip()
            if len(token) > 256 or (token and not re.fullmatch(r"[A-Za-z0-9_-]+", token)):
                raise ValueError("Invalid ntfy access token.")
            cfg["token"] = token
        if isinstance(data.get("events"), dict):
            for key in EVENTS:
                if key in data["events"]:
                    cfg["events"][key] = data["events"][key] is True
        if cfg["enabled"] and not cfg["topic"]:
            raise ValueError("Set an ntfy topic before enabling notifications.")
        # Separate local file: never include the token in general prefs, API reads or run history.
        folder = os.path.dirname(CONFIG) or "."
        os.makedirs(folder, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".push-", dir=folder)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            os.replace(tmp, CONFIG)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return status()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _send(cfg, event, name=""):
    title = "Fizgig Advanced"
    body = "Test notification from Fizgig Advanced." if event == "test" else EVENTS[event][0] + "."
    if cfg["names"] and name:
        body += " Run: " + str(name).replace("\r", " ").replace("\n", " ")[:160]
    headers = {"Title": title, "Content-Type": "text/plain; charset=utf-8",
               "Priority": "high" if event in ("failed", "oom") else "default"}
    if cfg["token"]:
        headers["Authorization"] = "Bearer " + cfg["token"]
    req = urllib.request.Request("https://ntfy.sh/" + cfg["topic"], data=body.encode("utf-8"),
                                 headers=headers, method="POST")
    try:
        with urllib.request.build_opener(_NoRedirect()).open(req, timeout=5) as response:
            if not 200 <= response.status < 300:
                raise OSError("Unexpected delivery status")
    except urllib.error.HTTPError as exc:
        raise ValueError("ntfy rejected the notification (HTTP %d)." % exc.code) from None
    except (OSError, ValueError):
        raise ValueError("Could not reach ntfy.sh. Check network access.") from None


def test():
    with _lock:
        cfg = _settings()
    if not cfg["enabled"] or not cfg["topic"]:
        raise ValueError("Enable and save push notifications first.")
    _send(cfg, "test")


def _drain():
    while True:
        cfg, event, name = _jobs.get()
        try:
            with _lock:
                current = _settings()
            if current == cfg and current["enabled"] and current["events"].get(event):
                _send(current, event, name)
        except Exception:
            # Delivery is best-effort; never log credentials, topic, run name, or break training.
            _log.warning("Push notification delivery failed.")
        finally:
            _jobs.task_done()


def emit(event, name=""):
    """Queue only selected events; no network I/O on the training or queue thread."""
    global _worker
    try:
        with _lock:
            cfg = _settings()
            if not cfg["enabled"] or not cfg["topic"] or not cfg["events"].get(event):
                return
            if _worker is None or not _worker.is_alive():
                _worker = threading.Thread(target=_drain, name="fizgig-push", daemon=True)
                _worker.start()
            _jobs.put_nowait((cfg, event, name))
    except Exception:
        _log.warning("Push notification could not be queued.")
