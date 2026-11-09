"""Is there a newer Fizgig Advanced release? The desktop app's check (lora_trainer_gui._check_for_update), for the web
UI's About box: the highest advanced-X.Y[.Z] tag on origin (one read-only `git ls-remote`, no GitHub API), and whether
this checkout already contains it. Upstream Fizgig's vX.Y.Z tags are never offered: this is the fork, and its releases
are its own (an install whose origin is upstream finds no advanced-* tag and stays silent). Silent when it can't tell
(offline, no origin, a ZIP download): never a false nag."""
import os
import re
import subprocess
import threading

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_NO_WINDOW = {"creationflags": 0x08000000} if os.name == "nt" else {}
_lock = threading.Lock()
_state = {"status": "", "tag": "", "running": False}
TAG_GLOB = "advanced-*"                       # Fizgig Advanced's release tags (advanced-1.4, advanced-1.3.1, ...)
_TAG = re.compile(r"advanced-(\d+(?:\.\d+)*)")


def release_key(tag):
    """advanced-1.3.1 -> (1, 3, 1); None for any other tag (upstream's v9.2.5 included)."""
    m = _TAG.fullmatch(str(tag or "").strip())
    return tuple(int(p) for p in m.group(1).split(".")) if m else None


def _git(*args, timeout=8):
    try:
        r = subprocess.run(["git", "-C", BASE_DIR, *args], capture_output=True, text=True, timeout=timeout,
                           **_NO_WINDOW)
        return r.returncode, r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 1, ""


def _latest_release_tag():
    code, out = _git("ls-remote", "--tags", "--refs", "origin", TAG_GLOB, timeout=15)
    if code or not out:
        return None
    best, best_key = None, None
    for line in out.splitlines():
        try:
            sha, ref = line.split(None, 1)
        except ValueError:
            continue
        tag = ref.strip().rsplit("/", 1)[-1]
        key = release_key(tag)
        if key is None:
            continue                       # not a plain advanced-X.Y[.Z] tag
        if best_key is None or key > best_key:
            best, best_key = (tag, sha), key
    return best


def check():
    """("update_available", tag) / ("up_to_date", "") / ("unknown", "") - the desktop's rules: up to date when HEAD
    contains the release's commit, or (a pod's shallow clone, which hasn't the tag's commit) is the remote tip."""
    latest = _latest_release_tag()
    if latest is None:
        return "unknown", ""
    tag, sha = latest
    has = _git("cat-file", "-e", f"{sha}^{{commit}}")[0] == 0
    if has and _git("merge-base", "--is-ancestor", sha, "HEAD")[0] == 0:
        return "up_to_date", ""
    remote = (_git("ls-remote", "origin", "HEAD", timeout=15)[1] or "").split()
    if remote and remote[0] == _git("rev-parse", "HEAD")[1]:
        return "up_to_date", ""
    return "update_available", tag


def start():
    """Check on a daemon thread (at start-up, and when the About box asks again)."""
    with _lock:
        if _state["running"]:
            return
        _state["running"] = True

    def run():
        status, tag = check()
        with _lock:
            _state.update(status=status, tag=tag, running=False)
    threading.Thread(target=run, daemon=True).start()


def status():
    with _lock:
        return dict(_state)
