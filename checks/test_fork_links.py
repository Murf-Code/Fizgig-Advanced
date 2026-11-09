"""Fizgig Advanced's own links and installs: the clone, the pod's default repo and branch, the release check on the
fork's advanced-* tags (never upstream's v9.x), and the app's guide links."""
import os
import re

from fizgig.web import pod_version, update_check

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORK = "https://github.com/Murf-Code/Fizgig-Advanced"


def _read(p):
    return open(os.path.join(ROOT, p), encoding="utf-8").read()


def test_release_check_reads_the_forks_tags_only(monkeypatch):
    assert update_check.release_key("advanced-1.3.1") == (1, 3, 1) and update_check.release_key("advanced-1") == (1,)
    assert update_check.release_key("v9.2.5") is None and update_check.release_key("advanced-x") is None
    asked = []

    def git(*args, timeout=8):
        asked.append(args)
        return 0, ("a1\trefs/tags/advanced-1.3.1\na2\trefs/tags/advanced-1.4\na3\trefs/tags/advanced-1\n"
                   "a4\trefs/tags/v9.2.5\n")
    monkeypatch.setattr(update_check, "_git", git)
    assert update_check._latest_release_tag() == ("advanced-1.4", "a2")
    assert asked[0][-1] == "advanced-*"                         # the fork's tags are what's asked for
    monkeypatch.setattr(update_check, "_git", lambda *a, **k: (128, ""))        # offline: silent, no nag
    assert update_check.check() == ("unknown", "")


def test_install_and_pod_point_at_the_fork(monkeypatch):
    for doc in ("README.md", "docs/INSTALL.md"):
        assert "git clone -b fizgig-advanced https://github.com/Murf-Code/Fizgig-Advanced.git\ncd Fizgig-Advanced" \
            in _read(doc)
    assert "git remote add origin https://github.com/Murf-Code/Fizgig-Advanced.git" in _read("docs/INSTALL.md")
    sh = _read("docker/entrypoint.sh")
    assert 'REPO_URL="${FIZGIG_REPO:-https://github.com/Murf-Code/Fizgig-Advanced.git}"' in sh
    assert 'REPO_REF="${FIZGIG_REF:-fizgig-advanced}"' in sh
    assert "FIZGIG_REPO=https://github.com/Murf-Code/Fizgig-Advanced.git" in _read("README.md")   # the template note
    monkeypatch.setattr(pod_version, "on_pod", lambda: True)
    for ref, follows in (("advanced-1.4", False), ("v9.2.5", False), ("fizgig-advanced", True), ("", True)):
        monkeypatch.setenv("FIZGIG_REF", ref)
        assert pod_version.allowed() is follows, ref


def test_app_links_go_to_the_fork():
    for p in ("src/fizgig/web/static/index.html", "src/fizgig/web/static/app.js", "src/fizgig/web/news.py",
              "src/fizgig/families/form.py", ".github/CONTRIBUTING.md"):
        s = _read(p)
        assert not re.search(r"shootthesound/Fizgig/(blob|tree)/master|shootthesound/Fizgig/(issues|discussions)"
                             r"|shootthesound/Fizgig/releases/latest", s), p
    for link in re.findall(re.escape(FORK) + r"/blob/fizgig-advanced/([^\"#)\s]+)", _read(
            "src/fizgig/web/static/index.html") + _read("src/fizgig/web/news.py") + _read("src/fizgig/families/form.py")):
        assert os.path.exists(os.path.join(ROOT, link)), link           # every guide it links is in the fork
