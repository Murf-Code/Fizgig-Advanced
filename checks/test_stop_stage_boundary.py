"""Stop racing a stage transition must be recorded, not ignored."""
from fizgig.web import server


class Exited:
    def poll(self):
        return 0


def test_stop_claims_a_run_after_subprocess_exit(monkeypatch):
    # The watcher has not advanced the stage yet; the old Popen is still stored.
    run = dict(server._run, state="running", proc=Exited(), ended_at=None)
    monkeypatch.setattr(server, "_run", run)
    assert server._stop() is True
    assert run["state"] == "stopped" and run["ended_at"] is not None
    assert server._stop() is False  # repeated Stop does not signal an exited PID


def test_stop_does_not_claim_a_completed_or_paused_run(monkeypatch):
    for state in ("completed", "paused", "stopped"):
        run = dict(server._run, state=state, proc=Exited(), ended_at=None)
        monkeypatch.setattr(server, "_run", run)
        assert server._stop() is False
        assert run["state"] == state


def test_stop_without_a_process_does_not_change_state(monkeypatch):
    run = dict(server._run, state="running", proc=None, ended_at=None)
    monkeypatch.setattr(server, "_run", run)
    assert server._stop() is False
    assert run["state"] == "running"
