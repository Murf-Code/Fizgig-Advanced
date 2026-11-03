"""The Samples tab's 'Change the previews of the run going now': prompts changed while a run trains reach that run's
previews (the epoch's own and Sample now). A tiny CPU family trains for real (test_save_stop's toy driver); the
pictures aren't drawn - what the previews would be made from is recorded instead."""
import json
import os

import pytest

from test_save_stop import data, tiny, _run  # noqa: F401  (fixtures)


@pytest.fixture
def seen(tiny, monkeypatch):
    T, cfg = tiny
    got = []

    def fake_render(driver, dit, net, vae, conds, out_dir, epoch, **kw):
        got.append((epoch, [float(c["emb"][0]) for c in conds]))
        return []
    monkeypatch.setattr(T, "_render_previews", fake_render)
    return T, cfg, got


def _emb(prompt):
    return float(len(prompt) % 7)            # the toy driver's encoding of a prompt


def test_a_changed_prompt_reaches_the_next_epoch_preview(seen, tmp_path):
    T, cfg, got = seen
    out = tmp_path / "run"
    out.mkdir()
    old, new = "a cat", "a dog in the snow"
    (out / ".sample_override.json").write_text(json.dumps({"prompts": [new], "seed": 5, "width": 64, "height": 64}))
    _run(T, cfg, out, max_train_epochs=2, save_state=False, sample_prompts=[old], sample_every_n_epochs=1,
         vae_path="vae", te_path="te")
    assert got, "no previews were made"
    assert all(e == [_emb(new)] for _ep, e in got if _ep > 0), got


def _post(port, path, body):
    import urllib.request
    from fizgig.web import server
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "X-Fizgig-Token": server.TOKEN,
                                          "Host": f"127.0.0.1:{port}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:                       # an HTTP error's answer
        return json.loads(e.read()) if hasattr(e, "read") else {"error": str(e)}


def test_the_page_to_the_trainer(tmp_path, monkeypatch):
    """What the Samples tab sends (sampleLiveSoon's body) lands where the trainer reads it, for the run going now."""
    import subprocess
    import sys
    import threading
    from http.server import ThreadingHTTPServer
    from fizgig.families import train as T
    from fizgig.web import server
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        monkeypatch.setitem(server._run, "proc", proc)
        monkeypatch.setitem(server._run, "output_dir", str(tmp_path))
        monkeypatch.setitem(server._run, "family", "krea2")
        monkeypatch.setitem(server._run, "stage", "Training")
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        port = httpd.server_address[1]
        body = {"on": True, "prompts": "a dog in the snow\nsecond one", "seed": "1234", "width": "1024",
                "height": "1024", "steps": "8", "cfg": "1", "turbo": "1", "every": "1", "until": 0, "after": 0,
                "look": False, "checkpoint": False}
        r = _post(port, "/api/run/sample_override", body)
        assert r.get("on") is True, r
        ov = T._read_sample_override(str(tmp_path))
        assert ov and ov["prompts"] == ["a dog in the snow", "second one"], ov
        assert _post(port, "/api/run/now", {"what": "sample"}).get("ok"), "Sample now"
        httpd.shutdown()
    finally:
        proc.kill()


def test_a_change_mid_run_and_sample_now(seen, tmp_path, monkeypatch):
    """Prompts changed after the run's first previews (encoded once already): Sample now and the next epoch use them."""
    T, cfg, got = seen
    out = tmp_path / "run"
    out.mkdir()
    old, new = "a cat", "a dog in the snow"
    real = T._render_previews

    def render(*a, **kw):
        r = real(*a, **kw)
        if len(got) == 1:                      # after the first round: change the prompts, press Sample now
            (out / ".sample_override.json").write_text(json.dumps({"prompts": [new], "seed": 5}))
            open(out / ".sample_now", "w").close()
        return r
    monkeypatch.setattr(T, "_render_previews", render)
    _run(T, cfg, out, max_train_epochs=2, save_state=False, sample_prompts=[old], sample_every_n_epochs=1,
         vae_path="vae", te_path="te", sample_at_first=True)
    assert got[0][1] == [_emb(old)], got
    assert len(got) >= 3 and all(e == [_emb(new)] for _ep, e in got[1:]), got


def test_sample_now_sends_a_just_typed_prompt_first():
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src/fizgig/web/static/app.js")
    js = open(here, encoding="utf-8").read()
    assert "const sampleLiveSoon = debounce(sampleLiveSend, 700);" in js
    assert 'if (S.values.__SAMPLE_LIVE) await sampleLiveSend(); now("sample",' in js


def test_override_prompts_see_the_sample_picture_like_the_launch_ones():
    from fizgig.families import train as T

    class D:
        def park_for(self, *a):
            return None

        def load_text_encoder(self, p, d):
            return "te"

        def unload_text_encoder(self, te):
            pass

        def encode_text(self, te, p):
            return ["text"] * len(p)

        def encode_text_with_image(self, te, p, img):
            return ["image"] * len(p)
    assert T._encode_override(D(), "", ["a"], None, "cpu", parkable=False) == ["text"]
    assert T._encode_override(D(), "", ["a"], None, "cpu", parkable=False, image=object()) == ["image"]
