"""Exercise the HTTP history route, not just image_edit.step directly."""
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from fizgig.web import caption_review, image_edit, server


def _post(path, body):
    # Call the actual authenticated request dispatcher with an in-memory
    # request, so branch-local imports and review locks are exercised too.
    request = SimpleNamespace(
        path=path, _host_ok=lambda: True, _api_ok=lambda: True,
        _body=lambda: body, json=lambda data, status=200: (data, status),
    )
    return server.Handler._post(request)


@pytest.mark.parametrize("operation", ["crop", "remove"])
def test_image_history_endpoint_restores_file_bytes(tmp_path, monkeypatch, operation):
    image = tmp_path / "a.png"
    mask = tmp_path / "a-masklabel.png"
    pixels = np.full((40, 50, 3), 80, dtype=np.uint8)
    pixels[12:20, 12:20] = 220
    Image.fromarray(pixels).save(image)
    Image.new("L", (50, 40), 255).save(mask)
    original = (image.read_bytes(), mask.read_bytes())
    queued = []
    monkeypatch.setattr(server, "_notify_image_change",
                        lambda folder, name, action: queued.append((folder, name, action)) or True)
    monkeypatch.setattr(server, "_gpu_training", lambda: False)

    def fill(rgb, hole, gpu=True):
        out = rgb.copy()
        out[hole] = 80
        return out

    monkeypatch.setattr(image_edit, "lama_fill", fill)
    body = {"folder": str(tmp_path), "name": image.name}
    if operation == "crop":
        body.update(left=5, top=5, right=35, bottom=30)
    else:
        import base64
        stroke = np.zeros((40, 50), dtype=np.uint8)
        stroke[11:21, 11:21] = 255
        buf = io.BytesIO()
        Image.fromarray(stroke).save(buf, "PNG")
        body["png"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    result, status = _post("/api/masks/edit/" + operation, body)
    assert status == 200, result
    edited = (image.read_bytes(), mask.read_bytes())
    assert edited[0] != original[0]
    step = result["step"]
    # Review locks must still apply to undo and redo.
    caption_review.mark(tmp_path, image.name, True)
    result, status = _post("/api/masks/edit/image_step", {"step": step, "redo": False})
    assert status == 400 and "locked" in result["error"].lower()
    assert (image.read_bytes(), mask.read_bytes()) == edited
    caption_review.mark(tmp_path, image.name, False)
    result, status = _post("/api/masks/edit/image_step", {"step": step, "redo": False})
    assert status == 200, result
    assert (image.read_bytes(), mask.read_bytes()) == original
    result, status = _post("/api/masks/edit/image_step", {"step": step, "redo": True})
    assert status == 200, result
    assert (image.read_bytes(), mask.read_bytes()) == edited
    assert sum(name == image.name and action == "updated" for _, name, action in queued) == 3
