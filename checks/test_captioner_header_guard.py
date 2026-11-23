"""Regression for upstream Fizgig #388, adapted for Advanced's Caption Workbench.

A GGUF saved as a captioner must be rejected before reading its weights. A misleading
.safetensors extension must not allow a huge/beyond-EOF header read either.
"""
import builtins
import json
import struct

from fizgig.web.captions import captioner_problem


def test_gguf_captioner_rejected_without_opening_it(tmp_path, monkeypatch):
    file = tmp_path / "qwen-Q4_K_M.gguf"
    file.write_bytes(b"GGUF" + struct.pack("<I", 3) + b"X" * 64)
    real_open = builtins.open

    def checked_open(path, *args, **kwargs):
        if str(path) == str(file):
            raise AssertionError("the GGUF file should never be opened")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", checked_open)
    assert captioner_problem(file) == "it isn't a safetensors file"


def test_disguised_gguf_header_does_not_trigger_massive_read(tmp_path, monkeypatch):
    file = tmp_path / "looks-legit.safetensors"
    file.write_bytes(b"GGUF" + struct.pack("<I", 3) + b"X" * 64)
    real_open = builtins.open
    read_sizes = []

    class RecordReads:
        def __init__(self, fp):
            self.fp = fp

        def read(self, size=-1):
            read_sizes.append(size)
            if size > 1024:
                raise AssertionError("asked to read an unreasonably large header")
            return self.fp.read(size)

        def fileno(self):
            return self.fp.fileno()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self.fp.__exit__(*exc)

    def checked_open(path, *args, **kwargs):
        fp = real_open(path, *args, **kwargs)
        return RecordReads(fp) if str(path) == str(file) else fp

    monkeypatch.setattr(builtins, "open", checked_open)
    assert captioner_problem(file) == "the file can't be read"
    assert read_sizes == [8]


def test_valid_captioner_header_remains_accepted(tmp_path):
    file = tmp_path / "qwen.safetensors"
    header = {
        "visual.patch_embed.weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]},
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [1, 2560], "data_offsets": [8, 16]},
    }
    raw = json.dumps(header).encode("utf-8")
    file.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(16))
    assert captioner_problem(file) == ""
