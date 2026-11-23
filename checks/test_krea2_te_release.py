"""Krea 2 encoder teardown must not copy its weights into RAM before dropping it.
Upstream Fizgig #380; adapted for Advanced Text Fusion and inference variants.
"""
import pytest

from fizgig.krea2.driver import Krea2Driver


class _Encoder:
    def __init__(self, fail_meta=False):
        self.device_moves = []
        self.fail_meta = fail_meta

    def to(self, destination):
        self.device_moves.append(destination)
        if destination == "meta" and self.fail_meta:
            raise RuntimeError("no meta move")
        return self


def test_te_release_uses_meta_instead_of_cpu_copy(monkeypatch):
    from fizgig.krea2 import driver as module
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    d = Krea2Driver.__new__(Krea2Driver)
    enc = _Encoder()
    d.unload_text_encoder(enc)
    assert enc.device_moves == ["meta"]


def test_te_release_falls_back_when_meta_is_unavailable(monkeypatch):
    from fizgig.krea2 import driver as module
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    d = Krea2Driver.__new__(Krea2Driver)
    enc = _Encoder(fail_meta=True)
    d.unload_text_encoder(enc)
    assert enc.device_moves == ["meta", "cpu"]


def test_file_backed_te_release_does_not_move_storage(monkeypatch):
    from fizgig.krea2 import driver as module
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    d = Krea2Driver.__new__(Krea2Driver)
    d._file_backed_inference = True
    enc = _Encoder()
    d.unload_text_encoder(enc)
    assert enc.device_moves == []
