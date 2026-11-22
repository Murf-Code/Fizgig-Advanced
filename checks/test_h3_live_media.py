"""CPU-only H3 live-media cache tests, with actual VAE loads replaced by stubs."""
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from fizgig.families import loss_watch


def media_watch(tmp_path, filename, audio=True, still=False):
    root = tmp_path / "source"
    root.mkdir()
    source = root / filename
    source.write_bytes(b"original source")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    size = (128, 128) if filename.endswith(".wav") else (256, 256)
    key = filename.rsplit(".", 1)[0]
    item = SimpleNamespace(
        item_key=key, original_size=size, bucket_size=size,
        latent_cache_path=str(cache_dir / f"{key}_{size[0]:04d}x{size[1]:04d}_minimaxh3.safetensors"))
    initial = {"latent_1x8x8": torch.zeros(24, 1, 8, 8)}
    if audio:
        initial["audio_latent"] = torch.zeros(2, 32)
    if filename.endswith(".wav"):
        initial["audio_only"] = torch.tensor(True)
    if still:
        initial["still_latent"] = torch.zeros(24, 8, 8)
        initial["still_frame"] = torch.tensor(0)
    save_file(initial, item.latent_cache_path)
    ds = SimpleNamespace(item_prefix="", image_directory=str(root), has_control=False,
                         batch_manager=SimpleNamespace(buckets={size: [item]}))
    ds.get_latent_cache_path = lambda it: str(cache_dir / (
        f"{os.path.splitext(os.path.basename(it.item_key))[0]}_"
        f"{it.original_size[0]:04d}x{it.original_size[1]:04d}_minimaxh3.safetensors"))
    wav_model = tmp_path / "audio.vae"
    if audio:
        wav_model.write_bytes(b"audio model stub")
    driver = SimpleNamespace(
        description=SimpleNamespace(key="minimax", display_name="MiniMax H3", clip_spec=object()),
        options={"audio_vae": str(wav_model) if audio else "", "clip_still_as_photo": ""})
    watch = loss_watch.Watch.__new__(loss_watch.Watch)
    watch.group = SimpleNamespace(datasets=[ds])
    watch.driver = driver
    watch.vae_path = str(tmp_path / "video.vae")
    watch.removed_keys = {key}
    watch.unsupported_media_keys = {key}
    return watch, item, source


def test_voice_refresh_replaces_audio_cache(tmp_path, monkeypatch):
    watch, item, source = media_watch(tmp_path, "voice.wav")
    monkeypatch.setattr("fizgig.minimax.audio_vae.load_minimax_h3_audio_vae",
                        lambda *a, **k: object())

    def fake_encode(scratch, audio_vae):
        assert scratch.item_key == str(source)
        assert audio_vae is not None
        save_file({"latent_1x8x8": torch.ones(24, 1, 8, 8),
                   "audio_latent": torch.ones(2, 32), "audio_only": torch.tensor(True)},
                  scratch.latent_cache_path)

    monkeypatch.setattr("fizgig.minimax.caching._encode_audio_item", fake_encode)
    watch._refresh_h3_media("voice", item, str(source), torch.device("cpu"))
    got = load_file(item.latent_cache_path)
    assert torch.equal(got["audio_latent"], torch.ones(2, 32))
    assert "voice" not in watch.removed_keys
    assert "voice" not in watch.unsupported_media_keys


def test_voice_without_audio_vae_fails_closed(tmp_path):
    watch, item, source = media_watch(tmp_path, "voice.wav")
    watch.driver.options["audio_vae"] = ""
    with pytest.raises(ValueError, match="Audio VAE"):
        watch._refresh_h3_media("voice", item, str(source), torch.device("cpu"))
    assert "voice" in watch.removed_keys
    assert load_file(item.latent_cache_path)["audio_latent"].sum() == 0


def test_clip_refresh_preserves_still_cache(tmp_path, monkeypatch):
    watch, item, source = media_watch(tmp_path, "clip.mp4", audio=False, still=True)
    save_file({"x": torch.zeros(1)}, watch.vae_path)
    frames = np.zeros((17, 256, 256, 3), dtype=np.uint8)
    monkeypatch.setattr("fizgig.minimax.clip.read_frames", lambda p: frames)
    monkeypatch.setattr("fizgig.families.clips.validate", lambda *a, **k: None)

    class DummyVae:
        def load_state_dict(self, *a, **k): pass
        def to(self, *a): return self
        def eval(self): return self

    monkeypatch.setattr("fizgig.minimax.vae.MiniMaxH3VideoVAEEncoder", DummyVae)
    observed = []
    def fake_encode(vae, scratch, audio_vae, device, dtype, clip_still):
        observed.append(clip_still)
        save_file({"latent_2x16x16": torch.ones(24, 2, 16, 16),
                   "still_latent": torch.ones(24, 16, 16),
                   "still_frame": torch.tensor(3)}, scratch.latent_cache_path)
    monkeypatch.setattr("fizgig.minimax.caching._encode_clip", fake_encode)
    watch._refresh_h3_media("clip", item, str(source), torch.device("cpu"))
    assert observed == [True]
    assert "still_latent" in load_file(item.latent_cache_path)
    assert "clip" not in watch.removed_keys


def test_h3_live_text_uses_native_training_cache_format(tmp_path):
    from fizgig.families.loss_watch import _save_live_text
    item=SimpleNamespace(item_key="clip",caption="a cinematic scene",
        text_encoder_output_cache_path=str(tmp_path/"clip_minimaxh3_te.safetensors"))
    desc=SimpleNamespace(key="minimax",arch_id="minimaxh3")
    _save_live_text(desc,item,{"hidden_states":torch.ones(3,8)})
    tensors=load_file(item.text_encoder_output_cache_path)
    assert set(tensors)=={"hidden_states","attention_mask"}
    assert tensors["hidden_states"].shape==(3,8)
    assert tensors["attention_mask"].tolist()==[True,True,True]
