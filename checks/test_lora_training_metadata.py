"""Full LoRA save metadata: kohya's ss_* training keys and one fizgig_training_config JSON. Names and numbers only:
no captions or tags, no full paths (folder and model names), never a token or key; the existing keys stay."""
import json
import types

from fizgig.training import metadata as M


def _ds(directory, count, repeats=1, weight=1.0, reg=False, buckets=((512, 512),)):
    return types.SimpleNamespace(image_directory=directory, num_repeats=repeats, loss_weight=weight, is_reg=reg,
                                 get_total_image_count=lambda: count,
                                 batch_manager=types.SimpleNamespace(bucket_resos=list(buckets)))


SECRET = "hf_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"


def _md(tmp_path):
    main = str(tmp_path / "data" / "zebra subject")
    sub = "H:\\Stable_Diffusion\\training\\zebra subject\\Close ups"
    reg = str(tmp_path / "reg")
    dss = [_ds(main, 20, repeats=2, weight=0.5, buckets=((768, 1024), (1024, 768))),
           _ds(sub, 5, repeats=3), _ds(reg, 10, reg=True)]
    return M.training_metadata(
        seed=7, batch_size=1, grad_accum=2, epochs=10, epoch=4, steps=200, steps_per_epoch=50, max_steps=500,
        num_images=75, scheduler="cosine", warmup_steps=10, resolution=[1024, 1024],
        buckets=M.bucket_resolutions(dss), base_precision="int8",
        base_model="H:\\Stable_Diffusion\\models\\krea2_dev.safetensors", datasets=dss,
        training_method="global_replay", selective_block_check=False, validation_groups=3, best_epoch=3,
        best_val_loss=0.1234, current_lr=1e-4, started_at=1000.0, now=1600.0) , (main, sub, reg)


def test_kohya_keys_and_training_config(tmp_path):
    md, _ = _md(tmp_path)
    for k in ("ss_seed", "ss_batch_size_per_device", "ss_num_epochs", "ss_steps", "ss_max_train_steps",
              "ss_lr_scheduler", "ss_resolution", "ss_mixed_precision", "ss_sd_model_name", "ss_dataset_dirs",
              "ss_reg_dataset_dirs", "ss_bucket_info", "ss_training_started_at", "ss_training_finished_at",
              M.TRAINING_CONFIG_KEY):
        assert k in md and isinstance(md[k], str), k
    assert md["ss_seed"] == "7" and md["ss_resolution"] == "(1024, 1024)" and md["ss_lr_scheduler"] == "cosine"
    assert md["ss_sd_model_name"] == "krea2_dev.safetensors"
    dirs = json.loads(md["ss_dataset_dirs"])
    assert dirs == {"zebra subject": {"n_repeats": 2, "img_count": 20}, "Close ups": {"n_repeats": 3, "img_count": 5}}
    assert json.loads(md["ss_reg_dataset_dirs"]) == {"reg": {"n_repeats": 1, "img_count": 10}}
    cfg = json.loads(md[M.TRAINING_CONFIG_KEY])
    assert cfg["training_method"] == "global_replay" and cfg["selective_block_check"] is False
    assert cfg["folders"]["zebra subject"] == {"weight": 0.5, "repeats": 2}
    assert cfg["validation_groups"] == 3 and cfg["epoch"] == 4 and cfg["best_epoch"] == 3
    assert cfg["best_val_loss"] == 0.1234 and cfg["current_lr"] == 1e-4 and cfg["training_seconds"] == 600.0
    assert cfg["trained_resolutions"][:2] == ["768x1024", "1024x768"] and "fizgig_version" in cfg


def test_no_paths_tokens_or_captions_leak(tmp_path):
    md, (main, sub, reg) = _md(tmp_path)
    blob = json.dumps(md)
    for bad in (str(tmp_path), "Stable_Diffusion", "H:\\\\", "/data/", "models"):
        assert bad not in blob, bad
    assert "a photo of zebra" not in blob                      # nothing reads captions at all
    md2 = M.training_metadata(base_model=f"/x/{SECRET}.safetensors", training_method=SECRET,
                              datasets=[_ds(f"/home/u/{SECRET}", 1)])
    assert SECRET not in json.dumps(md2)


def test_existing_build_metadata_is_unchanged():
    md = M.build_metadata(None, "krea2", 0, title="t", trigger_phrase="zx")
    assert md["modelspec.architecture"] == "Krea-2/lora" and md["modelspec.title"] == "t"
    assert M.TRAINING_CONFIG_KEY not in md
