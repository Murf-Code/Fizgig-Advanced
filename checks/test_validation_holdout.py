"""Validation hold-out: the fixed validation pictures never train, so validation measures pictures the LoRA hasn't
seen. A tiny CPU family trains for real (test_save_stop's), plus the form row and the launch flag."""
import json
import os
import types

from test_save_stop import _losses, _run, data, tiny  # noqa: F401 - the fixtures


def _val(out):
    with open(os.path.join(str(out), ".fizgig_loss_t.jsonl"), encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if '"k": "val"' in ln]


def test_held_out_pictures_never_train_but_are_checked(tiny, tmp_path, caplog, monkeypatch):
    T, cfg = tiny
    caplog.set_level("INFO")
    trained = set()
    from fizgig.training import val_loss
    real_offer = val_loss.FixedValidation.offer

    def offer(self, batch):                      # every training batch passes here: which pictures trained
        if not getattr(self, "_prefilling", False):
            trained.update(str(k) for k in batch.get("item_keys") or [])
        return real_offer(self, batch)
    real_prefill = val_loss.FixedValidation.prefill

    def prefill(self, *a, **k):
        self._prefilling = True
        try:
            return real_prefill(self, *a, **k)
        finally:
            self._prefilling = False
    monkeypatch.setattr(val_loss.FixedValidation, "offer", offer)
    monkeypatch.setattr(val_loss.FixedValidation, "prefill", prefill)
    out = tmp_path / "held"
    assert _run(T, cfg, out, validation_loss=True, validation_images=4, validation_holdout=True,
                max_train_epochs=2, save_state=False) == "done"
    log = _losses(out)
    assert len(log) == 2 * 2                                   # 4 of 8 pictures train: 2 steps an epoch, not 4
    assert "hold-out on: 4 validation picture(s) never train" in caplog.text
    assert len(trained) == 4
    held = {"p0", "p2", "p5", "p7"}                            # pick_keys: evenly through the sorted 8
    assert not {os.path.splitext(os.path.basename(k))[0] for k in trained} & held
    assert len(_val(out)) == 3                                 # the baseline and each epoch's check ran on them


def test_hold_out_never_takes_every_picture():
    from fizgig.training.val_loss import hold_out
    from fizgig.dataset.image_dataset import BucketBatchManager
    items = [types.SimpleNamespace(item_key=f"k{i}") for i in range(3)]
    ds = types.SimpleNamespace(batch_manager=BucketBatchManager({(64, 64): list(items)}, 2), num_train_items=3)
    group = types.SimpleNamespace(datasets=[ds], num_train_items=3)
    keys, held = hold_out(group, images=3)
    assert keys == ["k0", "k1", "k2"] and held == {} and group.num_train_items == 3
    keys, held = hold_out(group, picked=["k1"])
    assert set(held) == {"k1"} and group.num_train_items == 2 and len(ds.batch_manager) == 1


def test_form_row_and_launch_flag():
    from fizgig.families import form, launch
    from fizgig.families.registry import FAMILIES
    desc = FAMILIES["zimage"]
    rows = [r["key"] for s in form.form(desc)["sections"] for r in s["rows"]]
    assert "VALIDATION_HOLDOUT" in rows and form.DEFAULTS["VALIDATION_HOLDOUT"] is False
    assert "VALIDATION_HOLDOUT" in form.state(desc, {**form.DEFAULTS})["disabled"]          # needs validation on
    assert "VALIDATION_HOLDOUT" not in form.state(desc, {**form.DEFAULTS, "VALIDATION_LOSS": True})["disabled"]
    src = open(launch.__file__, encoding="utf-8").read()
    assert 'cmd.append("--validation_holdout")' in src
    from fizgig.families.train import setup_parser
    req = ["--family", "zimage", "--dit", "x", "--output_dir", "o", "--output_name", "n"]
    assert setup_parser().parse_known_args(req + ["--validation_holdout"])[0].validation_holdout is True
