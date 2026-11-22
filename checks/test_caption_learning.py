"""Regression checks: only matching, log-owning runs decorate images."""
import os
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from fizgig.web import caption_learning


class LearningSnapshotTest(unittest.TestCase):
    def test_missing_folder(self):
        self.assertEqual(caption_learning.snapshot("", [], lambda _: None),
                         {"run": None, "images": [], "auto_enable": False})

    def test_matching_run_and_missing_image(self):
        with tempfile.TemporaryDirectory() as root:
            run = {"id": "one", "family": "krea2", "image_folder": root,
                   "output_dir": root, "lora": "LoRA"}
            other = {"id": "wrong", "family": "krea2",
                     "image_folder": root + "_other", "output_dir": root}
            data = {"images": [
                {"key": "a", "verdict": "stuck", "recent": .8, "multiplier": .7},
                {"key": "missing", "verdict": "easy", "recent": -.2, "multiplier": 1.1}],
                "report": {"epoch": 5}}
            with patch.object(caption_learning.image_loss, "owner_runs",
                              return_value={"one": True, "wrong": True}), \
                 patch.object(caption_learning.image_loss, "residuals",
                              return_value=data), \
                 patch.object(caption_learning.image_loss, "image_paths",
                              return_value={"a": os.path.join(root, "a.png")}):
                result = caption_learning.snapshot(root, [other, run],
                    lambda _: SimpleNamespace(loss_watch=True))
            self.assertEqual(result["run"]["epoch"], 5)
            self.assertEqual(len(result["images"]), 1)
            self.assertEqual(result["images"][0]["verdict"], "stuck")

            # Historical loss remains visible, but must not auto-enable colors.
            self.assertFalse(result["auto_enable"])

    def test_active_loss_watch_before_first_epoch(self):
        with tempfile.TemporaryDirectory() as root:
            run = {"id": "live", "family": "krea2", "image_folder": root,
                   "output_dir": root, "lora": "New LoRA", "result": "running",
                   "values": {"KREA2_LOSS_WATCH": True}}
            with patch.object(caption_learning.image_loss, "owner_runs",
                              return_value={}):
                data = caption_learning.snapshot(root, [run],
                    lambda _: SimpleNamespace(loss_watch=True), active_run_id="live")
            self.assertTrue(data["auto_enable"])
            self.assertEqual(data["run"]["id"], "live")
            self.assertEqual(data["images"], [])

    def test_finished_and_disabled_runs_never_auto_enable(self):
        with tempfile.TemporaryDirectory() as root:
            run = {"id": "old", "family": "krea2", "image_folder": root,
                   "output_dir": root, "lora": "Old LoRA",
                   "result": "finished", "values": {"KREA2_LOSS_WATCH": True}}
            with patch.object(caption_learning.image_loss, "owner_runs",
                              return_value={}):
                finished = caption_learning.snapshot(root, [run],
                    lambda _: SimpleNamespace(loss_watch=True), active_run_id="old")
                run["result"] = "running"
                run["values"] = {"KREA2_LOSS_WATCH": False}
                disabled = caption_learning.snapshot(root, [run],
                    lambda _: SimpleNamespace(loss_watch=True), active_run_id="old")
            self.assertFalse(finished["auto_enable"])
            self.assertFalse(disabled["auto_enable"])



if __name__ == "__main__":
    unittest.main()
