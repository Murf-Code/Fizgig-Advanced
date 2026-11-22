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
                         {"run": None, "images": []})

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


if __name__ == "__main__":
    unittest.main()
