"""CPU tests: cache audit distinguishes masked padding, NaN replacement and bad real tokens."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from fizgig.dataset import cache_audit, compact_text


class CacheAuditTests(unittest.TestCase):
    def test_real_token_nan_becomes_zero_and_padding_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "audit.jsonl"
            path = Path(tmp) / "text.safetensors"
            cache_audit.enable(report)
            try:
                cond = {"hidden_states": torch.tensor([[float("nan"), 0.], [2., 3.], [0., 0.]]),
                        "attention_mask": torch.tensor([True, True, False])}
                cache_audit.record("before_clean", "variant", "test caption", cond)
                cond["hidden_states"] = torch.nan_to_num(cond["hidden_states"])
                compact_text.save({"v0__" + k: v for k, v in cond.items()}, str(path),
                                  {"fizgig_tags": "1"})
                cache_audit.finish()
                rows = [json.loads(line) for line in report.read_text().splitlines()]
                self.assertEqual(rows[0]["valid_nan"], 1)
                self.assertEqual(rows[0]["all_zero_rows"], 0)
                self.assertEqual(rows[1]["valid_nan"], 0)
                self.assertEqual(rows[1]["all_zero_rows"], 1)
                self.assertEqual(rows[1]["valid_zero"], 2)
                self.assertEqual(rows[1]["valid_tokens"], 2)
                self.assertEqual(rows[-1]["summary"]["readback/variant"]["embeddings"], 1)
                self.assertEqual(rows[-1]["summary"]["compact_roundtrip"]["mismatches"], 0)
            finally:
                cache_audit._path = None
                cache_audit._totals.clear()


if __name__ == "__main__":
    unittest.main()
