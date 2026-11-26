"""CPU-only checks for the fixed numerical diagnostic queue settings."""
import unittest
from fizgig.web.polora_diagnostic import settings

class DiagnosticSettingsTest(unittest.TestCase):
    def test_controls(self):
        cases = settings("test", "/tmp")
        self.assertEqual(len(cases), 7)
        self.assertEqual(len({v["LORA_OUTPUT_DIR"] for v in cases}), 7)
        self.assertEqual(len({v["LORA_NAME"] for v in cases}), 7)
        for v in cases:
            self.assertEqual(v["MAX_TRAIN_EPOCHS"], "1")
            self.assertEqual(v["SEED"], "42")
            self.assertEqual(v["NETWORK_DIM"], "8")
            self.assertEqual(v["DATASET_MEGAPIXELS"], "1.0")
            self.assertFalse(v["SAVE_STATE"])
            self.assertFalse(v["SAVE_STATE_ON_TRAIN_END"])
            self.assertFalse(v["SAMPLE_ENABLED"])
        a, b, c, d, e, f, g = cases
        self.assertEqual([v["COMPILE_BLOCKS"] for v in cases[:4]], ["outside", "outside", "Off", "Off"])
        self.assertEqual([v["POLORA_COMPILE"] for v in cases[:4]], [True, False, True, False])
        self.assertEqual([v["MASKED_TRAINING"] for v in cases], [True, True, True, True, False, True, True])
        self.assertEqual(f["OPTIMIZER_TYPE"], "adamw8bit")
        self.assertFalse(f["ADAPTIVE_LR"])
        self.assertEqual(g["FAMILY_PRECISION"], "bf16 (full precision)")
        self.assertEqual(g["BLOCKS_SWAP"], "Auto (detect from GPU)")
        for v in (a, b, c, d, e, g):
            self.assertIn(f"compile={v['POLORA_COMPILE']}", v["OPTIMIZER_ARGS"])
            self.assertEqual(v["LEARNING_RATE"], "0.02")

if __name__ == "__main__":
    unittest.main()
