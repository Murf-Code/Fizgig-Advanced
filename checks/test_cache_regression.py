"""CPU tests of the diagnostic planner and filesystem isolation (no model needed)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import toml
from fizgig.web import cache_regression as cr


class CacheRegressionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.image = self.root / "a.png"
        self.image.write_bytes(b"image")
        self.image.with_suffix(".txt").write_text("trigger, cat, black fur")
        self.body = toml.dumps({"general": {"resolution": [992, 992]},
                                "datasets": [{"image_directory": str(self.root), "cache_directory": str(self.source)}]})
        self.configs = [dict(__cache_regression=k, __cache_prompt="trigger, cat",
                             LORA_OUTPUT_DIR=str(self.root / "group" / k)) for k in cr.CASES]
        self.names = ["a_0992x0992_krea2drv.safetensors", "a_krea2drv_te.safetensors", "a_krea2drv_tags.safetensors"]
        for n in self.names:
            (self.source / n).write_bytes(n.encode())
        self.fake_modules = {
            "fizgig.dataset.image_dataset": NS(glob_images=lambda *a, **kw: [str(self.image)],
                BucketSelector=lambda *a: NS(get_bucket_resolution=lambda wh: wh),
                ImageDataset=NS(latent_cache_matches_reso=lambda *a: True)),
            "fizgig.families.cache_preflight": NS(_header=lambda p: (992, 992, "stamp")),
            "fizgig.dataset.caption_file": NS(read_caption=lambda p: Path(p).read_text()),
            "fizgig.dataset.tag_variants": NS(TagSettings=NS(from_params=lambda **kw: NS(caption_dropout=0)),
                variant_captions=lambda *a: ["variant"], variants_path=lambda p: p.replace("_te.safetensors", "_tags.safetensors")),
        }

    def make_plan(self, values):
        config = str(self.root / "dataset.toml")
        plan = NS(problems=[], stages=[NS(name="Training", cmd=["train"])], dataset_toml=self.body,
                  files=[(config, self.body)], dirs=[], console=[])
        launch = NS(Stage=lambda name, cmd: NS(name=name, cmd=cmd),
                    cache_command=lambda desc, inputs, stage: ["cache", "--model", stage, "--skip_existing"])
        cr.configure_plan(NS(key="krea2"), dict(values, DATASET_CONFIG=config), plan, values, launch)
        return plan

    def snapshot(self):
        with patch.dict(sys.modules, self.fake_modules):
            return cr.prepare_caches(self.body, self.configs, "krea2drv")

    def test_mask_and_gpu_recipe_fixed_across_arms(self):
        for gpu, masks, swap, compile_mode in [("4070-ti", False, "24", "Off"),
                                                ("3090", True, "0", "outside"),
                                                ("3090", False, "0", "outside"),
                                                ("4070-ti", True, "24", "Off")]:
            arms = cr.settings("test", str(self.root), "prompt", masked=masks, gpu_profile=gpu)
            self.assertEqual(len(arms), 4)
            for arm in arms:
                self.assertEqual(arm["MASKED_TRAINING"], masks)
                self.assertEqual(arm["BLOCKS_SWAP"], swap)
                self.assertEqual(arm["COMPILE_BLOCKS"], compile_mode)
                self.assertEqual(arm["__cache_gpu_profile"], gpu)

    def test_unknown_gpu_profile_refused(self):
        with self.assertRaises(ValueError):
            cr.settings("test", str(self.root), "prompt", gpu_profile="auto")

    def test_four_stage_plans(self):
        expected = [["Training"], ["Text Encoder Caching", "Training"],
                    ["Cache Preparation", "Training"], ["Cache Preparation", "Text Encoder Caching", "Training"]]
        for v, names in zip(self.configs, expected):
            p = self.make_plan(v)
            self.assertFalse(p.problems)
            self.assertEqual([s.name for s in p.stages], names)
            self.assertTrue(all("--skip_existing" not in s.cmd for s in p.stages))
            self.assertIn("--sample_at_first", p.stages[-1].cmd)
            self.assertEqual(toml.loads(p.dataset_toml)["datasets"][0]["cache_directory"],
                             str(Path(v["LORA_OUTPUT_DIR"]) / "isolated-cache"))
            self.assertEqual(toml.loads(p.dataset_toml)["datasets"][0]["image_directory"], str(self.root))

    def test_cache_copies_exact_and_isolated(self):
        manifest = self.snapshot()
        self.assertEqual(len(manifest["files"]), 3)
        for v, names in zip(self.configs, [self.names, self.names[:1], self.names[1:], []]):
            dest = Path(v["LORA_OUTPUT_DIR"]) / "isolated-cache"
            self.assertEqual(set(p.name for p in dest.iterdir()), set(names))
            for n in names:
                self.assertEqual((dest / n).read_bytes(), (self.source / n).read_bytes())
        b = Path(self.configs[1]["LORA_OUTPUT_DIR"]) / "isolated-cache" / self.names[0]
        b.write_bytes(b"different")
        self.assertEqual((self.source / self.names[0]).read_bytes(), self.names[0].encode())
        self.assertEqual((Path(self.configs[0]["LORA_OUTPUT_DIR"]) / "isolated-cache" / self.names[0]).read_bytes(), self.names[0].encode())

    def test_missing_variants_rejected(self):
        (self.source / self.names[2]).unlink()
        with self.assertRaisesRegex(ValueError, "variants missing"):
            self.snapshot()
        self.assertFalse((self.root / "group").exists())

    def test_missing_latents_rejected(self):
        (self.source / self.names[0]).unlink()
        with self.assertRaisesRegex(ValueError, "Existing cache missing"):
            self.snapshot()

    def test_verify_start_refuses_tamper_and_reuse(self):
        snap = self.snapshot()
        v = self.configs[1]
        p = self.make_plan(v)
        out = Path(v["LORA_OUTPUT_DIR"])
        recipe = {"case": v["__cache_regression"], "dataset_toml": p.dataset_toml,
                  "stages": [{"name": s.name, "cmd": s.cmd} for s in p.stages]}
        (out.parent / "diagnostic-settings.json").write_text(json.dumps({"plans": [recipe], "snapshot": snap}))
        cr.verify_start(p, v)
        retained = out / "isolated-cache" / self.names[0]
        retained.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            cr.verify_start(p, v)
        retained.write_bytes(self.names[0].encode())
        p.stages[-1].cmd.append("--changed")
        with self.assertRaisesRegex(ValueError, "plan changed"):
            cr.verify_start(p, v)
        p.stages[-1].cmd.pop()
        (out / "diagnostic.log").touch()
        with self.assertRaisesRegex(ValueError, "already run"):
            cr.verify_start(p, v)

    def test_held_cache_preserved_without_training_variants(self):
        held = self.source / "_fizgig_holdout"
        held.mkdir()
        for name in self.names[:2]:
            (self.source / name).rename(held / name)
        (self.source / self.names[2]).unlink()
        with patch.dict(sys.modules, self.fake_modules):
            snap = cr.prepare_caches(self.body, self.configs, "krea2drv", [str(self.image)])
        self.assertEqual(snap["holdout"], [str(self.image)])
        self.assertEqual({r["file"] for r in snap["files"]},
                         {"_fizgig_holdout/" + name for name in self.names[:2]})
        for v, names in zip(self.configs, [self.names[:2], self.names[:1], self.names[1:2], []]):
            dest = Path(v["LORA_OUTPUT_DIR"]) / "isolated-cache"
            self.assertEqual({p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()},
                             {"_fizgig_holdout/" + name for name in names})

    def test_no_pairs_or_multi_dataset(self):
        cfg = toml.loads(self.body)
        cfg["datasets"][0]["control_directory"] = "pairs"
        with self.assertRaises(ValueError):
            cr.source_config(toml.dumps(cfg))
        cfg["datasets"][0].pop("control_directory")
        cfg["datasets"].append(cfg["datasets"][0].copy())
        with self.assertRaises(ValueError):
            cr.source_config(toml.dumps(cfg))


if __name__ == "__main__":
    unittest.main()
