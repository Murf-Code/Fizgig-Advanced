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
        plan = NS(problems=[], stages=[NS(name="Training", cmd=["train", "--max_train_epochs", "2",
                    "--optimizer_type", "polora", "--output_dir", values["LORA_OUTPUT_DIR"],
                    "--dataset_config", config])], dataset_toml=self.body,
                  files=[(config, self.body)], dirs=[], console=[])
        launch = NS(Stage=lambda name, cmd: NS(name=name, cmd=cmd),
                    cache_command=lambda desc, inputs, stage: ["cache", "--model", stage, "--dataset_config", inputs["DATASET_CONFIG"], "--skip_existing"])
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

    def test_4070_inherits_dataset_only_and_3090_ignores_it(self):
        saved = {"DATASET_MEGAPIXELS": "0.5", "ENABLE_BUCKET": False,
                 "BUCKET_NO_UPSCALE": False, "DATASET_CAPTION_EXT": ".caption",
                 "TAG_SHUFFLE": False, "TAG_VARIANTS": "2", "TAG_DROPOUT": "5",
                 "TAG_VARIANT_CAP": "6", "CAPTION_DROPOUT": "0.05",
                 "BLOCKS_SWAP": "0", "LEARNING_RATE": "9", "MASKED_TRAINING": True}
        for arm in cr.settings("test", str(self.root), "prompt", gpu_profile="4070-ti",
                               dataset_values=saved, masked=False):
            for key in cr.DATASET_KEYS:
                if key in saved:
                    self.assertEqual(arm[key], saved[key])
            self.assertEqual(arm["BLOCKS_SWAP"], "24")
            self.assertEqual(arm["LEARNING_RATE"], "0.02")
            self.assertFalse(arm["MASKED_TRAINING"])
        for arm in cr.settings("test", str(self.root), "prompt", gpu_profile="3090",
                               dataset_values=saved, masked=True):
            self.assertEqual(arm["DATASET_MEGAPIXELS"], "1.0")
            self.assertTrue(arm["ENABLE_BUCKET"])
            self.assertTrue(arm["TAG_SHUFFLE"])
            self.assertEqual(arm["TAG_VARIANTS"], "4")
            self.assertEqual(arm["BLOCKS_SWAP"], "0")
            self.assertTrue(arm["MASKED_TRAINING"])

    def test_unknown_gpu_profile_refused(self):
        with self.assertRaises(ValueError):
            cr.settings("test", str(self.root), "prompt", gpu_profile="auto")

    def test_normal_training_recipe_refused(self):
        config = str(self.root / "dataset.toml")
        v = self.configs[0]
        for flag, val in [("--max_train_epochs", "30"), ("--optimizer_type", "muon")]:
            cmd = ["train", "--max_train_epochs", "2", "--optimizer_type", "polora",
                   "--output_dir", v["LORA_OUTPUT_DIR"], "--dataset_config", config]
            cmd[cmd.index(flag) + 1] = val
            plan = NS(problems=[], stages=[NS(name="Training", cmd=cmd)], dataset_toml=self.body,
                      files=[(config, self.body)], dirs=[], console=[])
            cr.configure_plan(NS(key="krea2"), dict(v, DATASET_CONFIG=config), plan, v, NS())
            self.assertIn(flag, plan.problems[0])
            self.assertEqual(plan.dataset_toml, self.body)

    def test_private_baseline_plan_and_snapshot(self):
        for gpu, fresh in [("4070-ti", False), ("3090", True)]:
            arms = cr.settings("test-" + gpu, str(self.root), "prompt",
                               gpu_profile=gpu, fresh_baseline=fresh)
            first = self.make_plan(arms[0])
            self.assertFalse(first.problems)
            self.assertEqual([s.name for s in first.stages],
                             ["Baseline Latent Caching", "Baseline Text Caching", "Training"])
            out = Path(arms[0]["LORA_OUTPUT_DIR"])
            baseline = out.parent / "baseline-cache"
            body = next(body for path, body in first.files if str(path).endswith("baseline-dataset.toml"))
            self.assertEqual(toml.loads(body)["datasets"][0]["cache_directory"], str(baseline.resolve()))
            self.assertEqual(toml.loads(first.dataset_toml)["datasets"][0]["cache_directory"],
                             str((out / "isolated-cache").resolve()))
            self.assertEqual(first.stages[0].cmd[first.stages[0].cmd.index("--dataset_config") + 1],
                             str(out.parent / "baseline-dataset.toml"))
            self.assertFalse((out / "isolated-cache").exists())
            baseline.mkdir(parents=True)
            (out.parent / "baseline-dataset.toml").write_text(body, encoding="utf-8")
            manifest_path = out.parent / "diagnostic-settings.json"
            manifest_path.write_text(json.dumps({"gpu_profile": gpu, "cases": arms, "snapshot": {
                "baseline_pending": True, "baseline_config": body, "arch": "krea2drv",
                "holdout": [], "files": []}}), encoding="utf-8")
            for n in self.names:
                (baseline / n).write_bytes(n.encode())
            with patch.dict(sys.modules, self.fake_modules):
                self.assertTrue(cr.finish_baseline(out))
            snap = json.loads(manifest_path.read_text())["snapshot"]
            self.assertEqual(snap["provenance"], "fresh_private_baseline")
            self.assertFalse(cr.finish_baseline(out))
            self.assertEqual(len(snap["files"]), 3)
            for v, names in zip(arms, [self.names, self.names[:1], self.names[1:], []]):
                cache = Path(v["LORA_OUTPUT_DIR"]) / "isolated-cache"
                self.assertEqual({p.name for p in cache.iterdir()}, set(names))

    def test_3090_defaults_to_original_cache(self):
        arms = cr.settings("test", str(self.root), "prompt", gpu_profile="3090")
        self.assertFalse(arms[0]["__cache_fresh_baseline"])
        self.assertEqual([s.name for s in self.make_plan(arms[0]).stages], ["Training"])

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
