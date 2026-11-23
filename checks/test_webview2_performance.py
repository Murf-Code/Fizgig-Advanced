"""Desktop WebView2 browser-like flag defaults and hook regression tests.

No pywebview/torch import: test safe on Linux GitHub Actions runners.
"""
import importlib.util
from pathlib import Path
import ast
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/fizgig/web/webview2_performance.py"
spec = importlib.util.spec_from_file_location("webview2_perf_test", MODULE)
perf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perf)


class WebView2PerformanceTests(unittest.TestCase):
    def test_default_matches_browser_acceleration(self):
        self.assertEqual(perf.browser_arguments({}, ""), "")
        self.assertEqual(perf.browser_arguments({"qt_disable_gpu": "0"}, ""), "")
        self.assertIn("No known", perf.acceleration_warning(""))

    def test_gpu_enabled_by_default_on_windows_and_linux(self):
        self.assertFalse(perf.gpu_disabled({}, "win32"))
        self.assertFalse(perf.gpu_disabled({"qt_disable_gpu": ""}, "win32"))
        self.assertFalse(perf.gpu_disabled({}, "linux"))
        self.assertTrue(perf.gpu_disabled({"qt_disable_gpu": "1"}, "linux"))
        self.assertTrue(perf.gpu_disabled({"qt_disable_gpu": "1"}, "win32"))
        self.assertFalse(perf.gpu_disabled({"qt_disable_gpu": "0"}, "win32"))
        self.assertEqual(perf.browser_arguments({}, "", platform="win32"), "")
        self.assertEqual(perf.browser_arguments({"qt_disable_gpu": "0"}, "",
                                                 platform="win32"), "")
        self.assertEqual(perf.browser_arguments({}, "", platform="linux"), "")
        self.assertEqual(perf.browser_arguments({"qt_disable_gpu": "1"},
                                                 "--disable-gpu", platform="win32"),
                         "--disable-gpu")

    def test_user_gpu_disable_remains_explicit(self):
        self.assertEqual(
            perf.browser_arguments({"qt_disable_gpu": "1"}, ""),
            "--disable-gpu")
        self.assertIn("GPU acceleration DISABLED",
                      perf.acceleration_warning("--disable-gpu"))
        self.assertEqual(perf.browser_arguments({"qt_disable_gpu": "0"},
                                                "--disable-gpu"), "--disable-gpu")

    def test_video_compatibility_is_opt_in(self):
        self.assertNotIn("--disable-accelerated-video-decode",
                         perf.browser_arguments({}, ""))
        self.assertIn("--disable-accelerated-video-decode",
                      perf.browser_arguments({}, "", software_video=True))
        inherited = "--window-size=1280,900 --disable-accelerated-video-decode"
        self.assertEqual(perf.browser_arguments({}, inherited, software_video=True),
                         inherited)

    def test_inherited_flags_preserved_without_duplicates(self):
        args = '--user-agent="Fizgig test" --disable-gpu'
        self.assertEqual(perf.browser_arguments({"qt_disable_gpu": "1"}, args), args)
        self.assertEqual(perf.browser_arguments({"qt_disable_gpu": "1"},
                                               "--disable-gpu-compositing"),
                         "--disable-gpu-compositing --disable-gpu")

    def test_window_still_attempts_to_remove_resource_callbacks(self):
        app = (ROOT / "src/fizgig/web/app_window.py").read_text(encoding="utf-8")
        ast.parse(app)
        self.assertIn("core.WebResourceResponseReceived -= self.on_web_resource_response", app)
        self.assertIn("core.WebResourceRequested -= self.on_web_resource_request", app)
        self.assertIn("WebView2 event hook removal:", app)
        self.assertIn("core.ProcessFailed += on_process_failed", app)
        self.assertIn("self._fizgig_process_failed_handler = on_process_failed", app)
        self.assertNotIn('setdefault("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"', app)


if __name__ == "__main__":
    unittest.main()
