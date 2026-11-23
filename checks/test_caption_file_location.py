"""Regression tests for the local-only Caption Workbench file-manager action."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fizgig.web import file_location


class FileLocationTests(unittest.TestCase):
    def test_windows_selects_file_without_shell(self):
        self.assertEqual(
            file_location.file_manager_command(r"C:\images\blue jay.png", os_name="nt", platform="win32"),
            ["explorer.exe", "/select,", r"C:\images\blue jay.png"],
        )

    def test_macos_reveals_file(self):
        self.assertEqual(
            file_location.file_manager_command("/images/blue jay.png", os_name="posix", platform="darwin"),
            ["open", "-R", "/images/blue jay.png"],
        )

    def test_linux_opens_containing_folder(self):
        self.assertEqual(
            file_location.file_manager_command("/images/blue jay.png", os_name="posix", platform="linux"),
            ["xdg-open", "/images"],
        )

    def test_reveals_existing_file_and_never_uses_shell(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder, "blue jay.png")
            image.write_bytes(b"image")
            with patch.dict(os.environ, {"DISPLAY": ":99"}), patch.object(
                file_location.subprocess, "Popen"
            ) as popen:
                self.assertEqual(
                    file_location.reveal_file(str(image), folder, image.name), {"ok": True}
                )
                self.assertEqual(popen.call_count, 1)
                self.assertNotIn("shell", popen.call_args.kwargs)

    def test_rejects_wrong_folder_and_traversal(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder, "blue jay.png")
            image.write_bytes(b"image")
            with patch.object(file_location.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(ValueError, "does not match"):
                    file_location.reveal_file(str(image), os.path.dirname(folder), image.name)
                with self.assertRaisesRegex(ValueError, "Invalid file name"):
                    file_location.reveal_file(str(image), folder, "../blue jay.png")
                popen.assert_not_called()

    def test_missing_file_does_not_open_explorer(self):
        with tempfile.TemporaryDirectory() as folder:
            image = str(Path(folder, "missing.png"))
            with patch.object(file_location.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(ValueError, "no longer exists"):
                    file_location.reveal_file(image, folder, "missing.png")
                popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
