"""Offscreen native Qt image-widget regression checks (no GPU, no network)."""
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from fizgig.web.qt_images import NativeImageModel, NativeImageRail, NativeImageCanvas


class NativeImageWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_model_selection_and_source_reset(self):
        model = NativeImageModel()
        rail = NativeImageRail(model)
        items = [
            {"path": "/tmp/native-a.png", "name": "A", "version": 1},
            {"path": "/tmp/native-b.png", "name": "B", "version": 2},
            {"path": "/tmp/native-c.png", "name": "C", "version": 3},
        ]
        model.replace(items)
        self.assertEqual(model.rowCount(), 3)
        self.assertEqual(model.data(model.index(1), Qt.ItemDataRole.UserRole),
                         "/tmp/native-b.png")
        calls = []
        rail.selection_changed.connect(lambda paths, active: calls.append((paths, active)))
        rail.sync_selection(["/tmp/native-a.png", "/tmp/native-c.png"],
                            "/tmp/native-c.png")
        self.assertEqual(len(rail.selectionModel().selectedIndexes()), 2)
        self.assertFalse(calls, "Synchronizing web selection must not echo to web")
        self.assertEqual(rail.currentIndex().row(), 2)
        rail.set_mode("samples")
        self.assertEqual(rail.mode, "samples")
        model.replace(items[:1])
        self.assertEqual(model.rowCount(), 1)
        self.assertNotIn("/tmp/native-c.png", model.positions)
        rail.close()

    def test_zoomable_native_canvas(self):
        viewer = NativeImageCanvas()
        image = QImage(64, 32, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.red)
        viewer.resize(400, 300)
        viewer.set_image(image)
        self.assertEqual(len(viewer.scene().items()), 1)
        viewer.set_image(None)
        self.assertEqual(len(viewer.scene().items()), 0)
        viewer.close()


if __name__ == "__main__":
    unittest.main()
