"""Reusable native image widgets for Fizgig's *optional* Qt desktop shell.

The Python dataset/training services remain the authority for edits. Qt only
browses, decodes and previews media; it never modifies images or captions.
No PySide6 dependency is introduced into Fizgig's normal browser/pywebview path.
"""
from collections import OrderedDict
import os

from PySide6.QtCore import (
    QAbstractListModel, QModelIndex, QObject, QRunnable, QSize, Qt,
    QThreadPool, Signal, Slot, QItemSelection, QItemSelectionModel, QSignalBlocker
)
from PySide6.QtGui import QIcon, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QListView, QGraphicsView, QGraphicsScene, QDialog, QVBoxLayout,
    QLabel, QMenu
)

THUMB_WIDTH = 144
THUMB_MAX_HEIGHT = 432
THUMB_CACHE_LIMIT = 192
THUMB_QUEUE_LIMIT = 24


class ImageLoaded(QObject):
    finished = Signal(int, str, str, object)


class DecodeImage(QRunnable):
    """QImage is safe to decode off-thread; QPixmap is made on the GUI thread."""
    def __init__(self, generation, key, path, signals, width, max_height):
        super().__init__()
        self.generation = generation
        self.key = key
        self.path = path
        self.signals = signals
        self.width = width
        self.max_height = max_height

    def run(self):
        image = None
        if os.path.isfile(self.path):
            reader = QImageReader(self.path)
            reader.setAutoTransform(True)
            dimensions = reader.size()
            if dimensions.isValid() and dimensions.width() > 0:
                height = max(1, round(self.width * dimensions.height() / dimensions.width()))
                if height > self.max_height:
                    reader.setScaledSize(QSize(
                        max(1, round(self.width * self.max_height / height)),
                        self.max_height))
                else:
                    reader.setScaledSize(QSize(self.width, height))
            image = reader.read()
        self.signals.finished.emit(self.generation, self.key, self.path, image)


class NativeImageModel(QAbstractListModel):
    """Viewport-driven image model; never creates one QWidget per image."""
    def __init__(self, parent=None, cache_limit=THUMB_CACHE_LIMIT):
        super().__init__(parent)
        self.records = []
        self.positions = {}
        self.cache = OrderedDict()
        self.pending = set()
        self.failed = set()
        self.generation = 0
        self.cache_limit = cache_limit
        self.signals = ImageLoaded(self)
        self.signals.finished.connect(self._received)
        self.workers = QThreadPool(self)
        self.workers.setMaxThreadCount(2)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.records)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self.records):
            return None
        record = self.records[index.row()]
        path = record["path"]
        if role == Qt.ItemDataRole.DisplayRole:
            return record.get("name") or os.path.basename(path)
        if role == Qt.ItemDataRole.UserRole:
            return path
        if role == Qt.ItemDataRole.SizeHintRole:
            return QSize(186, 180)
        if role == Qt.ItemDataRole.DecorationRole:
            key = self._cache_key(record)
            if key in self.cache:
                icon = self.cache.pop(key)
                self.cache[key] = icon
                return icon
            if key not in self.pending and key not in self.failed \
                    and len(self.pending) < THUMB_QUEUE_LIMIT:
                self.pending.add(key)
                self.workers.start(DecodeImage(
                    self.generation, key, path, self.signals,
                    THUMB_WIDTH, THUMB_MAX_HEIGHT))
        return None

    @staticmethod
    def _cache_key(record):
        return str(record["path"]) + "\0" + str(record.get("version", ""))

    @Slot(int, str, str, object)
    def _received(self, generation, key, path, image):
        self.pending.discard(key)
        if generation != self.generation:
            return
        row = self.positions.get(path)
        if row is None or self._cache_key(self.records[row]) != key:
            return
        if image is None or image.isNull():
            self.failed.add(key)
        else:
            self.cache[key] = QIcon(QPixmap.fromImage(image))
            self.cache.move_to_end(key)
            while len(self.cache) > self.cache_limit:
                self.cache.popitem(last=False)
        self.dataChanged.emit(self.index(row), self.index(row), [
            Qt.ItemDataRole.DecorationRole])

    def replace(self, records):
        records = [
            {"path": str(item["path"]), "name": str(item.get("name", "")),
             "version": str(item.get("version", ""))}
            for item in records if item.get("path")
        ]
        # Reuse unchanged thumbnails, but discard retired versions and paths.
        live = {self._cache_key(row) for row in records}
        self.beginResetModel()
        self.generation += 1
        self.records = records
        self.positions = {row["path"]: index for index, row in enumerate(records)}
        self.pending.clear()
        self.failed.intersection_update(live)
        for key in tuple(self.cache):
            if key not in live:
                self.cache.pop(key, None)
        self.endResetModel()

    def invalidate(self, paths):
        """Evict changed images; only their visible rows are repainted."""
        for path in paths:
            row = self.positions.get(path)
            if row is None:
                continue
            key = self._cache_key(self.records[row])
            self.cache.pop(key, None)
            self.failed.discard(key)
            self.dataChanged.emit(self.index(row), self.index(row), [
                Qt.ItemDataRole.DecorationRole])


class NativeImageRail(QListView):
    """Common browser for Captions, Image Prep, Samples and future image grids."""
    selection_changed = Signal(list, str)
    request_action = Signal(str, str)
    open_preview = Signal(str)

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.setModel(model)
        self.mode = ""
        self.setSelectionMode(QListView.SelectionMode.ExtendedSelection)
        self.setSelectionBehavior(QListView.SelectionBehavior.SelectRows)
        self.setUniformItemSizes(True)
        self.setVerticalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        self.setIconSize(QSize(THUMB_WIDTH, THUMB_WIDTH))
        self.setMinimumWidth(186)
        self.setMaximumWidth(340)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.doubleClicked.connect(
            lambda index: self.open_preview.emit(index.data(Qt.ItemDataRole.UserRole)))
        self.selectionModel().selectionChanged.connect(self._emit_selection)

    def _emit_selection(self, *args):
        selected = sorted(self.selectionModel().selectedIndexes(), key=lambda i: i.row())
        paths = [i.data(Qt.ItemDataRole.UserRole) for i in selected]
        current = self.currentIndex().data(Qt.ItemDataRole.UserRole)
        self.selection_changed.emit(paths, current or (paths[-1] if paths else ""))

    def sync_selection(self, paths, active):
        desired = set(paths)
        positions = self.model().positions
        rows = sorted(positions[p] for p in desired if p in positions)
        model = self.selectionModel()
        # One block per consecutive range, not one Qt selection operation per image.
        with QSignalBlocker(model):
            selection = QItemSelection()
            if rows:
                start = previous = rows[0]
                for row in rows[1:]:
                    if row != previous + 1:
                        selection.select(self.model().index(start),
                                         self.model().index(previous))
                        start = row
                    previous = row
                selection.select(self.model().index(start),
                                 self.model().index(previous))
            model.select(selection, QItemSelectionModel.SelectionFlag.ClearAndSelect)
            active_row = positions.get(active)
            if active_row is not None:
                target = self.model().index(active_row)
                model.setCurrentIndex(target, QItemSelectionModel.SelectionFlag.NoUpdate)
                if not self.visualRect(target).intersects(self.viewport().rect()):
                    self.scrollTo(target, QListView.ScrollHint.EnsureVisible)

    def set_mode(self, mode):
        self.mode = mode

    def _menu(self, position):
        index = self.indexAt(position)
        if not index.isValid():
            return
        path = index.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        menu.addAction("Open native preview", lambda: self.open_preview.emit(path))
        menu.addAction("Copy file name(s)", lambda: self.request_action.emit("copy_names", path))
        if self.mode == "captions":
            menu.addSeparator()
            menu.addAction("Toggle review complete", lambda: self.request_action.emit("review", path))
            menu.addAction("Toggle validation images", lambda: self.request_action.emit("validation", path))
            menu.addAction("Toggle sample images", lambda: self.request_action.emit("sample", path))
            menu.addSeparator()
            menu.addAction("Remove selected image(s)…", lambda: self.request_action.emit("remove", path))
        menu.exec(self.viewport().mapToGlobal(position))


class NativeImageCanvas(QGraphicsView):
    """Shared zoomable/pannable native preview. Edit operations stay with Fizgig."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHints(self.renderHints())
        self._pixmap = None

    def set_image(self, image):
        self._scene.clear()
        self.resetTransform()
        if image is None or image.isNull():
            self._pixmap = None
            return
        self._pixmap = self._scene.addPixmap(QPixmap.fromImage(image))
        self._scene.setSceneRect(self._pixmap.boundingRect())
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap is not None and self.transform().m11() == 1:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        if event.angleDelta().y() == 0:
            return super().wheelEvent(event)
        if .03 < self.transform().m11() * factor < 30:
            self.scale(factor, factor)


class NativePreviewDialog(QDialog):
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(os.path.basename(path))
        self.resize(1100, 800)
        self.canvas = NativeImageCanvas(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.label = QLabel("Loading preview…")
        layout.addWidget(self.label)
        self.signals = ImageLoaded(self)
        self.signals.finished.connect(self._preview_ready)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.pool.start(DecodeImage(0, path, path, self.signals, 2200, 2200))

    @Slot(int, str, str, object)
    def _preview_ready(self, _gen, _key, _path, image):
        if image is None or image.isNull():
            self.label.setText("Cannot decode this local image")
        else:
            self.canvas.set_image(image)
            self.label.setText("Drag to pan · mouse wheel to zoom · read-only preview")
