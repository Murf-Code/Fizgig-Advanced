"""Optional native Qt thumbnail-rail A/B trial for a running local Fizgig server.

Install: pip install PySide6 PySide6-Addons
Run: python -m fizgig.web.qt_caption_trial --url http://127.0.0.1:7860
The existing pywebview/browser UI is not replaced.
"""
import argparse
import json
import os
import sys

try:
    from PySide6.QtCore import (QAbstractListModel, QModelIndex, QObject, QRunnable,
                                QSize, Qt, QThreadPool, QTimer, QUrl, Signal, Slot)
    from PySide6.QtGui import QIcon, QImageReader, QPixmap
    from PySide6.QtWidgets import QApplication, QHBoxLayout, QListView, QMainWindow, QWidget
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError as exc:
    raise SystemExit("Install optional Qt trial dependencies: pip install PySide6 PySide6-Addons") from exc

STATUS = """(() => {
  if(typeof S==='undefined'||!S.cap)return null;
  return {folder:S.imageFolder||'', count:(S.cap.items||[]).length,
          selected:S.cap.studioPath||'', revision:S.cap.studioRevision||0};
})()"""
SNAPSHOT = """(() => {
  if(typeof S==='undefined'||!S.cap)return null;
  return {folder:S.imageFolder||'', count:(S.cap.items||[]).length,
          selected:S.cap.studioPath||'', revision:S.cap.studioRevision||0,
          items:(S.cap.items||[]).map(x=>({path:x.path,name:x.name||''}))};
})()"""
SELECT = """((path) => {
  if(typeof S==='undefined'||!S.cap||!(S.cap.items||[]).some(x=>x.path===path))
    return false;
  S.cap.studioSelection=new Set([path]);
  S.cap.studioSelectionAnchor=path;
  S.cap.studioPath=path;
  if(typeof renderCaptions==='function')renderCaptions();
  return true;
})"""


class Signals(QObject):
    loaded = Signal(str, object)


class Decode(QRunnable):
    def __init__(self, path, signals):
        super().__init__()
        self.path, self.signals = path, signals

    def run(self):
        reader = QImageReader(self.path)
        reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid() and size.width():
            reader.setScaledSize(QSize(144, max(1, 144 * size.height() // size.width())))
        img = reader.read()
        if not img.isNull():
            self.signals.loaded.emit(self.path, img)


class ImageModel(QAbstractListModel):
    def __init__(self):
        super().__init__()
        self.items = []
        self.positions = {}
        self.icons = {}
        self.pending = set()
        self.signals = Signals(self)
        self.signals.loaded.connect(self.received)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.items)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self.items):
            return None
        entry = self.items[index.row()]
        path = entry["path"]
        if role == Qt.ItemDataRole.DisplayRole:
            return entry.get("name") or os.path.basename(path)
        if role == Qt.ItemDataRole.UserRole:
            return path
        if role == Qt.ItemDataRole.SizeHintRole:
            return QSize(185, 182)
        if role == Qt.ItemDataRole.DecorationRole:
            if path not in self.icons and path not in self.pending and os.path.isfile(path):
                self.pending.add(path)
                self.pool.start(Decode(path, self.signals))
            return self.icons.get(path)
        return None

    @Slot(str, object)
    def received(self, path, image):
        self.pending.discard(path)
        row = self.positions.get(path)
        if row is None:
            return
        self.icons[path] = QIcon(QPixmap.fromImage(image))
        self.dataChanged.emit(self.index(row), self.index(row),
                              [Qt.ItemDataRole.DecorationRole])

    def replace(self, items):
        self.beginResetModel()
        self.items = list(items)
        self.positions = {entry["path"]: i for i, entry in enumerate(items)}
        # Keep cached icons only for the current dataset (limited below).
        self.icons = {p: icon for p, icon in self.icons.items() if p in self.positions}
        if len(self.icons) > 256:
            self.icons = dict(list(self.icons.items())[-256:])
        self.endResetModel()


class Trial(QMainWindow):
    def __init__(self, url):
        super().__init__()
        self.setWindowTitle("Fizgig native QListView trial")
        self.resize(1400, 850)
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(holder)
        self.rail = QListView()
        self.rail.setMinimumWidth(195)
        self.rail.setMaximumWidth(320)
        self.rail.setIconSize(QSize(144, 144))
        self.rail.setUniformItemSizes(True)
        self.rail.setVerticalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        self.model = ImageModel()
        self.rail.setModel(self.model)
        self.web = QWebEngineView()
        layout.addWidget(self.rail)
        layout.addWidget(self.web, 1)
        self.key = None
        self.busy = False
        self.rail.clicked.connect(self.choose)
        self.web.loadFinished.connect(lambda ok: setattr(self, "key", None))
        self.web.setUrl(url)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(800)
        self.statusBar().showMessage("Open Caption Workbench in the right pane")

    def choose(self, index):
        path = self.model.data(index, Qt.ItemDataRole.UserRole)
        if path:
            self.web.page().runJavaScript(SELECT + "(" + json.dumps(path) + ")")

    def poll(self):
        if not self.busy:
            self.busy = True
            self.web.page().runJavaScript(STATUS, self.status)

    def status(self, result):
        self.busy = False
        if not isinstance(result, dict):
            return
        key = (result.get("folder"), result.get("count"), result.get("revision"))
        if key != self.key:
            self.key = key
            self.web.page().runJavaScript(SNAPSHOT, self.snapshot)
        else:
            self.highlight(result.get("selected"))

    def snapshot(self, result):
        if not isinstance(result, dict):
            return
        self.model.replace(result.get("items") or [])
        self.highlight(result.get("selected"))
        self.statusBar().showMessage(str(len(self.model.items)) + " images loaded")

    def highlight(self, path):
        row = self.model.positions.get(path)
        if row is not None:
            index = self.model.index(row)
            if index != self.rail.currentIndex():
                self.rail.setCurrentIndex(index)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Running local Fizgig URL")
    args = parser.parse_args(argv)
    url = QUrl(args.url)
    if url.scheme() not in ("http", "https") or url.host() not in (
            "localhost", "127.0.0.1", "::1"):
        parser.error("Only loopback URLs are permitted")
    app = QApplication(sys.argv[:1])
    window = Trial(url)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
