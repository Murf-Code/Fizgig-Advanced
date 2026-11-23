"""Optional native Qt desktop shell for image-heavy Fizgig views.

Fizgig's training backend and existing browser/pywebview shell are unchanged.
Launch a running local server with:
    python -m fizgig.web.qt_caption_trial
Optional dependencies: pip install PySide6 PySide6-Addons

The shared browser supports Captions, Samples and Image Prep's Look Filter.
Edits continue to use Fizgig's original validated web handlers.
"""
import argparse
import json
import os
from pathlib import Path
import sys

try:
    from PySide6.QtCore import QUrl, QTimer
    from PySide6.QtWidgets import (
        QApplication, QHBoxLayout, QMainWindow, QWidget
    )
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from fizgig.web.qt_images import (
        NativeImageModel, NativeImageRail, NativePreviewDialog
    )
except ImportError as exc:
    raise SystemExit(
        "Qt image shell requires optional packages: pip install PySide6 PySide6-Addons"
    ) from exc


STATIC = Path(__file__).resolve().parent / "static" / "qt_native_image_bridge.js"


class NativeShell(QMainWindow):
    """Qt image browser and embedded existing Workbench, not a second backend."""
    def __init__(self, url):
        super().__init__()
        self.setWindowTitle("Fizgig · Native Image Shell (experimental)")
        self.resize(1520, 900)
        self._generation = 0
        self._source_key = None
        self._selection_key = None
        self._pending_selection_key = None
        self._busy = False
        self._preview_dialog = None

        root = QWidget(self)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.setCentralWidget(root)

        self.model = NativeImageModel(self)
        self.rail = NativeImageRail(self.model, root)
        self.rail.setVisible(False)
        self.rail.selection_changed.connect(self._native_selection)
        self.rail.request_action.connect(self._action)
        self.rail.open_preview.connect(self._preview)

        self.web = QWebEngineView(root)
        self.web.loadFinished.connect(self._loaded)
        layout.addWidget(self.rail)
        layout.addWidget(self.web, 1)
        self.statusBar().showMessage(
            "Experimental native browsing · existing browser and desktop builds unchanged")
        self.web.setUrl(url)

        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(800)
        self.poll_timer.timeout.connect(self._poll)
        self.poll_timer.start()

    def _loaded(self, ok):
        self._generation += 1
        self._busy = False
        self._source_key = None
        self._selection_key = None
        self.model.replace([])
        self.rail.setVisible(False)
        if not ok:
            self.statusBar().showMessage("Couldn't load the local Fizgig web server")
            return
        # Inject the bridge only in this QWebEngineView. Browser and pywebview
        # sessions continue to use their existing DOM rail.
        source = STATIC.read_text(encoding="utf-8")
        self._busy = True
        generation = self._generation
        self.web.page().runJavaScript(source, lambda _:
            self._bridge_ready(generation))

    def _bridge_ready(self, generation):
        if generation != self._generation:
            return
        self._busy = False
        # Re-render the already-open caption view once, skipping its hidden
        # thumbnail construction. All its non-image controls remain available.
        self.web.page().runJavaScript(
            "(typeof renderCaptions==='function' && "
            "document.querySelector('#tab-captions.active')) ? "
            "(renderCaptions(),true) : false")
        self._poll()

    def _poll(self):
        if self._busy:
            return
        self._busy = True
        generation = self._generation
        self.web.page().runJavaScript(
            "window.FizgigNativeImages?.status?.()||null",
            lambda state: self._status(generation, state))

    def _status(self, generation, state):
        if generation != self._generation:
            return
        if not isinstance(state, dict):
            self._busy = False
            return
        source_key = (state.get("tab"), state.get("key"))
        if source_key != self._source_key:
            self.web.page().runJavaScript(
                "window.FizgigNativeImages.snapshot()",
                lambda snapshot: self._snapshot(generation, snapshot))
            return
        if state.get("selection") != self._selection_key:
            self._pending_selection_key = state.get("selection")
            self.web.page().runJavaScript(
                "window.FizgigNativeImages.selection()",
                lambda selection: self._selection(generation, selection))
            return
        self._busy = False

    def _snapshot(self, generation, snapshot):
        if generation != self._generation:
            return
        self._busy = False
        if not isinstance(snapshot, dict):
            return
        tab = snapshot.get("tab", "")
        entries = snapshot.get("items") or []
        self.model.replace(entries)
        self._source_key = (tab, snapshot.get("key"))
        self._set_selection(snapshot)
        self.rail.setVisible(bool(entries) and tab in ("captions", "samples", "prep"))
        self.statusBar().showMessage(
            f"{len(entries):,} native thumbnails · {tab or 'no image source'} · "
            "double-click for native zoom/pan preview")

    def _selection(self, generation, selection):
        if generation != self._generation:
            return
        self._busy = False
        if not isinstance(selection, dict):
            return
        if (selection.get("tab"), selection.get("key")) != self._source_key:
            self._source_key = None
            return
        self._set_selection(selection)
        self._selection_key = self._pending_selection_key

    def _set_selection(self, selection):
        selected = selection.get("selected") or []
        active = selection.get("active") or ""
        self.rail.sync_selection(selected, active)
        # A snapshot requires one status round-trip to establish its
        # selection fingerprint; subsequent idle polls do not rebuild Qt ranges.
        self._selection_key = None

    def _native_selection(self, selected, active):
        payload = json.dumps([selected, active], ensure_ascii=True)
        # Avoid applying a stale web selection while a native selection is pending.
        self._selection_key = None
        self.web.page().runJavaScript(
            f"window.FizgigNativeImages?.select?.(...{payload})")

    def _action(self, action, path):
        if action == "copy_names":
            selected = self.rail.selectionModel().selectedIndexes()
            entries = [i.data() for i in selected]
            if entries:
                QApplication.clipboard().setText("\n".join(entries))
            return
        payload = json.dumps([action, path], ensure_ascii=True)
        self.web.page().runJavaScript(
            f"window.FizgigNativeImages?.action?.(...{payload})")

    def _preview(self, path):
        if not path or not os.path.isfile(path):
            self.statusBar().showMessage(
                "Native preview requires a local image. Use the web preview for remote media.")
            return
        if self._preview_dialog:
            self._preview_dialog.close()
        self._preview_dialog = NativePreviewDialog(path, self)
        self._preview_dialog.show()


def local_url():
    """Read the same saved port as Fizgig's normal desktop launcher."""
    from fizgig.user_data import PREFS_FILE
    from fizgig.web.port import saved_port, is_fizgig
    try:
        with open(PREFS_FILE, encoding="utf-8") as stream:
            prefs = json.load(stream)
    except (OSError, ValueError):
        prefs = {}
    port = saved_port(prefs)
    if not is_fizgig(port):
        raise RuntimeError(
            f"Fizgig is not running on saved port {port}. "
            "Start Fizgig first or supply --url with the running local port.")
    return f"http://127.0.0.1:{port}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Optional running local Fizgig URL")
    options = parser.parse_args(argv)
    try:
        chosen = options.url or local_url()
    except RuntimeError as exc:
        parser.error(str(exc))
    url = QUrl(chosen)
    if url.scheme() not in ("http", "https") or url.host() not in (
            "localhost", "127.0.0.1", "::1"):
        parser.error("Only loopback Fizgig URLs are supported in the Qt trial")
    app = QApplication(sys.argv[:1])
    window = NativeShell(url)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
