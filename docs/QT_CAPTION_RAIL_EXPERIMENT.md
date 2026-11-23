# Experimental Qt Caption Workbench rail

Branch: `test/caption-workbench`.

This is an opt-in prototype: a native PySide6 `QListView` and `QAbstractListModel` on the left, and the **existing** Fizgig web UI inside a `QWebEngineView` on the right. Neither the existing pywebview desktop window nor browser Captions tab is replaced.

## Run (Windows)

1. Start Fizgig normally in **browser mode**. The trial automatically discovers Fizgig's saved local port. If the server requires sign-in, sign in inside the trial web pane.
2. From Fizgig's activated Python environment, install the optional Qt dependencies: `python -m pip install PySide6 PySide6-Addons`.
3. Start the experiment: `python -m fizgig.web.qt_caption_trial`. An explicit `--url` remains available as an override.
4. Open the Captions tab in the right-hand pane and load an image dataset.

The native thumbnail list should populate. Clicking a native thumbnail selects the corresponding image in the embedded Workbench; selecting an image in the web UI updates the native highlight. The trial currently handles single-image selection only; multi-select, arrow-key navigation handoff, context menus, review colors, and filtering parity are **not** yet implemented.

## Architecture and limitations

- `QListView` requests thumbnail data for visible items, without mounting an HTML node for each image.
- `QImageReader` decoding uses a bounded two-worker pool; `QPixmap` creation stays on the Qt UI thread. The icon cache is capped and old icons can be redecoded.
- A lightweight JavaScript status poll retrieves full paths only when the dataset key/count/revision changes. The trial uses the Workbench's existing `S.cap.items`, `S.cap.studioPath`, and `renderCaptions()` state; it does **not** create a second caption-editing backend or alter training.
- The existing browser thumbnail rail is **still rendered** inside the embedded web pane, so overall system resource usage includes both rails. This is an A/B functional experiment, not yet a resource-saving production port.
- The Qt web view has its own browsing session. Password/cookie authentication from Chrome or pywebview does not carry over automatically.
- Network paths or unsupported formats may not decode as local thumbnails. This experiment is restricted to a local loopback URL; remote pods are out of scope.
- It is **not** benchmarked or verified on the user's Windows hardware.

## Before any replacement

Capture scrolling frame time, Qt GUI thread utilization, thumbnail decode latency, peak resident memory, and selection consistency on the same dataset in both the existing browser rail and Qt rail. Ensure >10k image datasets, refreshes with unchanged count, multi-selection/shift range, filtering, image edits, all views, DPI scaling, and training-time update synchronization are correct. Only ship Qt as default if the measured improvement offsets its added dependencies and integration complexity.
