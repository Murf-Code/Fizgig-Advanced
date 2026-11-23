# Native Qt image system — experimental foundation

Branch: `test/caption-workbench`.

The native Qt trial that was reported substantially smoother is becoming a **shared, opt-in desktop image layer**, not a complete Qt rewrite of Fizgig. The regular browser UI and pywebview launcher remain supported.

## Architecture

| Component | Implementation | Status |
| --- | --- | --- |
| Shared native image list | `qt_images.NativeImageModel` + `NativeImageRail(QListView)` | Implemented |
| Background thumbnail decoding | Bounded `QThreadPool`; `QImageReader` decodes `QImage` off the UI thread | Implemented |
| Thumbnail cache | LRU cache, 192 icons, 24 pending requests, stale generation discard | Implemented |
| Shared native zoom/pan | `NativeImageCanvas(QGraphicsView)`, launched from rail double-click | Read-only preview implemented |
| Captions image source | Exact filtered/sorted Workbench item list and Qt/web multi-selection | Integrated in Qt trial |
| Samples gallery | Native image list (image files only, leaves audio/video in web UI) | Integrated in Qt trial |
| Image Prep Look Filter | Native browser of scored images; **does not mark/move** items | Integrated in Qt trial |
| Caption image actions | Existing Workbench deletion confirmations, review locks and history, called from Qt | Remove wired |
| Fully native crop/masks | Existing browser editors and LaMa backend remain active | Not ported |
| Native duplicate comparison/Waterfall/Constellation | Existing browser interface remains active | Not ported |
| Native image grid in other tabs | Shared widgets can be reused | Not yet integrated |

## Try it on Windows

1. Pull `test/caption-workbench` and start Fizgig using **Launch Fizgig in browser.bat**. Keep the server running.
2. In Fizgig's environment, install the **optional** dependencies once:
   `venv\Scripts\python.exe -m pip install PySide6 PySide6-Addons`
3. Double-click **Launch Fizgig Native Images.bat**. It autodetects the port.
4. Sign in to the Qt window if requested (separate browser profile), then visit Captions, Samples, or Image Prep. For the latter, run Look Filter to populate its native image source.

Or run `venv\Scripts\python.exe -m fizgig.web.qt_caption_trial` with `PYTHONPATH=src`. `--url http://127.0.0.1:8081` remains an optional override only when that exact port is in use.

The web Captions **thumbnail rail is not painted** inside the Qt trial. The web filter, Missing Tags button, and action footer are retained so the Qt experiment doesn't silently discard controls. Outside the trial, the complete browser thumbnail rail still works as before.

## Selection and actions

- Single click, Ctrl/Cmd selection and Shift range use native Qt selection semantics; changes are reflected in the web Caption Workbench.
- Web-side selection changes update the native highlight, with selection snapshots transferred only after changes.
- Right-click offers native preview, copy filenames, and — only for Captions — removal via the web handler and its confirmation. Do **not** bypass the web confirmation workflow.
- Double-click uses the shared zoom/pan preview. Crop, mask painting and LaMa removal still use the existing editor.

## Tradeoffs and safety

Qt's model/delegate architecture avoids mounting hundreds of HTML thumbnail nodes, but it still decodes images on the CPU. The embedded web view still renders the other UI. Remote pod media may be unavailable to the native local-file decoder; use the regular browser for remote media.

Cache memory is capped; thumbnail tasks are limited to two workers; changes to a dataset invalidate cached results through version and generation keys. The native shell only reads local files and sends selection/action requests back to the same browser session. It does not write caption/image files directly.

**No Windows performance benchmark or end-to-end Qt GUI regression test has been performed in CI.** New checks cover JS source/selection logic and Python/JS syntax. Do not replace the default launcher until Ctrl/Shift selection, tag filtering, dataset refresh, image edits during training, 10k+ images, navigation, screen scaling, and resource usage have been validated on a Windows machine.

## Next migration phases

1. Benchmark the new shared rail against the old Qt trial on the same 10k+ dataset; confirm no performance regression from the additional selection bridge.
2. Move more image selection workflows (validation picks, duplicate finder, all preview galleries) to adapters using the same `NativeImageModel`.
3. Move edit surface rendering to `NativeImageCanvas` behind an experimental toggle; preserve the existing crop, mask and LaMa backend API, and verify atomic undo/redo and training-sync safety.
4. Add automated Qt interaction/performance tests and provide a persistent Qt desktop profile. Keep the browser and remote-pod paths fully functional.
