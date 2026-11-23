# Architecture decision: one shared web image interface

**Date:** 2026-11-23  
**Status:** Accepted — remove native Qt image shell from `test/caption-workbench`

## Decision

**Fizgig uses one production image UI:** the existing browser-based Caption
Workbench, Image Prep and Samples implementation. Windows desktop uses
pywebview/WebView2 with hardware acceleration enabled by default; browsers
connect to the same backend. On Linux, pywebview may use another supported
web engine; it does not need a separate native image component. No native Qt image renderer,
second preview stack, independent image-selection model, optional Qt launcher,
or periodic Qt/JavaScript synchronization bridge is retained.

This is an architecture choice, not a claim that Qt itself performs poorly.

## Evidence from the current and previous builds

- The experimental Qt `QListView` was built while the native Windows
  desktop window was extremely sluggish. A browser comparison originally
  looked faster, but the true distinguishing factor was the app-window
  **Disable GPU** preference.
- A controlled process comparison confirmed `--disable-gpu` and
  `--disable-gpu-compositing` when the checkbox was checked. With it
  unchecked, those switches were absent.
- The user then tested GPU-enabled Windows WebView2 on six areas: Caption
  Workbench, tag editing, Waterfall/Constellation, image editor, Image
  Prep/Samples, and resize/fullscreen. **6/6 passed, as smooth as browser**.
- Chromium already has a virtualized thumbnail rail with bounded preload,
  natural wheel scrolling, lazy decoding, and CPU-optimized tag and graph
  calculations. Qt did not demonstrate an additional measured speedup after
  the GPU-enabled comparison.
- The Qt experiment used a second desktop engine (Qt WebEngine), native file
  decoding, independent selection state, a JavaScript bridge, 800ms status
  polling, a second thumbnail cache and optional PySide6 dependencies. Image
  and caption edits still routed to the existing browser UI. That creates
  another maintenance/test surface without replacing the core editor.

## Alternatives considered

1. **Use Qt as the main interface** — requires replacing or duplicating
   complex caption editing, LaMa brush behavior, masks, GPU-integrated tools,
   training progress and existing web interactions. No verified performance
   benefit justified that ongoing cost.
2. **Keep Qt as an optional second shell** — avoids a disruptive transition
   but doubles UI integration paths, QA, packaging and support, and relies on
   ongoing selection synchronization. Rejected to avoid permanent complexity.
3. **Keep one shared web image interface** — preserves tested behavior,
   one UI architecture across the browser and desktop, and shared fixes.
   **Chosen.**

## Removed

- `src/fizgig/web/qt_images.py`
- `src/fizgig/web/qt_caption_trial.py`
- `src/fizgig/web/static/qt_native_image_bridge.js`
- `Launch Fizgig Native Images.bat`
- Qt-only regression tests, Qt CI smoke job, and retired experiment guide
- Qt-only state generation, action handlers, native-rail branch and selection
  hand-off in `src/fizgig/web/static/app.js`

No trained models, datasets, captions, image-editing endpoints or
browser-based performance optimizations were removed. The old experiment
remains available in Git history (e.g. `b9fff4ae8`), not installed or
maintained as another supported UI.

## Conditions for reconsidering

Revisit a native image layer **only** if repeatable measurements on a real
GPU-enabled desktop/browser identify an unresolved workflow bottleneck or
new operating-system requirement that the Chromium architecture cannot
satisfy. Benchmark on large datasets with measured FPS/interaction latency,
CPU usage, retained DOM, decoded-image memory, and VRAM **while training**;
compare candidate native implementations under the same conditions.

The threshold for revisiting Qt is a demonstrated product benefit that
outweighs losing a shared browser/desktop UI—not an assumed FPS advantage,
a hypothetical dataset size, or the prior GPU-disabled WebView2 regression.
