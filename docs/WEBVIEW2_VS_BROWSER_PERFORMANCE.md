# Windows desktop WebView2 performance vs normal browser

**Branch:** `test/caption-workbench`

## Why this investigation changed

The same Caption Workbench was reported **very smooth in a normal browser** but
slow in the Windows *Fizgig* pywebview desktop window. This is evidence to
investigate the **WebView2 host, GPU setup and Python/.NET event bridge** first.
It is **not** evidence that the underlying tag editor needs a Qt rewrite.

The opt-in `qt_caption_trial.py` stays available, but do not replace the
existing browser image components solely on the basis of that comparison.

## Desktop wrapper adjustments

- Stop forcing `--disable-accelerated-video-decode` in every app window.
  Hardware-accelerated decode is again Chromium's default. This specific flag
  affects video, so its removal is **not proof** of a thumbnail fix.
- Honor the already-existing **Preferences → App window → Disable GPU**
  compatibility toggle. When it is enabled, log a clear warning that image
  rendering may become slow; the normal browser does not use this toggle.
- Preserve explicit user-provided
  `WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS` switches, including intentional
  GPU-disable overrides.
- On pywebview 6.x, independently remove request, response and resource
  filter handlers and **log failures**. On pywebview 5.x, those hooks were not
  installed; skip the removal instead of reporting a false failure.
- Record the pywebview and WebView2 Runtime versions at startup in
  `user_data/app_window.log`. These are diagnostic data, not performance
  measurements.

If a specific machine still hangs when a video starts, re-enable the original
software-video workaround **for that session only**:

```bat
set FIZGIG_WEBVIEW2_SOFTWARE_VIDEO=1
venv\Scripts\python.exe fizgig.py
```

Do **not** make this the default for everyone.

## Windows A/B acceptance test

1. Open Preferences → App window in either interface. Confirm that
   **Disable GPU for the app window** is unchecked. Restart the desktop app.
2. Open exactly the same dataset in the desktop window and in
   **Launch Fizgig in browser.bat**. Avoid simultaneously loading AI models.
3. Compare idle CPU, normal scrolling, fast scrolling, image selection, tag
   typing, image preview/mask brushing and view switching. Note whether a
   training run was active.
4. If the desktop still lags, inspect `user_data/app_window.log` for
   `[app] WebView2 rendering:`,
   `[app] WebView2 event hook removal:`, pywebview version and Runtime version.
   Logs indicating a failed response hook or GPU-disabled flag are actionable.
5. For advanced profiling compare the renderer/GPU processes and CPU usage
   in Windows Task Manager. Use WebView2 ETW traces if both configs look
   identical but lag remains.

If the browser is fast and the desktop still stutters after the change, keep
using the existing browser launcher. Do not redirect more engineering effort
to Qt rendering until WebView2 host overhead and GPU mode have been measured.

## Scope / caveats

No Windows GUI, GPU utilization, ETW or real scrolling FPS measurement has
been run for this patch. A fast browser is the known-good baseline, but we
have not proved which WebView2 setting causes the difference on the user's
computer. The existing app close/training safety workflow, backend HTTP
services, file handling and pywebview fallback remain unchanged.
