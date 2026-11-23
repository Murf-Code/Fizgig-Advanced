# Windows desktop WebView2 performance vs normal browser

**Branch:** `test/caption-workbench`

## Corrected A/B result (November 23, 2026)

The user compared two actual WebView2 process launches of Fizgig's desktop
window. **Unchecking** “Disable GPU for the app window” made it smooth.
Earlier discussion accidentally reversed this result.

| Setting | Browser / renderer flags | Actual performance |
| --- | --- | --- |
| **Unchecked**: GPU acceleration enabled | No tracked GPU-disabling switches | **FAST — preferred** |
| **Checked**: GPU acceleration disabled | `--disable-gpu`; renderer `--disable-gpu-compositing`; ANGLE WARP on helper | **SLOW** |

Therefore the checkbox must default to **OFF/unchecked** on both Windows and
Linux, preserving any explicit saved user setting. Normal Chrome and Edge
use their own acceleration settings, unaffected by this checkbox.

The GPU enabled process is the known-good path on this computer. Nothing in
these measurements proves a fault in GPU acceleration, Meta Virtual Monitor,
or the NVIDIA driver.

### Display-adapter result

- RTX 4070 Ti: Windows status OK, `ConfigManagerErrorCode=0`
- Meta Virtual Monitor: Windows status Error, `ConfigManagerErrorCode=22`

Code 22 means the virtual device is **disabled**, not proof of a faulty
driver. No device changes are recommended based on this result alone.

### Remaining diagnostics

The optional `scripts/diagnose_webview2.ps1` prints sanitized WebView2
GPU-related process flags and adapter error codes. The wrapper logs
pywebview/WebView2 versions and optional WebView2 process-failure events.

An optional temporary `FIZGIG_WEBVIEW2_NO_BLUR=1` test remains available
for users with **GPU enabled but still slow**. It is unnecessary to run when
the GPU-enabled desktop is already smooth.

This is a corrected interpretation of the user's observed runs, not
a measured FPS comparison or an explanation of why the software-rendered
path was unusually slow.

## Wrapper improvements retained

- Default Chromium GPU and accelerated video rendering settings, unless the
  user has explicitly enabled the GPU-disable compatibility toggle or supplied
  custom arguments.
- Do not silently swallow failed request-hook removals in pywebview 6.x.
- Record runtime version and actual graphics preferences for troubleshooting.
- Keep normal browser/remote workflows and all training services unchanged.
- Keep optional native Qt prototype separate: the fast browser and GPU-enabled
  WebView2 prove a Qt migration is not required to fix the reported slowdown.

## Sources

- [Unsloth Studio WebView2 GPU modes](https://github.com/unslothai/unsloth/blob/main/studio/src-tauri/src/gpu_rendering.rs)
- [Microsoft WebView2 performance](https://learn.microsoft.com/en-us/microsoft-edge/webview2/concepts/performance)
