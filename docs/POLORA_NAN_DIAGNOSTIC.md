# Krea 2 / PoLoRA NaN diagnostic

Use the same model files and original dataset (including masks and captions). Stop the old run; do not resume a NaN checkpoint. In the web UI select Krea 2, select the dataset, and click **PoLoRA NaN test** in the Run card. Confirm to add all seven arms to the existing queue; if the queue is held, press Start queue. Do not edit queued diagnostic runs. A failed run normally advances automatically. Do not press Stop unless you want to hold the remaining queue.

The matrix uses fresh rank-8/alpha-8 LoRAs, seed 42, 1 MP (992px nominal), batch size 1, one epoch, fixed explicit compile settings and independent output folders. The 0.98 MP in the original training log is the effective resolution, not the nominal setting. Samples, training maps and resume-state exports are disabled. PoLoRA arms use magnitude 3, curvature beta 0.99, adaptive LR 0.01–0.04 and per-image LR; the AdamW8bit control uses fixed 2e-4. BF16 uses automatic block swapping to fit the GPU. No automatic OOM fallback changes the diagnostic recipe.

The baseline is a reduced reproduction, not a bitwise replay: unknown original settings and seed, samples/maps off, and the current release base rather than the friend's older commit 39593d9565204327f3901f2caaea8a13dc27d287. If no arm fails, repeat with the original revision/settings, maps and samples as needed.

## Collect results

After all cases finish, zip the complete timestamped directory under the configured output directory's `polora_diagnostics/`. It contains `diagnostic-settings.json` and per-case `diagnostic.log` files (including full subprocess output and stage exit codes). Large LoRA files can be excluded from the zip.

Diagnostic-only finite checks distinguish forward loss, pre-clip gradient norm, PoLoRA inverse-square-root/polar/spectral calculations, optimizer state, and post-update adapter weights. A nonfinite result aborts that case before the usual checkpoint/EMA path. These checks synchronize GPU work deliberately and are not intended as speed benchmarks. A successful one-epoch run is evidence against the immediate failure, not proof of long-run stability.

Reported environment: Linux, RTX 3090, Python 3.14, torch 2.10.0+cu128, Triton 3.6.0, driver 595.99.02. The trainer logs the actual Python, torch, CUDA build, Triton and device capability in each case.
