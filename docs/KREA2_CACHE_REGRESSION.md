# Krea 2 cache regression queue

This is a diagnostic, not a production cache fix. Based on WIP 7878f133, retaining WIP numerical safeguards and stream synchronization. The former seven-arm test is not included.

## Run

1. Stop/finish any active run, select Krea 2 and the same original image folder.
2. No existing cache is required. RTX 3090 retains the original **1 MP (992 nominal)** recipe and first tries to snapshot the current compatible cache. If files are missing, the bucket is incompatible, or required variants are absent, the UI automatically requests a fresh private baseline. RTX 4070 Ti inherits the saved Krea 2 dataset resolution, bucket/no-upscale, caption extension and tag/caption dropout/variant settings and always generates a fresh private baseline. Neither path alters the original cache. If the previous failed run's cache was overwritten, even an existing-cache snapshot is not a snapshot of the historical failing bytes.
3. Click **Krea 2 cache test — RTX 3090** or **Krea 2 cache test — RTX 4070 Ti** in the Run card (choose the matching GPU) and enter one preview prompt containing the trigger word. Confirm.
4. The existing queue runs A/B/C/D in order, advancing after failures. Keep the models, captions, images, masks, preferences, and dataset unchanged during the comparison. There is no need to manually switch settings between arms.

| Arm | Image latents | Caption embeddings (including variants) |
|---|---|---|
| A | existing | existing |
| B | existing | new |
| C | new | existing |
| D | new | new |

The original cache is read-only for this diagnostic. When a fresh baseline is required, A first runs two cache-only stages in a new baseline-cache directory, then snapshots its results before any training. A receives real copies of the expected files; the other arms copy only their retained axis from A before any arm begins. Fresh axes start empty, and their explicit cache commands omit --skip_existing. The launch checks retained cache bytes and the frozen stage plan before each arm. Source cache completeness and latent bucket compatibility are checked; malformed or stale source text is deliberately not repaired in A/C. The manifest records whether the baseline is a current existing-cache snapshot or a fresh private baseline. A failed preparation cannot launch the remaining arms with missing cache data. Each run has a unique output name and isolated cache. The selected GPU profile is included in the comparison folder name and manifest; no automatic hardware-based profile selection is used. More disk space is needed for independent cache copies.

Fixed recipe: the original failing diagnostic recipe (INT8, explicit GPU profile: RTX 3090 with zero block swap and outside-checkpoint model compilation; RTX 4070 Ti with 24 swapped blocks and model compilation off, compiled PoLoRA, rank/alpha 8, LR 0.02, magnitude 3, batch 1, seed 42, masked training only if usable masks are found, floor 0.1, tag shuffle with four variants for the 3090; the 4070 Ti inherits the cache-related dataset fields only), with two epochs, adaptive/per-image LR and loss watch off, and a fixed automatic validation hold-out (one group; selected once per plan and verified across arms). Matching existing held-out cache files are copied into the separate validation directory, falling back to existing ordinary cache files for those pictures if needed. Missing training variants or a different source bucket cause the 3090 UI to retry with a fresh private baseline. Python, NumPy and Torch seeds are initialized. The sample prompt is fixed, 768 x 768, 24 steps, CFG 3.5, seed 42, with an epoch-zero render (initial training LoRA, no Turbo), then an epoch-end render. Actual resolved precision, compilation and other effective settings are logged. This reduced recipe is not a bitwise reproduction of the original failure.

## Return results

Under output_loras/polora_diagnostics/cache-regression-*/ (or configured output root), return diagnostic-settings.json, each case's diagnostic.log, sample images, validation/loss logs and preview-latents/ if present. The manifest contains cache SHA-256 hashes, not tensor contents. Do not include isolated-cache/ or model checkpoints unless specifically requested. The pre-decode latent files separate a sampler failure from a decoder failure. Logs include initial loss tensor statistics and WIP's numerical safety checks before and after optimizer updates.

If A fails and B succeeds, the text-cache path is implicated; C versus D provide the other matched comparison. If A does not reproduce the failure, the experiment is inconclusive, not proof of a fix. A fresh baseline can test cache generation and reuse on the current build, but cannot reproduce historical cache corruption that is no longer available. If base-only samples fail in all arms, inspect model/preview execution separately. A two-epoch pass is not a long-run stability guarantee.

CPU tests: checks/test_cache_regression.py exercises planning and file isolation with stubbed dataset/launch dependencies. Full web integration and GPU execution must be verified on the affected machine.
