# Krea 2 cache regression queue

This is a diagnostic, not a production cache fix. The former seven-arm PoLoRA NaN test is retired from the Run card and queue API; shared numerical instrumentation is retained.

## Run

1. Stop/finish any active run, select Krea 2 and the same original image folder.
2. Use the complete existing **1 MP (992 nominal), no-holdout** cache from the failing run. The test needs original caption embeddings, shuffled variants, and latent files. Do not clear/rebuild the source cache. If the old working checkout has already overwritten it, results describe the current cache, not the historical failing bytes.
3. Click **Krea 2 cache test** in the Run card and enter one preview prompt containing the trigger word. Confirm.
4. The existing queue runs A/B/C/D in order, advancing after failures. Keep the models, captions, images, masks, preferences, and dataset unchanged during the comparison. There is no need to manually switch settings between arms.

| Arm | Image latents | Caption embeddings (including variants) |
|---|---|---|
| A | existing | existing |
| B | existing | new |
| C | new | existing |
| D | new | new |

The original cache is read-only for this diagnostic. A receives real copies of the expected files; the other arms copy only their retained axis from A before any arm begins. Fresh axes start empty, and their explicit cache commands omit --skip_existing. The launch checks retained cache bytes and the frozen stage plan before each arm. Source cache completeness and latent bucket compatibility are checked; malformed or stale source text is deliberately not repaired in A/C. Missing cache/variants stop setup rather than change the experiment. Each run has a unique output name and isolated cache. More disk space is needed for independent cache copies.

Fixed recipe: old diagnostic arm A (INT8, zero block swap, compiled model outside checkpoint and compiled PoLoRA, rank/alpha 8, LR 0.02, magnitude 3, batch 1, seed 42, masks at 0.1, tag shuffle with four variants), with two epochs, adaptive/per-image LR and loss watch off, and a deterministic four-picture validation panel drawn from the training set (not hold-out). Validation is diagnostic rather than a generalization measure. Python, NumPy and Torch seeds are initialized. The sample prompt is fixed, 768 x 768, 24 steps, CFG 3.5, seed 42, with a base-only epoch-zero render (zero-initialized training LoRA, no Turbo), then an epoch-end render. Actual resolved precision, compilation and other effective settings are logged. This reduced recipe is not a bitwise reproduction of the original failure.

## Return results

Under output_loras/polora_diagnostics/cache-regression-*/ (or configured output root), return diagnostic-settings.json, each case's diagnostic.log, sample images, validation/loss logs and preview-latents/ if present. The manifest contains cache SHA-256 hashes, not tensor contents. Do not include isolated-cache/ or model checkpoints unless specifically requested. The pre-decode latent files separate a sampler failure from a decoder failure. Logs include initial loss tensor statistics and the existing NaN checks before and after optimizer updates.

If A fails and B succeeds, the text-cache path is implicated; C versus D provide the other matched comparison. If A does not reproduce the failure, the experiment is inconclusive, not proof of a fix. If base-only samples fail in all arms, inspect model/preview execution separately. A two-epoch pass is not a long-run stability guarantee.

CPU tests: checks/test_cache_regression.py exercises planning and file isolation with stubbed dataset/launch dependencies. Full web integration and GPU execution must be verified on the affected machine.
