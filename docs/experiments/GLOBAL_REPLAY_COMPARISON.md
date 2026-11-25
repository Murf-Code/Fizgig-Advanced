# Global Replay Experiment 4 — adaptive block selection

Select the same Ferwildir dataset, leave Test Mode enabled and click Add to queue. Six independent LoRAs, five epochs each. Keep the queue order: Adaptive Random depends on the completed Adaptive Gradient report from this group.

| Case | LR | Selection |
| --- | --- | --- |
| adaptive_descent | 0.00025 | original positive-descent rule, no fixed count |
| adaptive_descent-lr450 | 0.00045 | same rule at higher LR |
| adaptive_inverse_descent-lr450 | 0.00045 | nonpositive descent scores, no fixed count |
| adaptive_gradient-lr450 | 0.00045 | positive smoothed gradient RMS at least 0.5 times mean known RMS |
| adaptive_inverse_gradient-lr450 | 0.00045 | complement of the threshold on its own measured scores |
| adaptive_random-lr450 | 0.00045 | seeded random mask with exact per-probe counts from Adaptive Gradient |

All strategies can select 1–28 blocks. Descent policies preserve the original handling of unknown scores (active); if none qualify, train the best-scoring block for that policy. Gradient policies use smoothed RMS per parameter; missing scores are not treated as evidence, and if no scores are available all blocks train until reassessment. If no measured blocks qualify, keep one highest/lowest block respectively. The threshold is an experimental 0.5× mean, not a proven criterion. All-block warmup 8 updates, full-block probes every 8 updates, score decay 0.5. Random uses seed 42 and the source's recorded counts, not its own gradient counts; it refuses to start if the source failed, was removed, or has an incomplete schedule. Do not reorder or change the dataset midway.

Fixed controls: Krea 2 Raw, INT8, 22 swaps, Muon, rank/alpha 32, 0.25 MP, train/validation seed 42, same holdout, sample seeds 1234–1237, 512 × 512, 20 requested sampling steps, epochs 0–5. Four locked prompts and automatic ZIP unchanged. 30 epochs and 144 previews. No automatic LR multiplier in this experiment; it would confound the selection comparison. Prior Random and unrestricted Replay controls are reused, not replicated.

The report records probe masks, counts, smoothed scores, gradient-bearing updates per block, training/validation losses, wall-clock efficiency, VRAM/RAM and samples. A matched random run controls active-count exposure, not exact parameter updates. The legacy nonbenchmark policy and fixed-budget experiments remain available. Descent anchors reinitialize after resume as before; prefer uninterrupted runs for comparison.
