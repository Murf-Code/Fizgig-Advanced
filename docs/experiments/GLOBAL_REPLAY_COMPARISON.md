# Global Replay follow-up (test/global-replay-comparison)

Select the Ferwildir dataset, leave Test Mode enabled and click Add to queue. Nine runs, five epochs each (the eight Experiment 2 controls plus Walking Coverage). The server fixes four sample prompts: triggered style transfer, novel wolf pose, untriggered watercolor likeness, unrelated landscape. No watermark tag.

| Case | LR | Selection |
| --- | --- | --- |
| replay | 0.00025 | all main blocks |
| replay-lr450 | 0.00045 | all main blocks |
| standard | 0.00025 | normal full targets |
| standard-lr450 | 0.00045 | normal full targets |
| high_gradient-lr450 | 0.00045 | top 14 RMS gradients |
| uphill-lr450 | 0.00045 | bottom 14 cosine scores |
| random-lr450 | 0.00045 | deterministic random 14 |
| static_early-lr450 | 0.00045 | blocks 0–12 and 27 |
| walking_coverage-lr450 | 0.00045 | top seven RMS gradients + seven walking coverage slots |

All selective cases use the same eight-update warmup and full probe every eight updates. Static is fixed between probes, not permanently pruned. All settings except LR and strategy match the first test: INT8, 22 swaps, Muon, rank/alpha 32, 0.25 MP, seed 42, fixed holdout validation, sample seed 1234 plus prompt index, 512 × 512, 20 requested sampling steps, epoch 0–5. Standard has additional adapter targets. Keep dataset and model/preferences unchanged during a comparison. The artist tag is not necessarily a character identity trigger.

Diagnostic lines mixed with progress bars are retained and probe details carry explicit step numbers. Reports are saved at the comparison root.

Once all cases reach a terminal state, global-replay-results.zip contains both reports, per-run logs and all sample images. Checkpoints and optimizer state are excluded. Interrupted comparisons remain incomplete.

Walking Coverage retains the seven highest RMS-gradient blocks until the next full probe and rotates seven evenly spaced slots in the sorted remaining 21 blocks after each selective optimizer update. With priorities 0–6, the coverage sets are 7/10/13/16/19/22/25, 8/11/14/17/20/23/26, then 9/12/15/18/21/24/27. The phase skips full probes and is derived from saved optimizer steps, not reset by probes or resume. Numbered rotation records in JSON identify the next optimizer step, separately from probe records. Expected total: 45 epochs, 216 preview images.
