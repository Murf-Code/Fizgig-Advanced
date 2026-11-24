# Global Replay follow-up (test/global-replay-comparison)

Select the Ferwildir dataset, leave Test Mode enabled and click Add to queue. Eight runs, five epochs each. The server fixes four sample prompts: triggered style transfer, novel wolf pose, untriggered watercolor likeness, unrelated landscape. No watermark tag.

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

All selective cases use the same eight-update warmup and full probe every eight updates. Static is fixed between probes, not permanently pruned. All settings except LR and strategy match the first test: INT8, 22 swaps, Muon, rank/alpha 32, 0.25 MP, seed 42, fixed holdout validation, sample seed 1234 plus prompt index, 512 × 512, 20 requested sampling steps, epoch 0–5. Standard has additional adapter targets. Keep dataset and model/preferences unchanged during a comparison. The artist tag is not necessarily a character identity trigger.

Diagnostic lines mixed with progress bars are retained and probe details carry explicit step numbers. Reports are saved at the comparison root.

Once all cases reach a terminal state, global-replay-results.zip contains both reports, per-run logs and all sample images. Checkpoints and optimizer state are excluded. Interrupted comparisons remain incomplete.
