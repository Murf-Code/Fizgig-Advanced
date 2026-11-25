# Global Replay Experiment 3 (test/global-replay-comparison)

Select the same Ferwildir dataset, leave Test Mode enabled and click Add to queue. Six independent runs, five epochs each. Reuse Experiment 2's existing seed-42 unrestricted replay, high-gradient and random controls rather than rerunning them. Previous Experiment 2 reports remain unchanged; old queued cases remain supported.

| Case | LR | Selection |
| --- | --- | --- |
| balanced_random-lr450 | 0.00045 | shuffled complementary half-block masks, one pair per two probes |
| static_first_half-lr450 | 0.00045 | blocks 0–13 |
| rotational-lr450 | 0.00045 | alternate even and odd blocks at probes |
| hybrid_coverage-lr450 | 0.00045 | top seven RMS gradients + seven least gradient-updated remaining blocks |
| hybrid_coverage-lr350 | 0.00035 | same hybrid with lower LR |
| walking_coverage-lr450 | 0.00045 | top seven RMS gradients + seven walking coverage slots |

All selective cases use eight-update warmup and full probes every eight optimizer updates. Static is fixed between probes, not permanently pruned. Hybrid uses cumulative gradient-bearing block updates with seeded tie-breaking, persisted across resumes. Walking Coverage rotates after each selective optimizer update, not each microbatch or probe. With priorities 0–6, the green sets are 7/10/13/16/19/22/25, 8/11/14/17/20/23/26, then 9/12/15/18/21/24/27. The phase skips full probes and is derived from saved optimizer steps. If priority changes, the walk uses the sorted remaining blocks.

Controls: INT8, 22 swaps, Muon, rank/alpha 32, 0.25 MP, training seed 42 (same as Experiment 2), validation seed 42 and the same holdout selection, fixed routing RNG seed 42, sample seeds 1234–1237, 512 × 512, 20 requested sampling steps, epoch 0–5. Same four locked prompts: triggered fox style, novel wolf pose, untriggered watercolor likeness, unrelated landscape. No watermark tag. Keep dataset, model and preferences unchanged.

Reports include numbered probe and walking rotation records, plus cumulative per-block counts of optimizer updates with adapter gradients (not guaranteed nonzero parameter displacement). Expected total: 30 epochs, 144 preview images. Reusing prior results avoids duplicate runs but assumes identical model, dataset, hardware, and training configuration; this is not an independent-seed replication. The new per-block telemetry and any intervening implementation changes may affect timing comparisons. Standard and higher-LR unrestricted replay were evaluated in Experiment 2 and are not repeated in this coverage-focused recipe.

Once all cases reach a terminal state, global-replay-results.zip contains reports, logs and sample images; weights and optimizer states are excluded.
