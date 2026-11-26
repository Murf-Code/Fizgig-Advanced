# First memory-architecture GPU pilot — 2026-11-25

Branch at dcf7cd0, Krea 2 INT8 base, rank/alpha 64, swap 22, 0.25 MP, seed 42, 13 updates, H2D-only ring, no samples. Two holdout pictures. Raw user logs retained separately; this document contains no local paths.

| Arm | Actual updates | Report | Step ms | CUDA peak allocated bytes | CUDA peak reserved bytes | Sampled device GiB |
|---|---:|---|---:|---:|---:|---:|
| FP32 control | 13 | incomplete forward telemetry | 2509.9 | 8263791104 | 8835301376 | 10.56 |
| BF16 | 0 | startup exception | — | — | — | — |
| INT8-64 | 0 | startup exception | — | — | — | — |
| Split-Max-64 | 0 | startup exception | — | — | — | — |
| Sync activation | 13 | incomplete forward telemetry | 2877.2 | 8237425152 | 8537505792 | 10.35 |

Both completed arms recorded 13 backward and optimizer scopes, but only forward steps 1 and 8. The disk log progress-bar throttle can suppress structured telemetry when it shares a physical line with a redraw. Also the batched forward path had no scope. Keep strict completeness checks and rerun. All three compressed arms failed before training because a parameter naming loop shadowed the dataset group with an optimizer dictionary. No OOM was logged.

Activation: 364 accepted inputs (28/update), no skips, 420 MiB host snapshot high-water mark, zero live at each post-update inventory. 284 MiB less peak CUDA reserved, ~25.1 MiB less peak allocated, 14.6% higher measured step time. Observed forward peak reduced by ~425.5 MiB, incomplete sampling. Backward is the largest observed allocation phase. System RAM peak increased 0.751 GiB; shared GPU peak unchanged at 1.08 GiB. Same frozen-weight transfer counts and bytes. No end-of-run validation or checkpoint continuation. Final displayed moving losses differ (0.1543 vs 0.1680); not evidence of numerical equivalence or quality loss. Capture step-level losses on retry, then fixed-input CUDA gradient comparison before any promotion.

Rerun all five matched arms at the same settings; no combination, higher resolution, or auto-budget changes until the measurement and correctness gates are met.
