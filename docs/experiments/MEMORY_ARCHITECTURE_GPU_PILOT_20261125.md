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

The first retry retained the 128 MiB cap after fixing dataset shadowing and telemetry in 1ff0c21. That cap is now superseded; see below. No combination, higher resolution, or auto-budget changes until the measurement and correctness gates are met.

## Retry and rank-64 budget correction — 2026-11-26/27

User-pasted Split-Max run (2026-11-25 23:43 local): same 13 training / 2 holdout images, 264 LoRA modules, 234.6M parameters, batch1, actual H2D ring, swap22, seed42; baseline validation loss .18260. First forward and backward completed, optimizer scope `ok:false`, zero CUDA allocator retries and OOM counters. Exception: `MemoryError: one compressed Muon matrix exceeds estimated chunk budget`. UI labeled this system RAM exhaustion, but the exception was a deliberate scratch preflight refusal. No completed optimizer update. Other retry arms' logs have not been supplied/reviewed here.

Source inspection: tproj.1 projects 6144 to 36864, so rank64 LoRA up factor can be 36864 × 64. Conservative modeled scratch BF16 157.55 MiB, INT8-64 155.44 MiB, Split-Max64 155.61 MiB exceeds 128 MiB. At [d7df5b3](https://github.com/Murf-Code/Fizgig-Advanced/commit/d7df5b334a7b445b46fe160425e38be75b7f92b2) the queue pins a common 192 MiB across all five arms, checks shape budget before training and distinguishes `ScratchBudgetError` in report. Full-size CPU tests reported 27 standalone passes, including two updates per compressed format at 192 MiB and non-mutating rejection at 128 MiB. Full CUDA validation pending; no claim of a successful retry at 192 MiB. Recreate the five-arm queue with new settings and preserve original results separately.
