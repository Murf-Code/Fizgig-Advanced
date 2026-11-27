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

User-pasted Split-Max run (2026-11-25 23:43 local): same 13 training / 2 holdout images, 264 LoRA modules, 234.6M parameters, batch1, actual H2D ring, swap22, seed42; baseline validation loss .18260. First forward and backward completed, optimizer scope `ok:false`, zero CUDA allocator retries and OOM counters. Exception: `MemoryError: one compressed Muon matrix exceeds estimated chunk budget`. UI labeled this system RAM exhaustion, but the exception was a deliberate scratch preflight refusal. No completed optimizer update. The remaining retry arms were subsequently recovered; see the complete archive section below.

Source inspection: tproj.1 projects 6144 to 36864, so rank64 LoRA up factor can be 36864 × 64. Conservative modeled scratch BF16 157.55 MiB, INT8-64 155.44 MiB, Split-Max64 155.61 MiB exceeds 128 MiB. At [d7df5b3](https://github.com/Murf-Code/Fizgig-Advanced/commit/d7df5b334a7b445b46fe160425e38be75b7f92b2) the queue pins a common 192 MiB across all five arms, checks shape budget before training and distinguishes `ScratchBudgetError` in report. Full-size CPU tests reported 27 standalone passes, including two updates per compressed format at 192 MiB and non-mutating rejection at 128 MiB. Full CUDA validation pending; no claim of a successful retry at 192 MiB. Recreate the five-arm queue with new settings and preserve original results separately.

## Complete retry archive recovered — 2026-11-27

User supplied five report/log/ZIP sets. Verified embedded JSON byte-identical to corresponding loose reports. Results(5) removed queue, (6) stopped during caching, (7) six-arm Muon-ME completed, (8) initial memory pilot, (9) full second memory pilot (geometry-memory-20261125-234019-f670eb; report filename 20261127-210729 is export time, not execution time). All five arms in both memory pilots used 128 MiB; no 192 MiB run supplied. All three compressed retry arms succeeded in first forward/backward, then failed at the same explicit scratch preflight. Their reported performance step count 1 is an attempted step; optimizer phase ok:false, no state inventory or loss records. All arms had zero observed CUDA OOMs/retries. 

| Retry arm | Completed updates | Step ms | Allocated peak MiB (forward/backward/optimizer) | Reserved peak GiB | Sampled device GiB |
|---|---:|---:|---|---:|---:|
| FP32 | 13 | 2508.1 | 7802.47 / 7879.87 / 6688.60 | 8.229 | 10.54 |
| Sync activation | 13 | 2888.0 | 7390.94 / 7858.40 / 6681.30 | 7.951 | 10.35 |
| BF16 / INT8-64 / Split-Max64 | 0 each | — | incomplete | — | — |

Both successful arms have exactly 13 successful records for each phase, 13 state inventories and 13 loss entries. Offload accepted 364 inputs, no skips, 420 MiB host peak, zero live after each update per reported inventory. Forward allocation reduced 411.54 MiB, backward 21.48 MiB, reserved 284 MiB; step time +15.15%, system RAM +0.795 GiB, shared GPU 1.08 GiB unchanged. Identical final state inventory bytes (938446080 FP32 momentum), frozen-weight transfer counts and bytes (766 H2D, 332528615424 bytes; zero D2H). This independently repeats the first pilot's operational finding with complete phase telemetry. First pre-update loss differs .19066694 vs .15002115; mean of 13 losses .15571207 vs .17482471. Same holdout baseline .18260, no post-training validation. Input/initialization pairing not established; first-step discrepancy before optimizer updates cannot establish offload-induced gradient or quality error. Fixed-input parity and repeat control remain required. No new training execution or raw private data committed.

Next: create fresh 192 MiB queue after pulling latest research branch (do not reuse old entries). The current branch source pins 192 MiB. The 128 MiB control and activation results should be preserved, not pooled with the forthcoming changed-budget arms as if matched.
