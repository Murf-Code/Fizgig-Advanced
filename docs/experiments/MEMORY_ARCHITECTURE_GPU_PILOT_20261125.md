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

The planned fresh 192 MiB queue was subsequently completed on Nov 27; see below. Preserve the 128 MiB history separately, not as matched arms of the new queue.


## Round 3 — complete GPU feasibility at 192 MiB (2026-11-27)

Run `geometry-memory-20261127-131338-4712f5`. User archive matrix-geometry-results(10).zip and report/log export suffix 20261127-212149. ZIP integrity passed; embedded JSON and comparison log byte-identical to loose files; all structured phase/state/config/loss records matched the five individual raw logs exactly. Each arm had 13 successful scopes per forward/backward/optimizer, 13 state inventories, 13 finite losses and a completion marker. All five used chunk_memory_mib=192, rank/alpha64, seed42, LR .0002, 22 swaps, INT8 base, standard eager checkpointed LoRA, 13 training images (11 at 496x496 and one each 448x544 and 528x464), two holdouts, no previews or saves. All reported zero CUDA allocation retries/OOMs. Settings identical except momentum codec, activation flags, run names and output paths. First control rebuilt some caches; remaining arms reused caches, so compare training timings rather than startup-inclusive wall times. No paired actual tensor/input/initialization verification.

| Arm | Completed updates | State bytes | Allocated peak bytes | Reserved peak bytes | Device peak GiB (sampled) | Step ms | Training wall s |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 13 | 938446080 | 8261988864 | 8835301376 | 10.73 | 2433.8 | 34.266 |
| BF16 | 13 | 469223040 | 7773347840 | 8407482368 | 10.15 | 2613.0 | 36.750 |
| INT8-64 | 13 | 249274740 | 7547469312 | 8118075392 | 10.04 | 2808.6 | 39.578 |
| Split-Max64 | 13 | 267603765 | 7570945536 | 8136949760 | 10.04 | 2889.8 | 40.625 |
| Sync activation | 13 | 938446080 | 8237883904 | 8537505792 | 10.54 | 2797.0 | 39.375 |

| Arm | State saved MiB | Allocated saved MiB | Reserved saved MiB | Step time increase | Forward/backward/optimizer allocated peak MiB | Median optimizer host scope ms (steps 2–13) |
|---|---:|---:|---:|---:|---|---:|
| FP32 | 0 | 0 | 0 | 0% | 7799.72 / 7879.25 / 6719.40 | 93.5 |
| BF16 | 447.49 | 466.00 | 408 | 7.36% | 7341.29 / 7413.24 / 6202.61 | 250 |
| INT8-64 | 657.25 | 681.42 | 684 | 15.40% | 7126.17 / 7197.83 / 5983.37 | 437 |
| Split-Max64 | 639.77 | 659.03 | 666 | 18.74% | 7143.48 / 7220.22 / 6005.49 | 532 |
| Sync activation | 0 | 22.99 | 284 | 14.92% | 7388.30 / 7856.26 / 6712.88 | 62 |

State dtype/bytes stable across all 13 inventories: BF16 469223040 BF16; INT8 234611520 int8 + 14663220 FP32; Split-Max 234611520 int8 + 29326440 FP32 + 3665805 uint8. All arms have 938446080 bytes each FP32 parameters and gradients and identical 766 H2D weight copies (332528615424 bytes), no weight D2H. Shared GPU peak 1.08 GiB throughout; system RAM peaks 26.727/26.758/26.754/26.794/27.599 GiB. Device memory is NVML sampled whole-device, not exact allocator usage; optimizer phase times are host elapsed, not synchronized kernel timings. Offload 364 accepted, zero skips, 420 MiB host peak under 512 MiB cap, zero live post-step. Forward allocation savings 411.43 MiB, backward only 22.99 MiB. Backward remained dominant in every arm. Third operational replication of synchronous offload, not parity proof.

First training losses FP32/BF16/INT8/Split-Max/offload .134691/.189120/.121071/.100759/.150021; means .168736/.168945/.146201/.163602/.164088. Same fixed holdout baseline .18260, no endpoint validation, images or resume. The differences before first update (including differing FP32 controls across historical runs) defeat interpretation as paired learning quality. Do not rank codecs by these loss means. No claim of bitwise equivalence or improved convergence. BF16 is the provisional conservative candidate based on memory/speed and earlier synthetic numerical evidence; INT8 retains largest measured memory saving, Split-Max costs more and must justify its accuracy benefit. At rank16 the persistent state savings are roughly one quarter of rank64 assuming identical targets, so don't extrapolate full training savings or high-resolution fit.

**Gate passed: independent GPU feasibility at rank64/.25 MP/swap22.** Next build a paired numerical diagnostic: actual training-input/noise/timestep/conditioning and initial factor hashes, FP32 repeat control, fixed-input loss/gradient/compact effective BA displacement comparisons, then longer matched training and checkpoint/resume. Separate diagnostics overhead from clean speed measurements. Profile compressed codec/synchronization overhead (one finite check per chunk is a hypothesis, not a measured attribution). Async activation work only after correctness; target backward live buffers as well as forward, since forward-only improvement is not enough. No need to rerun the same five-arm feasibility queue. No combined intervention, auto-budget CUDA, 1 MP, resume or production promotion yet. Preserve raw user archive separately; documentation only.

## Round 4 — paired numerical diagnostic completed; FP32 baseline sensitivity (2026-11-27)

Run geometry-memory-paired-20261127-152657-80aa43; user ZIP/JSON export suffix 20261127-234442, comparison log suffix 234441. ZIP CRC check passed, embedded report/log byte-identical to loose files; all six raw logs exactly matched parsed phase, state, config, loss and pair records. Six arms completed 13 updates each (78 total); 39 successful phase scopes and 13 inventories per arm, zero observed CUDA retries/OOMs. Queue elapsed 8.77 minutes. Same rank64/alpha64, .25 MP, swap22, 192 MiB, seed42, LR .0002, H2D stream, INT8 base; no endpoint validation, sample generation or resume. All six initial manifests (264 modules) and all 13 actual input hashes/timesteps identical. All first losses .19066694378852844. This resolves the prior input-pairing problem, not full numerical equivalence.

| Arm | Mean train loss | Mean absolute per-step loss difference vs FP32 | Peak allocated saved MiB | Peak reserved saved MiB | Diagnostic step ms |
|---|---:|---:|---:|---:|---:|
| FP32 control | .174816615 | 0 | 0 | 0 | 2654.4 |
| FP32 repeat | .174714477 | .000297035 | 0 | 0 | 2673.2 |
| BF16 | .174673545 | .000250814 | 462.92 | 216 | 2890.2 |
| INT8-64 | .174816619 | .000186716 | 675.12 | 682 | 2958.8 |
| Split-Max64 | .174826954 | .000260786 | 657.63 | 668 | 3062.3 |
| Sync activation | .174726329 | .000276482 | 19.08 | 284 | 3028.1 |

State bytes reproduced Round 3 exactly. Offload accepted 364, zero skips/live after update, host peak 420 MiB. Forward allocated saved 407.57 MiB; backward saved 19.08 MiB. Timings include instrumentation; clean speed comparisons remain Round 3. Losses are paired by inputs, but later factor states are not identical; no training-quality ranking from these short trajectories.

Across sampled modules, aggregate relative Frobenius errors (%) = sqrt(sum squared errors / sum squared reference norms), computed separately for raw factor gradients and effective scaled BA step displacement; not an all-model estimate. Sample: blocks.0.attn.gate, blocks.18.mlp.up, blocks.3.mlp.up, tproj.1, txtmlp.3.

| Arm vs FP32 | Step1 grad / update | Step2 grad / update | Step13 grad / update |
|---|---:|---:|---:|
| FP32 repeat | 1.275 / 20.538 | 115.336 / 102.476 | 50.644 / 87.940 |
| BF16 | 1.316 / 20.927 | 116.305 / 98.977 | 63.936 / 82.956 |
| INT8-64 | 1.240 / 20.528 | 105.726 / 92.082 | 64.436 / 87.750 |
| Split-Max64 | 1.245 / 20.484 | 106.437 / 101.879 | 66.837 / 91.079 |
| Sync activation | 1.271 / 20.729 | 106.848 / 101.447 | 63.320 / 93.515 |

FP32 self-comparison zero gradient and ~1e-15 relative displacement QR roundoff. FP32 A/A step1 per-module gradient errors .634–2.532%, displacement errors 1.801–106.949%. tproj.1 specifically: gradient error 1.2345%, displacement error 106.949%, norms .00677637 vs .00678591, cosine .428898 computed by polarization of the recorded norms/errors. Every candidate shows the same tproj first-step pattern (106.734–108.988%, cosine .408–.432). Compressed momentum starts empty and current step uses unquantized EMA; therefore first-step divergence cannot be attributed to accumulated compressed momentum. The first loss is equal but backward gradients differ already before optimizer update. Small gradient perturbations may be amplified by Muon orthogonalization in poorly conditioned/rank-deficient directions; backend nondeterminism, streaming/recompute ordering, and/or implementation issues need isolation, not assumption. Later gradients are measured on diverged parameter trajectories, not local fixed-state errors. One FP32 repeat is not a statistical noise bound. No candidate is demonstrated numerically equivalent or inferior on this evidence alone. No raw gradient tensors in supplied ZIP; local .pair-reference may be retained for follow-up. 

**Gate: paired inputs and execution passed; numerical attribution blocked by A/A baseline.** Do not repeat the unchanged six arms. Next proposed: fixed-gradient optimizer replay using identical factors, full gradient tensors, LR, clipping and initial state; compare FP32 repeats, codecs over multiple updates, isolate state encoding vs NS amplification, report per-module norms/cosines and cumulative BA drift. Separately test repeated actual Krea2 forward/backward at identical frozen adapter state with no optimizer update; compare usual cuDNN SDPA vs supported deterministic/math backend, with streaming held fixed and later compare supported alternate streaming/checkpoint path if needed. Record reproducibility vs memory/speed cost; do not force deterministic production defaults or diagnose a race solely from this result. Then matched multi-epoch validation/visual and resume before production or combining techniques. BF16 remains provisional speed/memory candidate, INT8 memory leader; Split-Max accuracy advantage not established. No new training run performed by this documentation update.
