# Matrix geometry research register — updated 2026-11-27

Research only. Do not merge research code into release, WIP or Selective Global Replay. The completed Round 10/Muon-ME experiments are historical; consult the current memory pilot below before queueing. This register records prior experiments; it does not imply that their standalone prototypes have been integrated or that synthetic quality transfers to diffusion. Original brief: LoRA geometry, efficient learning, low VRAM, high resolution on a 12 GB GPU, compact standard-compatible export.

## Evidence and priorities

| Workstream | Evidence / status | Next gate |
|---|---|---|
| PoLoRA-ME | Chunked implementation and state views already in branch; rank-64 GPU fit. CPU temporary allocation reduction 2611 to 47 MiB at smallest tested chunk, not measured GPU peak savings. | Matched rank-64 CUDA chunk sweep, optimizer vs backward peaks, timing and parity. |
| Muon-ME | Isolated rank-64 CUDA gate passed: 128 MiB warm scratch 1358.16 to 60 MiB, peak reserved down 1.545 GiB, optimizer time down 13.68%; reference repeat and shape-local bitwise identical; chunked momentum exact under supplied gradients, small weight differences. See [CUDA results](MUON_ME_CUDA_RESULTS_20261125.md). | Opt-in integration and six closed-loop runs completed; see results below. Extend only after independent state/activation feasibility gates. |
| TSD-ME | Round 13 compact clipping prepass standalone: 48 trajectories/1440 steps bitwise equal, 11 selected checks and 12 resume checks. Estimated rank-64 cache 903.22 to 25.01 MiB, not GPU peak. | Review cumulative R11–13 changes, then CUDA. |
| Automatic budgeting | CPU-tested optional Muon-ME preflight for FP32 and compressed state, reserve and hysteresis; integrated at dcf7cd0. Disabled in fixed pilot. | CUDA calibration; PoLoRA/TSD-specific estimates and eventual shared planner. No silent setting changes or partial-step retry. |
| Activation offload | Three synchronous Krea2 GPU runs completed 13 updates each; Round 3 at 192 MiB with complete telemetry: forward allocated −411.43 MiB, backward −22.99 MiB, reserved −284 MiB, step time +14.92%; host peak 420 MiB, 364 inputs, zero live post-step. | Paired fixed-input/initialization parity, then async/PCIe coordination and backward live-buffer profiling; no quality or speed gain established. |
| Optimizer-state compression | CPU stages 1–2 archived. All three compressed modes pass 13 CUDA updates at rank64/.25 MP/swap22/192 MiB. BF16/INT8/Split-Max measured state savings 447.49/657.25/639.77 MiB, allocated peak reductions 466.00/681.42/659.03 MiB, step penalties +7.36/+15.40/+18.74%. INT4 deferred. | Paired numerical/quality/resume; profile codec overhead. BF16 provisional conservative candidate, INT8 memory leader; PoLoRA/TSD separately. |
| Scaled Adam | Gradient-first damped preconditioned FP64 Adam, no WD; synthetic gains during growth/conditioning; not gauge invariant. | Opt-in trainer version, Adam control, LR matching. |
| RoM/Riemannion | Literature and compact mathematical prototypes, not complete published implementations. | Defer full ports pending benefit. |
| Learning-regime awareness | Adaptive Muon SNR and displacement coherence tested; SNR can harm growth; cosine decay beat tested coherence controller. | Keep schedule controls; no default adaptive policy. |
| Rank intelligence | Real spectral analysis, CPU rank allocation and activation/functional sensitivity prototypes. | Actual held-out activation/downstream probes; dynamic rank redistribution, dormant reseeding and spectral regularization remain open. |
| Initialization | Gradient sketch with zero B preserves base; matched synthetic calibration improves early learning, unrelated calibration may not. Paired initialization breaks under independent dropout. | Calibrate without dense weight gradient; standard init control; paired remains experimental. |
| Rotation-compatible EMA | Compact weight-space EMA and balanced alignment tested on CPU; ordinary factor EMA fails under gauge changes. Repeated truncation approximate. | Benchmark standard EMA and cost; do not rotate live optimizer state casually. |
| Conditioning | Compact SVD balancing robust in ill-conditioned synthetic BF16 factors, not universally beneficial. | Threshold and quality tests. |
| Rotation/quantization | 32 modules per epoch from two rank-16 AdamW checkpoints (64 observations); full sweep timed out. Epoch 2 INT8 0.589% and INT4 10.727% relative effective-weight error with best gauge search; ternary 47.516%, binary 67.865%. Full-rank INT4 better than half-rank INT8 at similar bytes on sample. | Thin-layer fix, functional validation, mixed precision, packed loader/export; no visual quality claim. |
| Live low-bit factors | Pending direct packed-state CPU prototype, distinguish QAT from true persistent low-bit storage. | Rounding, scale drift, checkpoint/RNG, export; CUDA scratch/grad/state budget and quality. |

## Archive index

CPU memory stages 1–2 are now committed at 6010b0f (see [CPU results](MATRIX_MEMORY_CPU_RESULTS_20261125.md)). Other standalone evidence packages below are conversation artifact pointers, not necessarily vendored:
- functional_rank_research.zip
- adaptive_muon_research.zip
- tsd_compact_prepass_research.zip
- muon_me_research.zip
- memory_efficiency_research.zip
- optimizer_math_research.zip
- initialization_stability_research.zip
- rotation_quantization_research.zip
- fizgig_round10.zip through fizgig_round13.zip (independent Grok audits; credit Grok for proposed TSD refinements and audit, not trainer integration)
- rank_allocation_experiment.py / rank_allocation_results.json; activation_rank_experiment.py / activation_rank_results.json / activation_rank_stress_results.json

Keep source, tests, raw results and failure logs together. No checkpoint binaries, user datasets, generated images or private logs in Git. Archive filenames here are pointers, not a claim their bytes are in this repository.

## Experimental sequencing

1. Review incoming GPU report before altering the existing queue.
2. Separate memory-only changes from optimizer-learning changes. Measure peak allocated/reserved, device/shared memory, host RAM, optimizer scratch, wall time and quality.
3. Keep Muon and AdamW8bit reference, identical data/seed/holdout and sample prompts; repeat controls to estimate noise. Do not repeat interrupted AdamW unnecessarily.
4. Test actual 0.75–1 MP inputs, not merely target resolution settings. Cross-family validation and standard LoRA export remain outstanding.
5. No automatic production enablement until numerical, resume, CUDA and quality gates pass.

### Muon-ME closed-loop queue
Opt-in factory key `muon_me`, default chunk budget 128 MiB, fixed 22-swap rank-64 quality arms (five epochs) and two one-epoch rank-64 16-swap stretch arms. Earlier rank-16 16-swap and rank-64 22-swap PoLoRA results are separate, not proof of the combined setting. No compressed momentum or adaptive policy included. Preserve OOM failures. The VRAM planner conservatively uses the original Muon estimates until full-training data arrive.

### Muon-ME closed-loop results
Six runs completed. See [full training results](MUON_ME_TRAINING_RESULTS_20261125.md). ME128 faster optimizer, modest sampled full-training memory improvement, no proven quality difference. Rank64/16 swap completes but severe shared-memory pressure and 7.24x slowdown versus own 22 swap arm. Next phase-resolved profiling and activation/state-memory experiments, not a repeat of this queue.

## Current memory architecture gate — 2026-11-27

- [Integration protocol](MEMORY_ARCHITECTURE_INTEGRATION.md) and [GPU pilot evidence](MEMORY_ARCHITECTURE_GPU_PILOT_20261125.md) supersede the earlier prospective statuses for memory work. Source fixes: 1ff0c21 (dataset-group shadowing, telemetry retention, per-step loss); d7df5b3 (rank-64 scratch preflight and common 192 MiB cap).
- Both historical 128 MiB queues are now fully accounted for; the second confirms complete telemetry and repeated activation-offload feasibility. See 18a0c21 and the pilot evidence document. The first-step loss discrepancy predates optimizer updates; no paired numerical or quality claim.
- Round 3 completed Nov 27: all five arms, 65 updates, complete telemetry, no allocator retries/OOMs. See [Round 3 results](MEMORY_ARCHITECTURE_GPU_PILOT_20261125.md). Fixed 192 MiB queue feasibility closed; do not repeat unchanged.
- Round 4 paired A/A, initialization and actual input hashes, sampled gradient/effective displacement diagnostic completed. Numerical attribution remains blocked by FP32 repeat sensitivity; see latest gate below.
- Only after individual gates: async bounded activation transfers, combined memory techniques, swap20/21 profiling, actual 0.75–1 MP at rank16/32/64. Keep LittleBit frozen-base codecs, Selective policy and SOAP LR calibration independent. Consider SOAP budgeting after its separate calibration; do not port prematurely.
- Practical target: sustained genuine 0.75–1 MP at rank16/32 on 12 GiB with acceptable speed, numerical stability and standard LoRA export. Rank64 genuine 1 MP remains an ambitious stress target, not the user's only success criterion. The original user explicitly prioritized resolution and noted that rank16 learns well in Krea2. See [recovered resolution baseline](https://github.com/Murf-Code/Fizgig-Advanced/blob/07f34025d3cc254f1f1a5f0db1274b0f0892335f/docs/experiments/LORA_MATRIX_GEOMETRY.md).

### Historical evidence recovered 2026-11-27
Source: original chat export covering Nov 25; previous reported measurements, not fresh raw-archive analysis. The completed ten-arm PoLoRA sweep (2323a51) yielded 290 updates, no OOM. Matched rank64 128 MiB vs unchunked: allocated 7.70 vs 8.72 GiB, reserved 8.20 vs 10.94 GiB, device peak 10.50 vs 11.98 GiB, 6.96 vs 8.02 s/update. This measured benefit is from chunking, not later state views. Rank16 target .75 MP: actual mean .693 MP, 5.88 s/update, 10.82 GiB device peak, 98.3% utilization. Target 1 MP mean only .727 MP (one .984 MP image); do not claim sustained true 1 MP. Rank16 swap16 at .25 MP passed; separate rank64 Muon swap16 slow stress result does not contradict it.

Earlier rank16/swap18/.25 MP optimizer quality comparison: five complete arms (145 updates each), AdamW8bit interrupted by power loss (58 updates, telemetry corrupted). Reported final validation: Muon .19573, PoLoRA baseline .19590, state-view .19572, TSD balanced .35537, unbalanced .22674; common start .19972. Mean step seconds 2.479/3.486/3.473/5.998/5.937 respectively. No meaningful speed gain from state views at this tested rank; no quality advantage established for Muon vs PoLoRA. TSD 76,560 pair updates, no recorded cap violations, max .08123 vs .3, yet worse validation. Do not equate stability with learning quality. Round 10 adaptive TSD queue remains unreviewed/pending; do not infer completion from later independent CPU audit. User evidence pointers: matrix-geometry-report(2).json and results(1).zip for resolution; earlier quality archive and quality-adamw8bit.zip for interrupted control; consult original transcript for provenance. No raw images, logs or user paths committed.

Rank allocation caution: two recovered rank16 AdamW8bit checkpoints (29/58 steps, 264 pairs) had reported median effective rank 14.17/12.63; theoretical rank8 retained 92.17% total weight energy at step58. Later distinct rank32 replay checkpoints were nearly full entropy rank (>31 median across arms) and no final raw module retained 90% energy at rank16. Do not pool different optimizer/training recipes or infer quality from spectral energy. Functional finite-ablation and held-out visual/downstream checks remain necessary. This is historical transcript evidence, not a reopening of Selective research.

### Round 4 implementation
Paired numerical test is now implemented experimentally (see integration protocol), not GPU validated. Six arms: FP32 A/A, BF16, INT8-64, Split-Max64, sync activation. Actual inputs and initial factors must hash identically. Sampled exact compact displacement and raw gradient comparisons at steps 1, 2, final; diagnostic timings not clean throughput. No learning-quality conclusion. Next action run **Test memory correctness** on the research branch with the same dataset; preserve all failure records and the local reference until the queue ends.

### Round 4 result and revised gate
All six arms 13/13, 78 updates, complete telemetry and matched 264-module initialization plus actual input fingerprints. First loss equal; mean losses within 0.083% of reference. However FP32 A/A first sampled aggregate gradient/update error 1.275%/20.538%; tproj.1 displacement 106.949% with similar norms (cosine .429). Step2/13 large trajectory differences; candidates comparable in magnitude, not proven equivalent. See [full Round 4 evidence](MEMORY_ARCHITECTURE_GPU_PILOT_20261125.md). Next isolate fixed-gradient optimizer replay and repeated fixed-state actual backward before quality/resume, not another unchanged six-arm run. No production promotion or combined intervention. User's practical resolution target unchanged.

### Round 4 CPU follow-up
[CPU numerical isolation](MUON_NUMERICAL_ISOLATION_CPU_20261127.md) completed: synthetic low-rank gradient perturbations strongly amplified by NS, including FP32 arithmetic; exact-gradient replay separates compressed momentum error from backward variation. BF16 rank1 replay median cumulative error 35% vs INT8 106% at 13 updates; full-rank errors below 1%. These are not actual training gradients or quality evidence. Next capture real spectra/gradients and isolate fixed-state GPU backward. No defaults changed.

### Next major milestone (user priority)
Defer rank-aware NS investigation; preserve normalized replay observation in the CPU findings. Proceed with bounded asynchronous checkpoint-input offload and backward live-memory analysis as a separate experimental intervention. Prioritize transfer lifetime correctness, pinned host cap, event ordering and PCIe contention. Compare FP32 control/synchronous/asynchronous with actual activation telemetry; do not claim numerical equivalence from Round 4. Then combine only independently validated changes and test actual 0.75–1 MP at rank16/32, not merely nominal bucket target. No new implementation or GPU result in this documentation update.

### Seven-rank CPU sweep completed
See [full results](MUON_SEVEN_RANK_CPU_20261127.md). Similar normalized error curves confirmed approximately, especially INT8/Split-Max; damping and NS3 reduce relative errors vs their own uncompressed references but alter update maps. Defer further optimizer work, continue activation architecture.

### Round 5 implementation ready for isolated GPU test
[Async activation offload protocol and review request](ASYNC_ACTIVATION_OFFLOAD_20261127.md). Seven CPU tests pass, one CUDA skipped. Four paired arms, no compressed momentum, no new optimizer strategy. Click Test async activation, same dataset. Actual full-model parity and clean speed pending; backward live-memory profiling remains separate.

### Round 5 GPU feasibility result
[Async activation GPU results](ASYNC_ACTIVATION_OFFLOAD_20261127.md): 4/4 arms, 52 updates, tiny CUDA preflight 3/3, no allocator OOM/retries, exact accounting 364 restores and 420 MiB host peak, zero live/waits. Async step 2.8194 s vs sync 2.9909 and control 2.6064 (44.6% of sync overhead removed). Async forward -405 MiB, backward only -14.25 MiB, reserved -204 MiB vs control, +80 MiB vs sync. Shared GPU sampled higher. A/A first update divergence repeated; async step2 sampled gradient error 490% vs A/A 152%, not attributable on diverged trajectories. Operational feasibility passed, full-model numerical gate open. Next fixed-state repeated actual backward + live-memory profiling, clean timing after correctness. No combined/production promotion.
