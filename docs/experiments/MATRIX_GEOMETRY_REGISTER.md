# Matrix geometry research register — 2026-11-25

Research only. Preserve the pending Round 10 GPU queue and do not merge into release, WIP or Selective Global Replay. This register records prior experiments; it does not imply that their standalone prototypes have been integrated or that synthetic quality transfers to diffusion. Original brief: LoRA geometry, efficient learning, low VRAM, high resolution on a 12 GB GPU, compact standard-compatible export.

## Evidence and priorities

| Workstream | Evidence / status | Next gate |
|---|---|---|
| PoLoRA-ME | Chunked implementation and state views already in branch; rank-64 GPU fit. CPU temporary allocation reduction 2611 to 47 MiB at smallest tested chunk, not measured GPU peak savings. | Matched rank-64 CUDA chunk sweep, optimizer vs backward peaks, timing and parity. |
| Muon-ME | Isolated rank-64 CUDA gate passed: 128 MiB warm scratch 1358.16 to 60 MiB, peak reserved down 1.545 GiB, optimizer time down 13.68%; reference repeat and shape-local bitwise identical; chunked momentum exact under supplied gradients, small weight differences. See [CUDA results](MUON_ME_CUDA_RESULTS_20261125.md). | Integrate opt-in only; matched closed-loop training, full-training memory and quality, device/shape coverage. |
| TSD-ME | Round 13 compact clipping prepass standalone: 48 trajectories/1440 steps bitwise equal, 11 selected checks and 12 resume checks. Estimated rank-64 cache 903.22 to 25.01 MiB, not GPU peak. | Review cumulative R11–13 changes, then CUDA. |
| Automatic budgeting | Standalone conservative preflight, state/scratch estimates and hysteresis. | Shared manager across optimizers; CUDA calibration and host budget, no silent training-setting changes or partial-step retry. |
| Activation offload | Krea2 model accepts but ignores cpu_offload; targeted checkpoint-input CPU parity prototype. 1024² modeled 1.39 GiB saved inputs, 2.79 GiB round trip. | Synchronous CUDA with block swap before async; measure PCIe and host RAM, 32/64 GB host targets. |
| Optimizer-state compression | Standalone packed INT8/INT4 Muon momentum; INT8 nearly matched FP32 in synthetic tests, INT4 degraded. | CUDA quality/speed, resume; PoLoRA/TSD separately. |
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

Existing standalone evidence packages (conversation artifacts, not yet vendored in this commit):
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
Opt-in factory key `muon_me`, default chunk budget 128 MiB, fixed 22-swap rank-64 quality arms (three epochs) and two one-epoch rank-64 16-swap stretch arms. Earlier rank-16 16-swap and rank-64 22-swap PoLoRA results are separate, not proof of the combined setting. No compressed momentum or adaptive policy included. Preserve OOM failures. The VRAM planner conservatively uses the original Muon estimates until full-training data arrive.
