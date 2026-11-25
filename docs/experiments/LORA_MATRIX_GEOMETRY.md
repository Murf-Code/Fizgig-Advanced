# LoRA matrix geometry research

Isolated from release and Global Replay. No trainer behavior changes in this first increment.

## Initial laboratory
- Compact QR/SVD diagnostics, exact adapter difference norm and offline spectral balancing/truncation.
- Synthetic linear regression screening: AdamW, batched Muon, PoLoRA, LoRA-TSD with and without scalar balancing. Zero-up initialization, identical data and initial factors, explicit LR sweep.
- CPU/CUDA CLI; JSON records individual failures rather than silently substituting optimizers. The diagnostics check full-product references only on small matrices. BF16 tests use rounded values with FP32 decomposition; not BF16 optimizer validation.

Run: `python scripts/bench_lora_matrix_geometry.py --device cpu` from the repo root. On CUDA use `--device cuda --rank 64`. This is not the diffusion benchmark. Synchronization and diagnostic overhead are intentionally included in timings; peak allocation is process-wide, not isolated optimizer state.

## Gates before diffusion testing
1. Verify compact calculations, exact update cap including quadratic term, zero/rank-deficient behavior, Muon equivalence, optimizer resume and memory.
2. Audit source for paired parameter eligibility, EMA and dropout. Do not change live factor basis without consistent optimizer/EMA state handling.
3. Implement separately validated Riemannian preconditioning, spectral balancing, Riemannion and RoM prototypes using primary papers. Do not label a generic preconditioner as a reproduced paper.
4. Screen on multiple synthetic shapes, seeds and conditioning regimes. Select candidates based on convergence, stability and cost. Do not infer image quality from synthetic loss.
5. Build initial Krea 2 comparison: rank/alpha 32, five epochs, all eligible blocks, fixed seeds/data/holdout/samples, no replay/selective/adaptive LR/EMA/rank dropout. Tune learning rates on separate calibration data; report equal-step and equal-wall-time comparisons. Exclude SOAP, Prodigy, Adafactor. Export JSON, readable log and sample ZIP.
6. Offline rank 16/8 compression from rank 32; later rank 32/16/8 from rank 64, preserving alpha/r scaling and testing same prompts/seeds plus held-out quality. Never overwrite original adapters. Rank reduction and better learning are different outcomes.

No GPU evidence yet. The balanced-factor helper is an offline mathematical primitive, not a live optimizer. No claim of improved quality or file size.

## Increment 2: scaled-gradient and spectral experiments

Added four stateless synthetic controls: SGD, damped opposite-Gram scaled SGD, and both with spectral rebalancing every five steps (only full-rank FP32/64 factors). These are experimental controls, not a faithful implementation of Riemannion, RoM, Scaled AdamW, or spectral balancing with the best existing baseline. No trainer integration. Damping and cadence are hypotheses. See Zhang and Pilanci, ICML 2024, https://proceedings.mlr.press/v235/zhang24ax.html.

The standalone script accepts --methods, --lrs, and --condition (target singular-value spread). Example: `python scripts/bench_lora_matrix_geometry.py --device cuda --rank 64 --condition 100 --methods sgd scaled-sgd sgd-spectral scaled-sgd-spectral --lrs 0.001 0.01 0.1`. Run multiple seeds. Synthetic controls do not establish diffusion quality. The new pytest check covers rotation equivariance, zero-up initialization, state resume, and counterexamples to the linearized cap and rank-deficient QR assumption. These counterexamples are mathematical, not proof of a particular TSD runtime failure. Production optimizer implementations remain unchanged.

## Increment 3: independent audit repairs

Credit: Grok's independent audit and round-2 reproduction package (2026-11-25) identified the zero-start instability, finite cap and projection issues, and demonstrated PoLoRA chunking. ChatGPT integrated the corrections on this isolated research branch.

LoRA-TSD now uses rank-controlled polar initialization, one-factor warm steps below twice the effective cap, rank-revealing masked factor bases, and a conservative path-safe triangle bound plus stored-dtype acceptance including any balancing drift. The cap is per pair, in scaled adapter units (alpha/r times strength); the family trainer supplies actual adapter scales. The scalar warm threshold, FP64 compact checks and 12 retry limit are correctness-first experimental choices, not established best settings. If no representable safe update is found, the weights are held unchanged; momentum has still observed the gradient. No full weight products in the optimizer. This is not a claim of an exact quartic maximizing step, rank growth, or GPU performance. EMA remains excluded from the initial benchmark; scalar balancing can still change the factor gauge. FP64 QR/eigh and host checks are deliberately retained pending a performance pass.

PoLoRA uses a default 128 MiB estimated per-chunk workspace budget (no new UI). The optional optimizer argument chunk_memory_mib=0 disables chunking for reference comparisons. The heuristic excludes persistent state, allocator and compiler overhead and cannot guarantee a 12 GB fit; even a single pair can exceed the budget. Skips, norms, and the original checkpoint layout are preserved. Tests cover low-rank and tiny starts, FP32/BF16 storage, alpha scaling, resume, chunk parity and skipped pairs. CUDA and diffusion tests are still required.

## Increment 4: queued Krea 2 feasibility pilot

Ported the batch enqueue and report design from Fizgig Advanced's test/global-replay-comparison branch, without its Selective Replay training changes. The research branch adds a **Queue geometry test** button next to Add to queue. It preflights all five runs before appending them to the existing queue; no separate scheduler or simultaneous GPU training. The fixed test uses AdamW8bit (not FP32 AdamW), Batched Muon, budgeted PoLoRA, and repaired LoRA-TSD with/without balancing. Rank/alpha 32, five epochs, INT8 base, 22 swapped blocks, 0.25 MP, batch one, fixed holdout and prior replay sample prompts. These choices target feasibility, not a guaranteed 12 GB fit. Rates are provisional recipe-based settings; not calibrated optima. EMA, replay, adaptive LR, per-image LR and automatic OOM setting changes disabled. OOM is reported as a failure. Standard queue stop/pause behavior applies; do not edit fixed runs. Report folder: matrix_geometry_comparisons/<group> under the default output directory. JSON, combined log and final ZIP of samples and logs. The pilot does not yet include per-epoch spectral metrics, compression, Scaled AdamW or Riemannion. CPU integration tests and GPU run still required.

## Rank adjustment

The initial queued diffusion pilot uses rank 32 / alpha 32 to reduce memory pressure on the 12 GB RTX 4070 Ti. All other pilot controls and provisional learning rates are unchanged. Rank 64 is reserved for later experiments if feasible. This change applies to newly queued runs, not previously saved queue entries. A short GPU preflight is not yet implemented; actual VRAM fit remains unverified.

## Rank 64 fit probe

The **Test rank 64 fit** button queues one PoLoRA run using rank/alpha 64, LR 0.02, magnitude 3, compile off, 128 MiB estimated chunk budget, and otherwise the pilot's fixed INT8/block-swap/data controls. It stops after at most 50 updates or one epoch if shorter, without sample generation or final checkpoint export. The report captures CUDA process allocator peak allocated/reserved bytes and training wall time; normal performance telemetry may also record device-wide memory. The five-run rank-32 queue is unchanged. Automatic OOM recovery remains disabled. A successful short probe is not proof of five-epoch fit or fit for other optimizers. The probe and integrated regression suite still require execution.

## Resolution and memory feasibility sweep (after first CUDA result)

The first rank-64 chunked PoLoRA probe on RTX 4070 Ti completed 29 updates at 0.25 MP, 264 modules, 234.6M trainable parameters, 7.70 GiB allocated / 8.23 GiB reserved CUDA peak and 11.48 GiB sampled device peak, 6.95 seconds per training step. No post-training validation or unchunked GPU comparison was performed. Credit: Grok's independent allocation audit motivated the chunking intervention.

Replace the visible rank-64 button with **Test PoLoRA resolution**: rank 16/32 each at 0.25, 0.50, 0.75, 1.00 MP (grouped by resolution for cache reuse), then matched rank-64 0.25 MP controls at 128 MiB and 0 (unchunked). All cases use the previous one-epoch/up-to-50-update probe, no samples, fixed LR 0.02, alpha=rank, magnitude 3, same seed and holdout. The old rank-64 API remains compatible with saved queues. The five-run optimizer pilot is unchanged. OOM is recorded, not automatically recovered by changing settings; existing queue proceeds after a training failure. A start/preflight failure may still hold the queue. Cache preflight checks resolution; actual bucket sizes are included in the report (training and validation buckets in log order, no-upscale may reduce real pixel counts). Baseline validation is not a final metric. Sampled system RAM and training-process RSS peaks are added to performance telemetry. Windows PDH shared GPU usage is best-effort, all adapters of the training process; missing values mean unavailable, not zero, and shared memory usage is not itself evidence of paging. NVML device usage and PyTorch reserved fallback are labeled separately. No long-run quality conclusion or GPU regression validation yet.

## Block-swap speed sweep

After the completed 10-case resolution baseline, add a separate **Test block swap speed** button.
Four short rank-16/alpha-16 PoLoRA cases at 0.25 MP: 22 (fresh baseline), 20, 18, 16 swapped blocks, in descending order. Same 128 MiB chunk budget, seed, LR 0.02, INT8, holdout, batch size 1, standard training and no previews; at most 50 updates or one epoch. The 22-block repeat helps identify run-to-run variation. The lowest swap count is a stress case, not a recommended default. OOM is data, not a reason to automatically alter settings. Existing reports store BLOCKS_SWAP and case labels. Choose the fastest configuration with acceptable VRAM margin, not the lowest swap count alone. Existing resolution and five-run optimizer queues are unchanged. No optimizer or training-loop modifications; newer state-view/index-cache research remains isolated. CPU test added; Windows/CUDA execution pending.
