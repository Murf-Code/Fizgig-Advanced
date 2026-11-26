# Krea2 frozen-base compression — v4 independent peer-review handoff

**Branch:** research/krea2-littlebit  
**Implementation:** independent-v4-paired-development  
**Previous independently reviewed revision:** dd39965372d564d7a54850db3b55f0c03c933c1f  
**Matrix geometry source:** research/lora-matrix-geometry, e6232036a97a80504526c0bfe4f7c2260c9b890f

## Objective

Discover the trade-off between portable LoRA learning, frozen-base
storage, actual dedicated VRAM, shared GPU memory, block swapping,
and runtime. 96% stock learning gain is the user's aspiration, NOT
an acceptance gate, a predicted outcome, or a claim. Do not discard
a candidate based solely on relative reconstruction error or retention.
The starting checkpoint must be original Raw BF16, not prequantized.

## Evidence prior to v4

The original external pilot's synthetic single-projection transfer
failure remains a negative result, not an end-to-end diffusion finding.
Stock no-adapter MSE ~1.804e-5, stock-trained ~1.674e-5, and
LB2-trained on stock ~0.47321. That pilot is a different implementation.

Astra's review of dd39965 independently reported 30 passing CPU tests,
including four real-HQQ cases. On blocks.0.attn.wk.weight (1536 x 6144,
two synthetic BF16 activation/cotangent seeds), Astra reported these
worst-seed values, NOT results of v4:

| Codec | Packed bytes | Forward relative error | Input-gradient relative error |
|---|---:|---:|---:|
| INT8 | 9,443,328 | reference | reference |
| Q6 | 7,372,800 | 4.22% | 4.08% |
| Q6 Hadamard | 7,372,800 | 3.27% | 3.06% |
| Q6 fitted Hadamard | 7,372,800 | 3.26% | 3.03% |
| Q4 fitted Hadamard | 5,013,504 | 10.72% | 10.29% |
| HQQ | 7,102,464 | 5.79% | 5.52% |
| HQQ Hadamard | 7,102,464 | 5.47% | 5.26% |
| LB2 rank 512 | 493,568 | 74.23% | 75.22% |
| LB2 rank 512 + HQQ residual | 7,596,032 | 4.62% | 4.35% |

Q6 Hadamard is the current leading hypothesis, NOT a winner of actual
LoRA training. The hybrid earns a comparator; direct Q6 Hadamard
dominated it on this one matrix. Grok recommended real activations,
NF4 control, mixed precision, shorter training, and cautious interpretation.
Neither review established end-to-end Krea2 learning retention.

## Implemented changes

1. Common FP32 cross-tile backward accumulation for plain Q4/Q6 and
   geometry variants (individual GEMMs still compute in the input dtype).
2. Native trainer research-only observer. It reseeds fresh LoRA initialization
   and DataLoader shuffle; verifies declared Krea2 LoRA inventory, zero-up
   initialization and FP32 initial-adapter hash. Candidate runs compare
   actual item, latent, conditioning and noise-generator fingerprints and
   timestep against stock, failing on mismatch rather than assuming pairing.
3. Stock capture: on the first four actual optimizer updates, sample up
   to 16 matched token rows per each of 224 eligible block projections,
   with real activations and output cotangents from the loss backward.
   Actual timesteps are recorded; this is NOT yet a stratified timestep
   calibration set. Calibration is from training images, never holdout.
4. Task-factor gradient files before clipping, standard LoRA checkpoints
   at updates 1, 16 and 200 when reached, and the final checkpoint.
   The effective BA difference, norms, cosine and action on captured
   activation rows use the compact QR-based delta_norm from matrix geometry.
   Step-1 task gradients share initialization; later gradients are
   trajectory diagnostics, not estimates of pure compression bias.
5. Screen every block in each tested family on the actual captured rows.
   No 5% error veto. For fixed codecs the family must have 28 compatible,
   finite and smaller payloads; otherwise it remains INT8. Report
   per-projection forward/input-gradient error and cosine, packed bytes
   and fallback accounting. CPU INT8 reference math is still a local
   approximation to the full native CUDA training path.
6. Mixed layout: select one codec per projection family, uniform across
   28 blocks for classic swap compatibility. Reuse the fixed candidate
   archives; minimize a local mean-squared relative forward/input-gradient
   error proxy greedily toward nominal 70% of eligible INT8 bytes.
   This is one engineering storage budget, not a global optimum, a quality
   threshold or a measurement of total GPU residency. Report actual budget
   and exact selected family codecs even if target is unattainable.
7. Separate paired INT8 and Raw BF16 development evaluation. Compare
   actual saved/reloaded adapter inventory to the declared targets, not
   merely two loaded sets to each other. Save per-image/per-timestep
   losses, absolute gains and gain difference. Bootstrap image clusters
   rather than treating timesteps as independent. Retention is descriptive;
   do not treat it as interpretable if stock gain is unstable or fewer
   than four image clusters are available.
8. Native holdout snapshot prefilled and saved before training; disable
   ordinary in-run validation in this research worker. Deterministic
   image-key split into development and reserved confirmation when
   at least four holdout images are available. The queue scores only
   development. Fewer images means no confirmation split. No claim of
   near-duplicate/subject-level disjointness; check dataset before confirmation.
9. Same zero-first tested swap schedule for stock and compact runs:
   0,4,8,12,16,20,22,24,26. Native Krea2 NF4 cannot swap and tries only 0.
   Each genuine CUDA OOM triggers a fresh process and fresh optimizer.
   Each successful attempt records PyTorch allocated/reserved peaks and
   initial free/total memory; cap allocator to initial free VRAM minus
   512 MiB (unless an explicit existing simulator setting is in use).
   Optional Windows typeperf samples dedicated and shared process GPU
   memory, missing counters remain unavailable. Neither allocator cap nor
   sampled counters prove zero paging. No claim of exact minimum swaps.

## New automated development queue

One epoch or 200 actual updates, whichever first, seed 42, rank/alpha 32,
Muon LR 0.00025, constant scheduler, batch 1, 0.25 MP, full checkpointing,
no EMA, selective training, adaptive LR, samples or resume. Two holdout
groups request four images each; native holdout policy may yield fewer.
All training starts at 0 swaps. Evaluation on INT8 uses fixed 22 swaps;
Raw BF16 uses 26. These are evaluation-only counts.

| Case | Role |
|---|---|
| stock (native INT8) | Paired reference + real capture |
| nf4 | Native practical low-memory control |
| q6 | Unrotated control |
| q6-had | Leading tested hypothesis |
| q4-had | Lower-memory comparator |
| hqq-had | HQQ alternative |
| lb2-r512-hqq | Binary-plus-residual comparator |
| mixed | Greedy uniform-family mixed precision |

The longer original codec variants remain in the repository; they are not
queued. All feasible cases receive short training; no automatic numerical
quality cutoff or automatic five-epoch promotion. Compression conversion
is CPU-heavy, especially HQQ, and archive creation still accumulates packed
weights in host RAM. Fixed cases can differ in coverage if incompatible
families fall back; report coverage before comparing codecs. NF4 is not
part of the custom mixed codec because it has a different runtime layout.

## Verification and limitations

Local execution after v4: three pytest paired-metric tests passed (0.12 s);
two mixed-selector smoke checks passed (reachable and unreachable budgets).
The three local source/test file Git blob hashes matched committed blobs
b95741f995a6be68e601303b585b205872636325,
4a340bdd38c33be1b8c0ffd54f2667d20ae8d3c6, and
b3514d9a1f78696e65c659cfc0a2ae4b71ac81e9.
This does not reproduce Astra's 30-test run against dd39965.
The other new staged tests, full native trainer, real capture hooks,
HQQ, raw deployment evaluation, Windows typeperf and CUDA swap
integration were NOT executed here. The sandbox cannot clone GitHub
(DNS), has no CUDA, and no real Krea2 checkpoint was available locally.
No new real-model numerical findings are asserted.

Remaining: confirm real stock calibration completeness under non-reentrant
checkpointing; compare resident versus swapped packed layer and actual
optimizer updates on CUDA; verify data order fingerprints, all 264 declared
LoRA targets, full raw evaluation, and native NF4 fit. Instrumented training
wall time includes capture, hashing and snapshot I/O; do not claim a clean
throughput comparison. A fixed/common swap, no-observer throughput test,
explicit task-gradient calibration at stratified timesteps, a genuinely
common-coverage ablation, frozen activation-weighted correction,
multi-seed confirmation and automated full-run promotion are not yet built.

## Proposed follow-on after successful short runs

- Inspect actual gains (including negative values), memory, swaps and time;
  select at most two finalists from the trade-off frontier, not by 96%/90%.
- Run matched common-swap timing without observer I/O, and validate
  resident versus swap/recompute numerics. Test shared-memory spill behavior.
- Capture separate real development calibration at fixed early/middle/late
  timesteps, test seeded sign-Hadamard and activation-aware precision.
- Run matched five-epoch experiments, then evaluate untouched confirmation
  on both deployment bases with multiple paired seeds; decide how much
  compression is useful. Avoid claiming population inference from a tiny
  image holdout.

## Independent review request

Please review current branch rather than the obsolete 18-arm recipe.
In particular:

- Verify native training callbacks are invoked at precisely the intended
  boundaries and do not alter ordinary training. Can non-reentrant
  checkpoint recomputation duplicate or miss output hooks?
- Verify real activation/cotangent alignment, complete target inventory,
  hash pairing and stable seeds, and that holdout images never train.
- Check model loading, packed storage lifetime, classic swap buffer reuse,
  residual backward, BF16 deployment evaluation, and NF4 no-swap handling.
- Check the mixed precision allocation objective, identical coverage
  comparisons, denominator stability and lack of confirmation leakage.
- Identify the shortest actual CUDA smoke test and any blocker to running
  the queue on 12 GB. Please execute CPU tests and, if available, one real
  stock capture, one short Q6 Hadamard comparison and both deployment scores.
- Recommend changes with concrete code locations and evidence; explicitly
  distinguish findings from hypotheses. Do not infer any retention result
  for v4 until the paired training/evaluation actually runs.

Useful files: native_codec.py, geometry_codec.py, research_trace.py,
prepare_case.py, mixed_selector.py, task_diagnostics.py, paired_metrics.py,
native_runner.py, fit_queue.py, wddm_memory.py (in this directory);
src/fizgig/families/train.py, src/fizgig/training/val_loss.py,
src/fizgig/web/littlebit_benchmark.py; checks/test_littlebit_staged.py,
checks/test_littlebit_paired_metrics.py and the previous geometry/fit tests.

Suggested PowerShell CPU check (from repo root):
```powershell
.\venv\Scripts\python.exe -m pytest -q checks/test_native_littlebit_codec.py checks/test_littlebit_geometry.py checks/test_littlebit_fit_queue.py checks/test_littlebit_paired_metrics.py checks/test_littlebit_staged.py
```

Credits: Samsung LittleBit/LittleBit-2 research (paper-inspired independent
simplified binary code); Fizgig HQQ solver integration by rintic-13,
upstream commit 85c2043bd6fdc6b0e8e34404bd8e15a04c4bdea6; prior
Grok and Astra reviews provided by the user.
