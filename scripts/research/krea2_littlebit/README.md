# Krea2 frozen-base compression research

This branch is an isolated research track, not a new production precision option.
Start from Krea2 Raw BF16, never from an INT8, FP8, NF4 or already compressed checkpoint.
Only the frozen DiT is compressed; native LoRA weights, rank, optimizer, serialization
and stock-model inference remain unchanged.

## External pilot and licensing

Use the supplied krea2-littlebit-pilot-v2.zip as a separate local research dependency
(SHA-256 0429d848052b72251b61e9ce2095945c5d445a51f35009aeea378e6be8a35f13).
Extract it outside the repository. Its NOTICE.md identifies Samsung-derived material
under CC BY-NC 4.0; do not vendor that material into this Apache-2.0 project or treat
this branch as clearing commercial rights. The original pilot remains unchanged.
This folder contains independent experiment orchestration only.

References:
- Lee et al., LittleBit (NeurIPS 2025): https://research.samsung.com/research-papers/LittleBit-Ultra-Low-Bit-Quantization-via-Latent-Factorization
- Lee and Kim, LittleBit-2 (ICML 2026): https://proceedings.mlr.press/v306/lee26y.html
- Reference implementation: https://github.com/SamsungLabs/LittleBit

## Evidence at kickoff (from the supplied pilot, not reproduced full-model here)

At identical rank-128 packed storage on block 0 attention key, LittleBit and
LittleBit-2 had relative weight errors of 0.7432 and 0.5232. At rank 512,
LittleBit-2 was 0.3890 (0.3772 after 48 local reconstruction steps).
The matched three-seed synthetic-activation rank-16 adapter transfer experiment
failed: stock INT8 held-out MSE without adapter 0.00001804, stock-trained
adapter 0.00001674, LittleBit-2-trained adapter 0.47321 when reloaded on stock.
These are projection regressions, not diffusion quality scores.
Residual Q4/Q6 improved fidelity, but first LoRA gradient cosine remained
0.029/0.066, with storage 58.63%/83.61% of INT8 for that projection.
Do not promote this rank-512 configuration to lengthy training without
improved screening. Real model tensors and teacher adapter are not included.

## First milestone: projection screen

From the Fizgig checkout, in PowerShell (adjust paths):
```powershell
$py = ".\venv\Scripts\python.exe"
$pilot = "D:\research\krea2-littlebit-pilot-v2"
$raw = "D:\models\krea2_raw_bf16.safetensors"
& $py scripts/research/krea2_littlebit/screen.py --pilot $pilot --raw $raw --key blocks.0.attn.wk.weight --key blocks.14.attn.wk.weight --key blocks.27.attn.wk.weight --rank 128 --rank 512 --out littlebit-screen.json
```
The screen compares both LittleBit initializations at equal rank/bytes plus
Q4 and Q6 controls, using current Fizgig native INT8 forward/backward from
the same Raw matrix and a BF16 raw reference. It reports actual encoded bytes,
forward drift and input-gradient drift/cosine. By default it uses deterministic
synthetic rows: these are diagnostic only. Optionally supply
`--activations heldout.safetensors` with CPU tensors named
`<weight key>.x` and `<weight key>.dy` (same number of rows, last dimension
input/output width respectively), collected from a stock run on real latents
and timesteps. Do not calibrate on held-out rows. `--teacher` optionally uses
a normal kohya-format Krea2 LoRA to express output drift relative to its
adapter contribution. This is not a training transfer test.
`--qat-steps` applies pilot's local weight reconstruction only to LittleBit.
Use small projection sets first; CPU decomposition of wide matrices can be slow.

## Staged validation and implementation

1. Capture real calibration and disjoint held-out activations/cotangents
   from representative early/middle/late blocks and attention/MLP projections.
   Preserve case/timestep provenance; compare native INT8 and raw BF16.
2. Compare equal packed bytes, not only nominal bits: LittleBit, LittleBit-2,
   SVD, Q4/Q6, INT8 and frozen residual corrections. Explore activation-weighted
   fitting, diffusion-output distillation and mixed-precision fallback.
   Select compression only where measured fidelity improves enough.
   Preserve consistent layouts across swapped blocks; otherwise disable swap
   in the test and investigate heterogeneous swap support.
3. Test full denoiser drift on held-out noise/timesteps before training.
   Verify forward/input-gradient parity, ordinary LoRA export/reload, cache
   identity, resume identity, gradient checkpointing and block swap/send-only.
   The external pilot's train_compact.py is process-local and has not been
   validated end-to-end on this WIP revision.
4. Matched ordinary LoRA training, fixed dataset split/seed/rank/alpha,
   optimizer/LR/scheduler, precision, validation and sample seeds; use native
   INT8 stock control and candidate from the same Raw model. Add raw BF16
   deployment evaluation separately when hardware permits. Evaluate both
   trained adapters on the SAME stock base at strength 1, not their training
   bases. Retention = (no-adapter loss - candidate loss) /
   (no-adapter loss - control loss). Nonpositive/near-zero gain is inconclusive;
   report raw losses, per-seed retention and uncertainty. Target >=96% learning
   gain retention, with matching fixed-seed likeness/style and preservation.
5. Record allocated/reserved GPU peaks, device memory, host RAM, disk bytes,
   decompression scratch, transfer time and steady-state step time. Start
   on 12 GiB, then test 8/6/4 GiB only if total memory permits. Avoid full
   dense persistent reconstruction; transient unpack cost also counts.
6. Add opt-in native trainer integration and cache provenance only after
   correctness gates pass and licensing is resolved. Never silently fall
   back to an incompatible training mode. Keep WIP/main unchanged.

Weight storage reduction is not total VRAM reduction. Activation, LoRA,
optimizer, attention, swap buffers, and runtime overhead still impose limits.
No GPU or full-model performance claim is made at this milestone.

## Milestone 2 — geometry and activation-aware adaptive screening

Ported the read-only matrix geometry helpers verbatim from
research/lora-matrix-geometry at e6232036a97a80504526c0bfe4f7c2260c9b890f
(no optimizer changes). Compact QR/SVD measures the effective rank of the
selected packed factor product without materializing it. For dense Raw
matrices, randomized SVD estimates captured energy and stable rank; these
are diagnostics, not a certified exact spectrum.

`adaptive.py` tests Q4/Q6 controls, LittleBit, LittleBit-2 and BF16 SVD at
multiple ranks. Each factor codec is tested with and without diagonal
activation weighting: fit W diag(s), then apply to x / s, where s is
the clipped calibration RMS per input channel. The BF16 scale vector
is included in the storage budget. This is an independent simple
activation-aware experiment, not a reproduction of Samsung's QAT.

Choose the smallest encoded candidate below INT8 bytes that passes BOTH
forward and input-gradient relative L2 errors against native INT8 on
calibration rows. The default 5% local tolerance is provisional; it is
not the 96% LoRA learning-gain target. Evaluate the chosen candidate once
on held-out rows; reject to INT8 on failure rather than using those rows
to tune rank. This is a conservative per-projection screening rule, not
a globally optimal memory allocation or proof of full-denoiser fidelity.
Data must be collected separately by case and timestep; hashes and exact
row overlap checks are not a substitute for real data provenance.

Run the reproducible synthetic CPU checks first:
```powershell
& $py scripts/research/krea2_littlebit/check_adaptive.py --pilot $pilot
```
The checks exercise all five pilot codecs on small synthetic matrices,
a structured rank-one target, a strict random-matrix fallback and a
deliberate calibration-only fit that must fail the held-out gate.
They use a dense reference, not the native INT8 path or real Krea2 data.

After collecting disjoint activation/cotangent archives, run:
```powershell
& $py scripts/research/krea2_littlebit/adaptive.py --pilot $pilot --raw $raw --calibration cal.safetensors --heldout held.safetensors --key blocks.0.attn.wk.weight --rank 128 --rank 256 --rank 512 --out adaptive-screen.json
```
Omit --key to screen all eligible projections (potentially very slow).
Archives must contain keys such as blocks.0.attn.wk.weight.x and
blocks.0.attn.wk.weight.dy with aligned 2D rows. No raw model or actual
activation captures were available in the CPU development environment.
The report is a proposal only: no compressed checkpoint is written,
no CUDA/trainer support is claimed, and its bytes exclude non-targeted
weights, archive metadata, runtime scratch and activation storage.
The next engineering milestone is a bounded stock-training capture
harness and mixed-codec archive/loader with cache identity and swap tests.

## Milestone 3: native web test queue

The Training tab now has an opt-in LittleBit Test Mode next to Add to queue,
adapted from test/global-replay-comparison. Select an ordinary Krea2 dataset
and the extracted unmodified external pilot folder, then Add to queue.
The original Raw BF16 model, VAE and encoder must be selected in Preferences.
The app preflights six cases as one queue operation. Stock INT8 trains first;
Q6, Q4, LittleBit rank 512, LittleBit-2 rank 512, and LittleBit-2 rank 1024
follow. All passing arms train five epochs with rank/alpha 32, Muon,
LR 0.00025, seed 42, 0.25 MP, batch 1, 22 classic swaps, four held-out images
and four validation timesteps. Selective replay, EMA, auto recovery, resume,
and previews are not used. Do not reorder, edit, or change the dataset.
The fixed recipe isolates the effect of frozen weights.

Before each compressed run, real Raw matrices are screened on two sets of
synthetic activation/cotangent rows against the native INT8 path (5% relative
forward AND input-gradient tolerance). Only entire corresponding projection
families that pass in all 28 blocks are archived, preserving the pilot's
uniform swap layout. A candidate with no passing family exits before training;
it is reported as screen_rejected and the queue advances. Raw and holdout
hashes must match the stock control. Nonpassing projections stay INT8.
This is deliberately NOT the adaptive per-matrix activation-weighted archive
yet; real activation capture and heterogeneous layouts remain research tasks.

The compact trainer is launched explicitly from the external pilot; the final
evaluation loads both LoRAs onto the same stock INT8 base and stock holdout
snapshot. No evaluation on a different base is substituted. Negative or
zero stock improvement makes retention inconclusive. Report, combined log,
and final diagnostic ZIP are under output_loras/littlebit_comparisons/<group>
(or your configured output root). Full compact archives are retained separately
and excluded from the ZIP. Each passing archive can be nearly as large as Raw
on disk because unselected weights are copied; allow sufficient disk space.
Do not mistake archive bytes or selected weight bytes for total GPU memory.
No Windows/CUDA end-to-end execution has been performed here. First test
is a feasibility and quantitative stock-transfer experiment, not an image
quality comparison.

## Milestone 4 — independent built-in experiment (supersedes pilot-folder queue instructions)

LittleBit Test Mode now uses native_codec.py and native_runner.py in this repository.
No external ZIP, download, checksum allowlist, or directory picker is required for
the web queue. The earlier separately licensed pilot is retained only as historical
evidence and for the old standalone screen/adaptive scripts. New candidates are
independently written, paper-inspired simplified variants: single balanced SVD path,
binary factors with per-latent magnitude estimates, and an optional eight-iteration
joint orthogonal Procrustes rotation. These are not exact LittleBit/LittleBit-2
reproductions; they omit the reference residual path, QAT and optimized kernels.
Q4/Q6 are independent grouped bit-plane codecs. Only packed projections are written
to disk; the normal source supplies the remaining weights via Fizgig's streaming
loader. Classic swap requires uniform layouts. The stock gate, fixed settings and
stock-base transfer report are preserved. Re-run all candidates from scratch; old
pilot metrics are not interchangeable with this experiment. Prior code was inspected,
so this is not a claim of a formal clean-room implementation or legal clearance.
No real model or Windows/CUDA runtime verification at this change.

Remove any old ZIP-based queue entries before creating a new comparison.
The version guard blocks outdated saved queues rather than mixing implementations.

## Milestone 5 — zero-swap-first residency queue (supersedes fixed 22-swap recipe)

Stock INT8 remains at 22 swaps. Every passing compressed arm tries 0 swaps
first; a fresh subprocess trains the entire five-epoch workload at each
attempt. Explicit CUDA OutOfMemoryError triggers the next count in
0, 4, 8, 12, 16, 20, 22, 24, 26. Other errors fail rather than retry.
This finds the first successful count on the tested grid, not the exact
minimum. An OOM late in training restarts with the same initial seed,
optimizer, dataset, and holdout; no partial adapter is promoted. Checkpointing
is explicitly on across the queue. Stock-base transfer evaluation still
uses 22 swaps on stock INT8. No pause, resume, live preview or reclaim changes
during the comparison. Stop terminates the process tree.

The per-case fit-report.json records attempted counts, exit classifications,
elapsed time and PyTorch allocator peaks (bytes) aggregated across epoch
resets. Successful native performance files are copied to the case root.
All failed attempts are kept in fit_attempts; only a completed adapter and
snapshot are published for stock transfer evaluation. Reports distinguish
storage for eligible block matrices from full-device memory, and record
actual selected storage savings from the projection gate. The user-supplied
512 MiB/block is not assumed for model accounting. CPU mock-worker tests cover
zero-swap success, retries, non-OOM failure, exhausted retries, isolation of
outputs and unchanged holdout across attempts. Real CUDA behavior remains
unverified. Remove old queue entries and start a fresh comparison.

## Milestone 6 — geometry and residual codec experiments

18 queue arms (one stock control plus 17 compressed candidates), version
independent-v3-geometry. Keep the original six arms as controls and add:
- Q4/Q6 with normalized 64-wide Hadamard, with fitted symmetric BF16 group
  scales, and with both. The fitted scale search retains the best actual
  quantized L2 objective per group; unchanged packed storage.
- HQQ group 8 with compressed scale/zero metadata, with and without a
  Hadamard transform on each 64 input coordinates. Reuses the actual
  upstream Fizgig solver/dequantizer (src/fizgig/minimax/hqq4.py, credit
  rintic-13, issue #102, upstream 85c2043bd6fdc6b0e8e34404bd8e15a04c4bdea6).
  The HQQ package is already pinned in requirements.txt; missing HQQ
  marks those arms blocked with the updater instruction.
- LittleBit rank 512 and LittleBit-2 rank 512 with a block-Hadamard latent
  basis (direct rotation for LB1; initial Procrustes rotation for LB2).
  No additional stored parameters for this initialization.
- LB2 rank 512 plus HQQ residual, with and without input Hadamard on the
  residual. Fit against the actual decoded binary factors, not the
  unquantized truncated SVD. Store both payloads and count every byte.
  Hybrid costs more than HQQ alone; no claim of equal-budget improvement
  across these methods. Paired rotated/unrotated variants have equal
  storage. Gate retains INT8 fallback for rejected projection families.

Hadamard is its own inverse. Only frozen-base activation computation is
rotated; backward maps input cotangents back and normal trainable LoRA
remains in the original coordinate system. Use tiled unpack/dequantize
and float32 accumulation across tiles for the new variants. No fused
CUDA kernel or speed promise. CPU geometry tests cover rotation, fitting,
packed length, forward/backward across a partial tile, and hybrid
composition with a mock HQQ backend; real HQQ solver integration tests
require installed hqq. Full Krea2 CUDA execution, transfer quality and
speed remain unverified. Gate still uses the same 5% forward/backward
relative error cutoff; stricter than an empirical LoRA gain criterion.
All arms start at zero swap except stock at 22; OOM retries unchanged.
Reports now include per-probe cosine and bytes per original weight.
New archive format; requeue rather than resume old comparisons.

## Milestone 7 — paired real-model development protocol (current queue)

This supersedes the 18-arm five-epoch recipe and its universal 5% gate.
See [PEER_REVIEW_V4.md](PEER_REVIEW_V4.md) for the current protocol,
exact verification status, limitations, and questions for independent
review. LittleBit Test Mode queues eight short cases, starts every
supported base resident, captures real training signals from the stock
control, and evaluates matched adapter snapshots on both native INT8
and original Raw BF16. Native NF4 is a separate non-swappable control.
There is no required learning-retention percentage. Confirmation scoring
and five-epoch promotion are intentionally separate future decisions.
Old queue entries must be removed before testing this version.

## Milestone 8 — V5 focused GPU smoke (default)

This supersedes the V4 eight-arm default after Grok and Astra Round 2
reviews. The former codec arms are retained (settings(root, full=True)),
but the UI defaults to three cases: INT8 stock, INT8 repeat and Q6 Hadamard.

- CUDA preflight before full training: actual reduced-width 28-block Krea2,
  distinct random packed weights, two rank-4 Muon updates, resident twice
  and classic 22 swaps. Compare model output, input gradient, one projection's
  output/input gradient in every block, factor gradients and effective BA.
  Report numerical differences per tensor, fail on swap differences beyond
  max(1e-5 relative L2, 3x resident repeat) and 1e-7 absolute L2.
  This is a numerical guard, not a learning threshold.
- Stock and repeat capture only valid tokens from the actual padded mask:
  up to eight image and eight text, redistribute unused slots; record
  exact indices and counts. This stratified sampling is not a population
  average. Do not select by cotangent magnitude. Undefined probe ratios
  are recorded; a projection needs at least one usable probe.
- Full stock tries 0,4,8,12,16,20,22,24,26 on OOM. Repeat and Q6 train at
  the stock's successful count (no automatic change in the paired quality
  arms). The repeat checks input/capture indices and records all captured
  tensor, named horizon export/gradient and loss differences.
- Score both exports and no adapter on INT8, original Raw BF16 and the
  actual compact archive (same fallback layout). Aggregate effective
  adapter action across all stock captures including the actual final export.
- Separate timing after quality: fresh native processes, 16 updates max,
  four warmup updates, same seed/dataset; no activation capture, trace
  hashes or gradient/step adapter saves. Lightweight synchronized compute
  interval (not end-to-end throughput). First time stock and Q6 at common
  swaps, then try Q6 at lower counts, reusing common if all lower OOM.
  Report per-step timings, item/timestep pairing, memory counters and
  fit attempts. Timing errors are not interpreted as quality failures.
- No fixed retention gate, no automatic five-epoch promotion. Actual full
  CUDA, Windows memory counter sampling and performance integration must
  still be validated. Report file: littlebit-results.zip.

Astra independently reported 38/38 LittleBit CPU tests on V4 (including
real HQQ), 64/65 supplementary tests (one unrelated UI assertion), and
bit-identical 448/448 captured tensors per update on a reduced 28-block
model, with synthetic random weights. The actual Krea2 attention key and
MLP up matrices showed lower Q6 Hadamard error at equal bytes. These are
reviewer-reported V4 results, not newly measured V5 results.

## Milestone 9 — offline activation-fitted private corrections

V5 stays unchanged; run this after its stock arm completes. Inspired by
Astra's cross-layer review: independent Q6 beat all tested shared pairs
as a whole, but block 0 attention key's private rank-76 BF16 + Q5 residual
improved 2.35% to 1.69% weight error at slightly fewer bytes. These were
reviewer measurements, not new results from this code.

From repository root in PowerShell (replace the raw path):

```powershell
.\venv\Scripts\python.exe scripts\research\krea2_littlebit\offline_corrections.py --raw "H:\path\to\krea2_raw_bf16.safetensors" --comparisons output_loras\littlebit_comparisons --out output_loras\littlebit-correction-report.json
```

Or pass `--stock <comparison>/stock` to choose a specific completed stock
arm. The input is the actual stock folder, not the report zip (which does
not include the capture tensors). The source SHA256 must match preparation.
Output must be fresh. Default 40 fit steps, four CPU threads. No new GPU
training, no shared-weight loader, no modified native swap or LoRA export.

Three predefined pairs: keys 0/1, keys 18/20, MLP up 0/1. Each pair's budget
is two actual Q6 Hadamard payloads plus a fixed 4 KiB descriptor allowance.
Compare independent Q5/Q6 Hadamard, private BF16 balanced SVD factors plus
Q5 Hadamard residual, shared BF16 input basis plus unique BF16 coefficients
and Q5 residual, and their activation-fitted counterparts. Half and full
budget-derived ranks are tested; raw projection error and exact payload
bytes are recorded. The fit uses raw-minus-decoded-residual forward and
input-gradient targets, normalized separately by raw action energy. The
Q5 residual is fixed during fitting. Factors are optimized in FP32 with
straight-through BF16 rounding and the best fit objective retained.
This is an offline local surrogate, not task-gradient or LoRA training.

Group existing captures by item ID; require three distinct IDs minimum.
Fit on all but two groups, choose candidates on one development group,
then read the last group for confirmation of Q6, the best private variant
and the development winner. No confirmation selection. These are
disjoint capture image IDs, NOT new unseen images relative to V5 LoRA
training, and not near-duplicate-independent. Very small sample sizes.
Report per-image/text token subset, timestep, raw BF16 and CPU native
INT8 operator-relative errors. Native INT8 is a CPU approximation to
the actual GPU training path. The teacher that generated X/G was INT8.
Report is JSON only; no deployment archives yet, no automatic promotion,
no arbitrary retention requirement. CPU memory and conversion time are
not estimated by the packed-byte budget.

New tests: `checks/test_littlebit_private_correction.py`. Full-size offline
run requires a completed capture set and the original checkpoint.

The report also includes per-active-layer packed payload bytes and dense
multiply-accumulate estimates (excluding unpack, Hadamard, BF16 rounding,
temporary memory and transfer). These are not measured GPU performance.
Capture files and raw checkpoint are hashed before and after the run.

Verification in the implementation environment: four isolated CPU checks
of the extracted rank-budget, split, and private/shared fitting functions
passed on synthetic inputs. The full new pytest file and actual packed-Q5
integration have not been executed in that environment: repository access
through the GitHub connector succeeded, but the local runtime could not
resolve raw.githubusercontent.com. No real capture or CUDA was available.
Run the integration tests from the full checkout:

```powershell
.\venv\Scripts\python.exe -m pytest -q checks/test_littlebit_private_correction.py
```

## Preparation reuse and progress (experimental)

Q6 Hadamard screening reuses decoded BF16 tiles across the four probes, retaining each probe's original GEMM shape, FP32 inter-tile accumulation and final BF16 cast. No training kernel or optimizer change. CPU threads default to min(8, logical CPUs); override with FIZGIG_LITTLEBIT_CPU_THREADS. Progress reports completed/total projections and a rough weight-size-weighted ETA, not a training completion estimate.

Complete Q6 archives with all 224 projections are cached under the comparison parent _compressed_cache (override FIZGIG_LITTLEBIT_CACHE_DIR). Cache keys include raw SHA256, codec, group, rank, format, encoder source hashes and PyTorch version. Reuse verifies the stored archive SHA256 as well as the manifest, shapes and payload lengths. The checksum is an accidental-corruption check, not a trusted publisher signature. A missing or invalid checksum fails closed; move the invalid cache and checksum aside before retrying. Run archives are hardlinked where possible, copied with a disk-space check otherwise. Partial layouts are not cached. Older archives without cache identity are not automatically adopted. Never edit cached/run archives in place. Fresh captures and family eligibility screening are still required. Stock projection outputs/cotangents are cached within each comparison by raw/capture/code identity and CPU thread count. No reference reuse across different capture sets. Interrupted encoding before full archive publication is not resumable. GPU encoding and fused training kernels remain separate validation work. No redistribution until upstream terms are verified.
