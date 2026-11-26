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
