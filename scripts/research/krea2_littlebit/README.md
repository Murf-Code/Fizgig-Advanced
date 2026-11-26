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
