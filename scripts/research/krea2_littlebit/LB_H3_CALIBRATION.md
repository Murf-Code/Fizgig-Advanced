# LB-H3 — calibration coverage (capture phase)

Scope: maximize pure LittleBit. Hybrid remains parked. H2 demonstrated a large fixed-fit versus prior-development backward error gap, not an established solution.

## Capture-only queue

On research/krea2-littlebit, update and restart Fizgig. Select Krea2 and original Raw BF16, valid VAE and encoder. Select a **new diverse dataset of at least 100 distinct source images**, with unique item names and valid captions. Do not use the prior small comparison dataset, duplicates, or near duplicates. Enable LittleBit Test Mode and Calibration only (LB-H3), then Add to queue. Only stock INT8 is queued. Original V5 mode is unchanged when calibration-only is unchecked. Do not run multiple GPU tasks simultaneously.

Reuses native trainer, fixed standard rank32 LoRA/Muon, batch/accumulation 1, 0.25 MP, original V5 seed, checkpointing, classic swap and OOM retries. No new codec or runtime quantization. The adapter evolves during capture (ordinary actual stock training signals, not frozen-adapter probes), and final adapter is saved by existing worker for completion checks only. No repeat, Q6, projection search, transfer evaluation or CUDA-equivalence preflight in this mode; raw BF16 identity verification and normal holdout snapshot still required. One epoch maximum, 200-update safety limit, stop as soon as 80 distinct training image IDs are captured. If fewer than 80, fail explicitly. No intermediate LoRA/gradient snapshots. Failed attempts isolated.

Only blocks.0.attn.wk.weight, blocks.14.attn.wk.weight, blocks.0.mlp.up.weight captured; at most 32 valid stratified image/text token rows per update. Layout, exact token indices, step, timestep, item identifiers, input fingerprints, capture SHA256, raw SHA256 (prepare.json) recorded. Distinct IDs and hashes alone cannot establish source-image independence; user must ensure provenance. Timesteps sampled by ordinary training distribution, not forced uniform, assess spread before fitting.

Locked SHA256 order LB-H3-v1 partitions 80 unique capture IDs into 64 fit / 8 development / 8 confirmation. Fit sizes 2/8/16/32/64 use nested prefixes of the same 64. Confirmation is not for selection or further tuning. All three splits exclude trainer validation holdout; they are held out from offline compression fitting, not from stock LoRA updates. Additional independent dataset remains required before promotion.

Output under output_loras/littlebit_comparisons/<group>/stock (or configured output root). Send capture-bank.json and prepare.json or littlebit-results.zip. Keep calibration-*.safetensors locally; the ZIP deliberately does not include tensors. Do not delete successful stock directory before offline H3. Coverage fitting/rank-overlap runner is a separate next implementation step, not included in capture phase.

## Verification

Observer 36128b98; native worker 5b09ab67; sweep ae6e023c; queue f9d4752e; API bfa994af; UI e4b8103d; regression tests 5e756684. Inspected sources and verified isolated reconstructed partition logic (stable reverse order, 64/8/8 sizes, no overlap, rejects duplicate ID). Full committed tests and actual UI/trainer CUDA integration not run here. No WIP or production code changes. Next review capture completeness, timestep spread, fit-span coverage and data provenance; then run same-bit matched calibration-size ablation. No claims of LoRA quality or VRAM savings.


## H3 capture phase completed — 2026-11-27

User report littlebit-report(2).json SHA256 a2180021a3ff4f9dcfba5b4e8961fb956bfc848342ebf8c4cd219c833683f464, littlebit-results(1).zip and comparison log. Stock folder output_loras/littlebit_comparisons/littlebit-20261127-162827-376f37/stock. Raw SHA256 f99bb0ff8e362b77342bc4994e0c50906fe7ef7074864b181b7d48d2fa6d03d7. 100-image new dataset as reported, 8 trainer holdouts, 92 trainable, 80 completed updates and 80 unique captured IDs and recorded SHA256 values. Three intended projections; all 80 records 16 image +16 text rows; token index/count consistency checked, trace and capture-bank copies identical; deterministic hash partition recomputed and matches 64/8/8; ZIP CRC and embedded report consistency passed. 2560 paired rows per projection, fit prefixes 64/256/512/1024/2048 rows. Actual tensors not included in ZIP, numerical and file-hash verification pending local offline runner. Source image uniqueness/diversity beyond IDs not independently verified.

Timestep range .119578–.930930, mean .607109; five equal-width bins [0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1]: all counts 4/7/26/29/14; fit 4/6/19/24/11; dev 0/0/3/3/2; confirmation 0/1/4/2/1. First fit2 prefix t=.786406/.824103; fit8 includes .151403–.926383; caption valid token counts 54–303. Preserve predeclared split despite imbalance; fit size and timestep coverage covary, especially fit2, and low-timestep evaluation is limited. No selection or score on confirmation tensors yet. Fit/dev/conf from evolving stock adapter trajectory; not independent trained models.

RTX4070Ti torch 2.15.0+cu134: 0/4/8/12 swaps OOM, 16 succeeded. Successful worker 664.907s (attempt 668.625s); total attempt wall times 766.189s excluding prep/cache. Peak allocated 9.589 GiB, reserved 10.299 GiB (allocator cap), sampled WDDM dedicated 10.455 GiB, shared 1.830 GiB. No claim of all-resident or compression savings. OOM retries worked as designed. For subsequent capture runs, starting near previously successful 16 swaps is an optional efficiency change, not needed for this bank.

Next: offline H3 coverage screen, key14 first; validate files, shapes, finite tensors, metadata/raw hashes; compute fit activation/cotangent rank/energy and development span coverage before fitting; do not expose confirmation diagnostics until selection frozen. Use fixed E3 initialization, 1-bit budget and matched H4 (.2/.4/.4) and Weight50 (.5/.25/.25) schedules for nested 2/8/16/32/64 fit sizes, same update count and seed, bounded memory; objective fit aggregation and batching must be documented and matched across sizes. Compare held-out backward/forward and weight fidelity, then confirm locked comparisons. No need to regenerate bank. No runtime code changes in this review. Hybrid parked.


## H3 offline coverage runner implemented

binary_qat_coverage.py (1a050428, CLI import fix 5a71811e). Default key14, one projection per invocation. Uses completed stock bank, verifies prepare/raw hash, matching trace and capture-bank manifests, capture hashes and deterministic split; rechecks source/capture hashes at completion. Actual tensors checked for shape, floating dtype and finite values when read. No confirmation tensors decoded or scored until all per-arm checkpoints and overall development winner are locked; hashes may be verified earlier. Report all eleven prespecified comparisons on confirmation, never reselect on confirmation.

One E3 baseline + ten binary arms: nested 2/8/16/32/64 capture prefixes, each H4 control (.2/.4/.4) and Weight50 (.5/.25/.25). Same E3 initialization and one-bit cap, same H2 optimizer, seed42, sign LR .00005, scale .001, warmup20, 200 steps, checkpoints 0/20/50/100/150/200. All fit rows concatenated into one FP32 action objective per update; pooled energies exactly equivalent algebraically to H2 per-capture sums but finite precision reduction order differs. No sampling of fit rows inside updates, only weight-row sampling from H2. Concatenation avoids 64 separate autograd graphs, actual memory and runtime not measured. BF16 packed CPU per-capture evaluation with raw and stock INT8 reference, equal per-capture balanced RMS score selects checkpoints, plus fixed-fit and weight error diagnostics.

Before fitting, calculate FP64 CPU row Gram eigensystems separately for x and dy per nested fit prefix; report singular spectra, numerical rank (relative singular threshold 1e-4), entropy/participation effective ranks, 95/99% energy ranks and fraction of development signal energy projected into numerical fit row span. These are uncentered energy-weighted directions, not intrinsic semantic dimensionality. No dense feature covariance or dense task gradients; row Gram can reach 2048x2048. Fit counts and timestep coverage remain confounded; preserve split. No confirmation span diagnostic used for selection. All selected factors held on CPU until final scoring; report saves after each checkpoint, not a resumable model archive.

Isolated reconstructed geometry and pooled-fit code executed on torch 2.10 CPU: synthetic rank-deficient span (rank2 and rank1, both 50% projected energy), concatenated teacher outputs, pooled objective and dense toy gradient parity within 1e-6 passed. Added checks/test_littlebit_coverage.py. Full repo tests and real CUDA runner not executed here. No production, hybrid or prior runner changes. Independent model-wide LoRA validation still pending.

Command:
```powershell
.\venv\Scripts\python.exe scripts\research\krea2_littlebit\binary_qat_coverage.py `
  --raw "H:\Stable_Diffusion\Data\Packages\sd-webui-forge-neo - Krea2\models\Stable-diffusion\krea2_raw_bf16.safetensors" `
  --stock "output_loras\littlebit_comparisons\littlebit-20261127-162827-376f37\stock" `
  --key blocks.14.attn.wk.weight `
  --out output_loras\littlebit-coverage.json
```
