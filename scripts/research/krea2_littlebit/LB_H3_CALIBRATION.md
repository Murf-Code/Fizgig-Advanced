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
