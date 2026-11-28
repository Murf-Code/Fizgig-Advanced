# LB-H4 — rank representation diagnosis (2026-11-27)

H3 key14 64-image control selected at100 achieved dev .143724 and confirmation .156816, with dX RMS49.237% dev and51.882% confirmation despite full numerical cotangent span. Diagnose rank loss vs binary constraints/initialization/optimization before more capture collection. Keep pure binary research; hybrid remains parked.

## Scope and protocol

rank_representation_screen.py (78e3a670), checks/test_littlebit_rank_reference.py (9c8ad37c). Key14 default, same raw BF16 identity, H3 locked 64 fit/8 dev. Hash all bank files and check manifests and partition, recheck identities at completion. Confirmation tensors are not loaded or scored (H3 key14 confirmation already consumed). Same dev captures, BF16 forward/input gradient operator scoring, equal-capture balanced objective. Compare with existing H3 report, no rerunning expensive binary arms. Native INT8 and direct Q2/Q3 odd Hadamard (existing mixed_factor_screen.direct_lowbit), Q4/Q6 Hadamard existing codec, all on same dev. Eight candidates total: five direct/native controls and three continuous diagnostics. Continuous total rank exactly matches H3 two binary paths, 1194 for key14 (597+597), not storage. BF16 continuous payload 18,339,840 B, 15.546875 effective bits, compared with binary 1,179,348 B, .9997456868 bits. No practical compression claim. Continuous dense factor compute is also not a runtime kernel benchmark.

## Mathematical references

Use row-convention W m x n, X t x n and cotangent G t x m; E=B A-W. Pooled objective J = .2||E||F^2/||W||F^2 + .4||X E^T||F^2/||X W^T||F^2 + .4||G E||F^2/||G W||F^2. All fit rows concatenated, full weight term instead of H3 random row estimate.

1. rank-svd: full economy CPU FP64 SVD, truncate to total rank r. Eckart–Young weight optimum, squared relative error = discarded singular energy / total singular energy. This supplies a lower bound on J from the weight term.
2. rank-backward-weighted: alpha=.2/||W||F^2, beta=.4/||GW||F^2. H=I+(beta/alpha)G^T G, SPD because alpha>0. H=L L^T via FP64 Cholesky. Let T_r be truncated SVD of L^T W. W_hat=L^{-T} T_r, realized as balanced factors in transformed coordinates followed by triangular solve on left factor. Because multiplication by invertible L^T preserves rank, this is the global rank-r minimizer of alpha||E||F^2+beta||GE||F^2 up to numerical precision. Its objective alpha times discarded squared singular energy is a lower bound on full J (nonnegative forward term). Not a global optimum of joint objective, not a lower bound on per-capture held-out RMS, and no binary constraints. Fit weighting is pooled and can emphasize high-energy captures.
3. rank-joint: choose initial SVD reference by lower FP32 full pooled fit J, then optimize continuous factors on all 64 fit captures for 100 Adam updates, RMS-normalized factor parameters with relative LR .003, global gradient clipping 1. Checkpoints0/10/20/50/100. Dev balanced BF16 score selects checkpoint. Local feasible continuous reference, not a certified joint optimum. All continuous references report FP32 pooled fit J and BF16 decoded factor weight error and dev errors. Mathematical bounds use FP64; BF16 rounding changes exact represented product. No change to existing binary training or codec. 

## Interpretation and next gate

If continuous rank references achieve much lower dX than H3 at the same total rank, the rank constraint alone cannot explain H3's error; investigate binary rotation/init and unequal path splits under unchanged packed budget. If continuous references remain poor, inspect rank/conditioning/metric or optimizer before blaming binary signs. The optimal weight reference alone does not prove an action optimum. Weighted-backward reference is exact for its narrower fit objective; joint Adam failure is not evidence of a hard capacity limit. Direct quant controls are not bit-matched. Do not select on consumed H3 confirmation; collect fresh confirmation before promoting future finalists. No full model or LoRA transfer quality measured. 

## Verification and command

Independent reconstructed FP64 toy calculation on torch2.10 CPU (19x31 rank7, anisotropic 9-row cotangents) gave ordinary weight+backward loss .1662321364, weighted loss .0776953928467963, analytic tail bound .07769539284679625; matched to numerical precision. Added committed regression tests for exact weight and weighted optimum, full rank reconstruction, zero-energy guard, joint objective/gradient parity. Full committed tests and CUDA runner not executed here. No runtime/production changes. 

```powershell
.\venv\Scripts\python.exe scripts\research\krea2_littlebit\rank_representation_screen.py `
  --raw "H:\Stable_Diffusion\Data\Packages\sd-webui-forge-neo - Krea2\models\Stable-diffusion\krea2_raw_bf16.safetensors" `
  --stock "output_loras\littlebit_comparisons\littlebit-20261127-162827-376f37\stock" `
  --key blocks.14.attn.wk.weight `
  --out output_loras\littlebit-rank-reference.json
```
