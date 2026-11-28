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


## Completed H4 GPU report — 2026-11-27

User littlebit-rank-reference.json SHA256 e28eba88fe15134bfaf5ef78af40e94a8af571a274423d51aeee8897107de296; completed 212.922 s, torch 2.15.0+cu134, CUDA joint optimization, key14 1536x6144, total rank1194, binary reference rank597+597. Same H3 raw hash and development step IDs [8,22,33,5,17,55,1,28]; confirmation_scored=false. Eight candidates.

| Candidate | Effective bits | Relative weight error | Dev forward RMS | Dev dX RMS | Dev objective |
|---|---:|---:|---:|---:|---:|
| Q2 odd Had | 2.25 | .346405 | .133683 | .319930 | .060113020 |
| Q3 odd Had | 3.25 | .192979 | .068013 | .177567 | .018077896 |
| Q4 Had | 4.25 | .106891 | .036395 | .098653 | .005528511 |
| Q6 Had | 6.25 | .024149 | .008610 | .022454 | .000289158 |
| stock INT8 | 8.005208 | not measured | .018000 | .014105 | .000261474 |
| continuous SVD | 15.546875 | .178350 | .060381 | .125227 | .009663882 |
| backward-weighted SVD | 15.546875 | .182103 | .061963 | .102131 | .007135082 |
| joint continuous | 15.546875 | .183740 | .045023 | .103398 | .006359152 |

Exact FP64 optimal weight relative error .178334980; joint fit lower bounds .006360673 (weight only), .006917052406 (weight+backward). Pooled FP32 joint fit objective for SVD .013383109, backward-weighted .008433890, joint at100 .007680915 (11.043% above the stronger lower bound; comparison subject to FP32 numerical error). Joint initialized from weighted SVD by fit objective, dev checkpoints0/10/20/50/100 .007135091/.006505431/.006414741/.006366356/.006359152; small gain after50. Weight-backward SVD has best dX among continuous references; joint improves balanced objective and forward at slight dX/weight cost.

Against H3 64-control dev .143724/dX .49237 and weight error .611684, continuous rank-joint has 95.575% lower dev objective; weighted SVD has 79.257% lower dX RMS. These are feasible rank-matched high-precision references, NOT bit-matched and NOT attainable binary guarantees. Rank alone demonstrably cannot explain binary error; cannot yet separate binary representation capacity, initialization/rotation/scales and optimizer failure. The fit lower bound is not a held-out bound, and joint is not certified optimal. Direct Q2 still substantially better than current binary at 2.25x effective bits; no deployable 1-bit training result. No further confirmation was scored.

Next **H5 pure binary initialization and rank allocation**: screen equal 50/50 control, primary/residual 75/25 and 25/75 (optionally one path with exact byte accounting), ordinary vs backward-weighted continuous initialization in original weight coordinates, residual against actual BF16-rounded decoded first path. Keep cap1 bit, predeclare variants; verify actual packed bytes and compare total ranks because overhead differs. Weighted initialization must undo Cholesky transform before binarizing (no stored dense transform). Measure continuous pre-binarization, immediate binary, and after identical short scale warmup/QAT to isolate loss sources. Prioritize cheap initialization screen and only a small number of matched QAT finalists (100 steps on H3 64 fit, dev8), using H3 control as reference and no further confirmation scoring on old bank. Fresh independent confirmation required for future promotion. No code changes in this review; hybrid remains parked.
