# LB-H5 — pure binary geometry

H4 established substantially lower error at rank1194 in unrestricted continuous factors. H5 tests whether binary initialization and rank allocation recover part of this gap. No hybrid or deployment changes.

Six initial candidates: ordinary versus backward-weighted SVD, each with 597+597, 896+298, 298+896 primary/residual ranks for key14. Exact same total rank1194, payload1179348 B (0.9997456868 effective bits), including two sets of BF16 scale headers and sign packing. The exact equal-rank allocation is used to define total rank, rather than increasing rank in asymmetric arms. Other shapes fail explicitly if the requested mirrored splits cannot fit.

Ordinary primary uses the existing H3 seeded SVD sketch; residual uses seed43 and actual packed-decoded primary, reconstructed in FP32 tiles from BF16 scales and sign codes. Weighted variant uses fixed H=I+2||W||^2/||GW||^2 G^T G computed FP64 on all 64 fit captures (same original teacher normalization for both paths); H=LL^T. SVD sketch of L^T target in FP32, inverse triangular solve in FP64, then FP32 factors with latent norm balancing (product preserving), existing q.rotate and Path magnitude initialization. Metric not stored. Weighted initialization is approximate because the SVD sketch and finite precision are not the exact H4 optimum. Latent norm balancing is part of the weighted initialization bundle, not an isolated ablation. All candidates report pre-binarization continuous primary, packed binary primary, packed primary plus continuous residual, final binary initial. The continuous intermediate is a diagnostic, not a one-bit representation; BF16 factor execution/weight error, not an exact pre-rounding oracle.

Choose ordinary-50 control and two lowest initial development objective noncontrol candidates (name tie break). Release other states; only one GPU arm resident at a time. Run identical H3 control optimizer for100 updates, sign LR .00005, scale .001, 20 scale-only warmup, .2/.4/.4 objective, all64 fit captures, checkpoint0/20/50/100; fixed fit, dev and weight error, sign flip/scale stats. Select checkpoints by dev; report winner among finalists. Initial screening can miss a slower-starting candidate; the two best initial alternatives are not necessarily best after QAT. No repeat of H4 direct controls, no confirmation scoring or new capture collection. Same H3 reused dev8, old confirmation consumed, fresh independent bank required before promotion. No portable LoRA quality, actual VRAM or speed claim.

Runner commits 613456c6 and 505ce75a. Independent reconstructed CPU budget and inverse transform/norm balancing checks passed for key14 and 19x31 synthetic matrix. Added committed regression checks for budget, transform, zero-energy guard and decoded residual. Full repo suite and CUDA execution not performed here. Prior runner and production/WIP runtime unchanged.

```powershell
.\venv\Scripts\python.exe scripts\research\krea2_littlebit\binary_geometry_screen.py `
  --raw "H:\Stable_Diffusion\Data\Packages\sd-webui-forge-neo - Krea2\models\Stable-diffusion\krea2_raw_bf16.safetensors" `
  --stock "output_loras\littlebit_comparisons\littlebit-20261127-162827-376f37\stock" `
  --key blocks.14.attn.wk.weight `
  --out output_loras\littlebit-binary-geometry.json
```


## Completed H5 results — 2026-11-27

User littlebit-binary-geometry.json SHA256 964422da2e3681a857324c56714d21a7efd4108ab90ee8ee0b1de0daa12e6ac1; completed 1936.141s (32m16s), CUDA torch2.15.0+cu134, same raw SHA and locked H3 fit64/dev8, confirmation_scored=false. All six initial candidates use rank1194 and 1179348 B (.9997456868 bits). No per-phase timings or measured VRAM; no saved factor archive. 

| Arm | Continuous primary dev | Binary primary dev | Binary primary + continuous residual dev | Fully binary initial dev | Initial dX RMS |
|---|---:|---:|---:|---:|---:|
| ordinary-50 | .077213 | .388975 | .070098 | .193985 | .526297 |
| ordinary-75 | .031405 | .348275 | .105620 | .200379 | .537688 |
| ordinary-25 | .173861 | .469320 | .036146 | .202119 | .540725 |
| weighted-50 | .061444 | .381504 | .058226 | .184095 | .507575 |
| weighted-75 | .024880 | .338602 | .088387 | .186023 | .509658 |
| weighted-25 | .149516 | .457233 | .027499 | .191168 | .522565 |

Weighted initialization improves all three paired initial joint objectives and dX, but increases weight error. Selected QAT finalists ordinary-50, weighted-50, weighted-75 per protocol.

| Arm | Selected step | Dev objective | Dev forward | Dev dX | Fixed fit objective | Fit dX | Weight error |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary-50 |100|.143724328|.212171|.492374|.094060145|.380321|.611684|
| weighted-50 |100|.140023256|.210373|.485582|.07624459|.331816|.621485|
| weighted-75 |100|.14016702|.209613|.486206|.07530279|.329532|.619416|

Control reproduces H3 at step100 exactly in reported objective and dX. Weighted-50 winner gives 2.575% lower dev objective, 1.380% lower dev dX RMS (6/8 captures improved), 18.941% lower fixed-fit objective and 12.75% lower fit dX than control. Weighted-75 effectively tied (slightly better forward and fit, worse dev dX/joint). Fit/dev dX gap grows from .112053 control to .153766 weighted-50. Weight error grows in each optimized arm. No evidence unequal allocation improves over equal ranks under this screen; unadvanced arms might learn differently. Weighted-25 binary+continuous residual dX .203172 becomes .522565 after residual binarization; objective .027499 -> .191168 (6.95x). This intermediate is not bit-matched and not a deployment candidate. Binarization/scale initialization clearly destroys useful reconstruction in tested recipe; cannot infer intrinsic binary capacity limit. H4 rank oracle and H5 intermediates support changing binary fitting method rather than repeated rank/step sweeps. Same reused dev, no independent confirmation or actual LoRA transfer.

Recommended next H6: inexpensive conditional latent-gain solve. For fixed packed sign matrices and fixed row/col scales absorb row/col into B and A, concatenate paths; W_hat=B diag(c) A. Full pooled J is a convex quadratic in c. Compute exact rank-atom Gram H from Hadamard products of B^T B and AA^T (weight), B^T B and (XA^T)^T(XA^T) (forward), (GB)^T(GB) and AA^T (backward), each with original H3 normalized objective weights. Right-hand side is inner products of rank-one atom actions with corresponding teacher targets. Solve in FP64 with pseudoinverse or explicit regularization as labeled, then round gains to BF16 and evaluate packed operator. Preserve storage/signs, measure pre/post fit and dev; not global optimum for signs/row/col, and BF16 rounding may worsen conditional optimum. Start from regenerated equal ordinary/weighted init (H5 does not save states); avoid repeating long QAT just to recreate a checkpoint unless separately justified. Compare against historical H5, then decide if broader scale geometry or alternating discrete optimization needed. Profile phase times and batch evaluation before claiming speedup; verify parity with original BF16 per-capture evaluation. Hybrid parked; no production changes.
