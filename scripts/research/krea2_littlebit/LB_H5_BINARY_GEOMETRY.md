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
