# LB-H6 — conditional latent-scale least squares

H5 weighted-50 marginally improves H3 dev .143724 -> .140023, dX49.237% ->48.558%. H5 intermediate measurements show major loss during binarization. Determine whether the latent gain subproblem is being solved poorly before changing binary signs or adding storage.

## Protocol

Regenerate H5 ordinary-50 and weighted-50 initializations only, same SVD sketches/seeds, metric, rotation and actual packed primary residual. Do not regenerate QAT checkpoints (H5 did not save them). Equal ranks 597+597, packed1179348 bytes, .9997456868 bits for key14. Use existing H3 bank fit64/dev8, hash and split validation, no confirmation tensors evaluated. All 2048 fit token rows; no iterative QAT. H5 selected post-QAT scores are historical benchmarks, not the starting checkpoint for H6.

Absorb fixed decoded row/column scales into B/A and concatenate both paths. Each atom D_i=b_i a_i^T, W_hat=B diag(c) A. c is the actual latent coefficient (not shadow multiplier). Let X be fit activations, G cotangents, Y=X W^T, Z=G W. P=X A^T, Q=G B. Weights alpha=.2/||W||_F^2, beta=.4/||Y||_F^2, gamma=.4/||Z||_F^2. For J(c)=alpha||W_hat-W||_F^2+beta||X W_hat^T-Y||_F^2+gamma||G W_hat-Z||_F^2:

H=alpha (AA^T .* B^TB)+beta (P^TP .* B^TB)+gamma (Q^TQ .* AA^T);
rhs=alpha diag(B^T W A^T)+beta diag(P^T Y B)+gamma diag(Q^T Z A^T).

Solve Hc=rhs by CPU FP64 eigendecomposition of the diagonally normalized Gram. Zero atoms retain original coefficient; solve displacement from initial gain vector, retaining its normalized-coordinate component in eigen-directions below 1e-12 relative cutoff. Report effective rank, condition, relative normal residual and actual direct fit J before/after. Exact continuous quadratic minimizer if no nonzero modes are truncated (up to numerical precision); truncated pseudoinverse otherwise. No explicit atom design matrix, no dense transform stored. Dense FP64 weight residual is used for independent direct diagnostic metrics only.

Round gains to BF16, evaluate actual decoded BF16 operator on dev via H3 score, report direct pooled FP64 objective of rounded coefficients and full relative weight error. Signs and outer scales unchanged, signed latent coefficients allowed just as earlier QAT; a negative gain may flip an effective rank-one contribution without changing packed sign planes. BF16 rounding is not guaranteed globally optimal on the discrete coefficient grid. No dev selection or confirmation, no training fidelity or GPU memory claims. Phase timings saved and console progress before each major phase. No saved factor archive; report includes original, solved and rounded gains for diagnosis.

## Verification

Core commit 13077523; runner eb7ba380; regression tests f3033360. Isolated CPU copy of core passed three tests: explicit design matrix Gram/RHS and direct least-squares solution parity; rank deficiency with duplicate and zero atom; BF16 rounding fixed fields. Synthetic pooled J fell 1.4525709328466392 -> .9752699205646025. Full repository test suite and target CUDA initialization integration not run here. Mathematical solve on CPU, existing CUDA initialization unchanged. Runtime not yet measured on target hardware. No production code changes.

## Run

```powershell
.\venv\Scripts\python.exe scripts\research\krea2_littlebit\latent_scale_screen.py `
  --raw "H:\Stable_Diffusion\Data\Packages\sd-webui-forge-neo - Krea2\models\Stable-diffusion\krea2_raw_bf16.safetensors" `
  --stock "output_loras\littlebit_comparisons\littlebit-20261127-162827-376f37\stock" `
  --key blocks.14.attn.wk.weight `
  --out output_loras\littlebit-latent-scale.json
```

If direct conditional optimum substantially improves fit/dev over initial and competes with H5 QAT, investigate alternating direct gain solves and changes to outer scale fitting. If it improves fit but not dev, investigate objective conditioning/generalization. If limited fit improvement, focus on sign/outer-scale geometry, not repeated latent gain LR tuning. No conclusion about intrinsic capacity until signs and outer-scale restrictions are challenged. Hybrid remains parked.
