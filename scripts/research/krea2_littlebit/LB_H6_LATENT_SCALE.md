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


## Completed H6 results — 2026-11-27

User littlebit-latent-scale.json SHA256 13d7e3bde45cdaa199759005f977c41f8c31ff44929614744f5657565ce508af; completed 174.093s, torch2.15.0+cu134, key14 1536x6144, same raw hash f99bb0ff8e362b77342bc4994e0c50906fe7ef7074864b181b7d48d2fa6d03d7, fit64/2048 rows, dev8, confirmation_scored=false. Zero QAT steps. Both initial dev objectives and weight errors exactly reproduce H5 initialization. Each rank597+597, payload1179348 B, .9997456868 effective bits.

| Arm/stage | Dev objective | Dev forward RMS | Dev dX RMS | Pooled FP64 fit J | Weight RMS |
|---|---:|---:|---:|---:|---:|
| ordinary initial | .193984625 | .333137274 | .526297261 | .225719355 | .598044310 |
| ordinary BF16 solved | .181820087 | .292814440 | .527162097 | .216061826 | .603665081 |
| weighted initial | .184095456 | .332503964 | .507574651 | .186563092 | .613900247 |
| weighted BF16 solved | .170310003 | .284302127 | .509698251 | .174654651 | .622191460 |

Continuous optimum fit J ordinary .216060440044, weighted .174653434512. BF16 rounding adds only .000001386 and .000001217 to J, respectively. Both systems retain all1194 eigenmodes, normalized Gram condition 2.1288485 / 2.4142166, relative normal residual 6.16e-16 / 5.91e-16. No negative/zero gains or gain sign changes. Ordinary mean gain ratio 1.09506, weighted1.10470. Weighted dev objective improves 7.4882%, fit J improves 6.3831%, forward RMS improves 14.4966%; dX RMS increases .21236 percentage points (only 3/8 dev captures improve dX, same for ordinary). Weight error increases. Weighted solution still worse than H5 100-step QAT dev .140023256/dX .485581665. Pooled FP64 fit J is not the same metric as H5 equal-capture fixed-fit score. Numerical failure or BF16 gain rounding cannot explain the remaining gap for these fixed atoms; optimizing gains alone is insufficient. This does not establish intrinsic one-bit capacity or exclude benefits of gain solves after changes to signs or outer scales. No actual LoRA transfer/VRAM validation.

Timings: normal equations .594/.578s, eigensolve .078/.063s (combined1.313s); initializations2.953/3.016s; four evaluations13.859/14.062/14.250/14.219s; load fit/dev17.594s; metric .188s; continuous metrics .531/.547s. Total reported phases82.632s (remaining ~91.5s unattributed, includes validation/hashing and other runner overhead; cannot assign exact breakdown). End-to-end H6 ~11.1x shorter than H5 but different candidate counts/optimization, not equivalent-quality speedup.

Recommended H7 diagnostic: freeze signs, alternate conditional optimization of latent, row and column scales, with same original normalized .2/.4/.4 pooled objective and exact same payload. Each scale block is linear while other fields fixed; use matrix-free normal-operator CG (or a tractable exact row system), diagonal preconditioning, residual/iteration limits, compare against explicit tiny design matrices, guard BF16 rounded fit acceptance and gauge drift. Limit to weighted-50 plus initial/H6 controls, a few sweeps, record fit and dev after each, save packed factors for reuse and profile evaluation/validation. If fit barely improves despite well-converged conditional solves, change binary sign/atom construction rather than run additional gain sweeps. No further confirmation on old bank; fresh capture needed before promotion. Hybrid parked. This is a proposed next test, not implemented in this review.
