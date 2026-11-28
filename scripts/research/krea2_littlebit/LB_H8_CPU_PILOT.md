# LB-H8 — CPU-first direct binary sign pilot

2026-11-27. Executed on CPU, torch2.10.0+cpu, 4 threads, 6.65619932 s for nine synthetic trials. Report SHA256 a4afb2148aedc8b2a052f0a3e971f0f0e04eea04d03c59279444ca8425fc48c0. Synthetic pilot only; no Krea2 raw weights or captured actions used. No production changes.

## Method

W_hat = sum_p diag(row_p) B_p diag(c_p) A_p diag(col_p), signs +/-1, BF16-rounded scales represented in FP64 for objective calculation. E=W_hat-W, F=X E^T, D=G E. Original teacher normalized coefficients wa=.2/||W||², wf=.4/||XW^T||², wb=.4/||GW||². Fit J=wa||E||²+wf||F||²+wb||D||².

For a single effective atom b a^T (outer scales and gain absorbed), flipping an A sign at coordinate j changes a_j by d_j=-2a_j. Define h=wa b^T E+wf (F b)^T X+wb (G b)^T D. Exact single flip delta is 2 d_j h_j+d_j²(wa||b||²+wf||b||²||X_:j||²+wb||Gb||²). Propose all individually improving flips within one atom row. Because forward terms couple coordinates, verify the full joint delta 2 d^T h+(wa||b||²+wb||Gb||²)||d||²+wf||b||²||Xd||². If it is not negative, use the best improving singleton. Update E,F,D incrementally. For B column, swap the analogous roles: h=wa E a+wf F^T(Xa)+wb G^T(Da), diag=wa||a||²+wf||Xa||²+wb||a||²||G_:i||². Joint delta 2 d^T h+(wa||a||²+wf||Xa||²)||d||²+wb||a||²||Gd||². Process A then B for each atom, both paths share residual. No STE or sign learning rate. Threshold -1e-15. Exact accepted moves in real arithmetic for fixed scales, not global discrete optimum.

Scale updates: joint latent linear solve in FP64 using H6 Gram; row and col block matrix-free FP64 diagonal PCG, 100 iterations max, relative preconditioned residual 1e-8, guard BF16-rounded fit objective. No explicit atom design matrix; dense weight/forward/backward residuals are used in this prototype. No memory/throughput claims for actual Krea2.

## Protocol

128x512, two rank34 paths, 8136 modeled packed bytes (.9931640625 bits). Three seeds 1/2/3 each Gaussian, exp(-i/40) singular decay, and decay with lognormal row/column outliers (.6 log std). Synthetic X/G independent fit192/dev96 rows, shared coordinate lognormal anisotropy .7. Truncated SVD, ten polar rotations with random QR init, mean-absolute outer scales, latent starts1, residual actual BF16-rounded decoded reconstruction. This is not exact H5 initialization and differs from earlier H7 pilot seed/initialization details. Recompute matched gain-only and scale-only controls within this experiment. One latent solve followed by three row/col/latent sweeps (shared starting checkpoint); compare three additional scale-only sweeps, three direct sign sweeps with scales frozen, and three sign sweeps each followed by one row/col/latent sweep. Dev pooled J includes weight .2/.4/.4, unlike H3 action-only equal-capture dev metric. No selection on dev. BF16 coefficient rounding only, not BF16 staged matmul scoring. No actual checkpoint archive or deployment.

## Results

Mean dev J and backward RMS, three seeds:

| Family | Initial J | Gain J | Shared scale3 J | Scale-only control3 J | Signs3 J | Joint3 J | Control dX | Joint dX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gaussian | .605545 | .603451 | .593834 | .593785 | .518114 | .494025 | .757582 | .614992 |
| Decay | .394293 | .391681 | .384636 | .384621 | .337782 | .321575 | .613846 | .501527 |
| Outlier | .282429 | .277064 | .263915 | .263863 | .217441 | .204981 | .536525 | .446072 |

Mean per-seed reductions vs matched control3: signs-only 12.7584%/12.1560%/17.6803%; joint 16.8236%/16.3591%/22.4121%. Both signs-only and joint beat control in every trial. Joint dev J trajectories from shared checkpoint: Gaussian .593834/.535171/.507326/.494025; decay .384636/.350836/.331730/.321575; outlier .263915/.227026/.211584/.204981. Matched control fit J .582460/.374982/.258479; joint fit .396258/.257427/.165518. Fit improves more than dev, evidence of synthetic generalization gap. All 54 sign sweeps passed monotone fit and incremental residual parity checks (rtol/atol1e-10); max predicted vs recomputed objective change discrepancy 9.02e-17. Group proposals triggered fallback 9 times total. This is exact arithmetic move evaluation, not proof all beneficial moves found. No comparison to tuned QAT yet. Synthetic only; actual H3 bank not available here, no independent confirmation or LoRA transfer or VRAM evidence.

Next: H9 objective-aware orthogonal rotation on synthetic matrices with same sign/scale refinement budget and held-out controls; H10 staged residual refit. H8 is a promising candidate for real Krea2 gate, not yet a production algorithm. Full-size scaling of per-atom residual updates may be costly; profile before asking user to run it. Keep hybrid parked.
