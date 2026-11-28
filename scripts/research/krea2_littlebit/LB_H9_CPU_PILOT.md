# LB-H9 — CPU-first objective-aware orthogonal rotation

2026-11-27. Executed synthetic pilot on torch 2.10.0+cpu, four threads, 17.067515046 s, nine matrices and three arms. Report SHA256 d7e77c636a878efc40bc7d9aeefe610f79371a7c818ddcc27ac9e06565b7670f. No Krea2 tensors, training, production changes or independent real-model confirmation.

## Method

Reuse H8 synthetic generator, initialization and sign/scale refinement (128x512, ranks34+34, 8136 modeled bytes, .9931640625 bits, fit192/dev96, seeds1/2/3 for Gaussian/decaying spectrum/structured outliers). Baseline reproduces H8 final joint3 exactly. Compare ordinary ten polar iterations to the same initialization followed by three Givens pair sweeps per path, with fixed outer scales and unit latent gains. Pair order seeded independently of fit data and shared by weight and task arms. Candidate angles 0, +/-pi/4, +/-pi/6, +/-pi/12, +/-pi/24. The continuous product is invariant under A'=R^T A, B'=B R. A candidate changes two decoded rank-one terms; delta weight is U V, rank <=4, with U/V containing new and negative old terms. Calculate exact objective difference without materializing each candidate dense matrix: P=X V^T, Q=G U; E=decoded residual, F=X E^T, D=G E. For task weights from original teacher energies, delta J =
wa [2 <U^T E,V> + sum((U^TU).*(VV^T))] +
wf [2 <F U,P> + sum((U^TU).*(P^TP))] +
wb [2 <D V^T,Q> + sum((Q^TQ).*(VV^T))].
Weight-only arm sets wa=1/||W||², wf=wb=0; task arm uses .2/.4/.4 with original teacher energies. Select minimum strictly negative candidate (threshold -1e-13) on fit, no dev selection. Residual path refit to actual decoded first path; second path may have different continuous basis across arms. For each arm run one gain solve and three scale sweeps followed by three H8 sign+scale sweeps. Same number of refinement sweeps, not matched total compute. BF16-rounded scale coefficients, FP64 arithmetic, pooled dev J includes weight (not actual H3 BF16 per-capture metric). The search is a restricted discrete Givens grid, not a general learned rotation or global optimum.

## Results

Mean dev J, three seeds each:

| Family | Arm | Initial | Scale3 | Joint3 | Joint3 dev dX RMS | Joint3 fit J |
|---|---|---:|---:|---:|---:|---:|
| Gaussian | baseline | .605544796 | .593834309 | .494024680 | .614991863 | .396258336 |
| Gaussian | weight | .606146621 | .593440691 | .494807425 | .615183033 | .396833038 |
| Gaussian | task | .601316509 | .589038911 | .493924121 | .610828215 | .394711988 |
| Decay | baseline | .394293033 | .384636299 | .321575353 | .501527121 | .257427076 |
| Decay | weight | .388382764 | .379532860 | .321656054 | .502922988 | .257325225 |
| Decay | task | .386962874 | .378133089 | .320003198 | .503924475 | .255833056 |
| Outlier | baseline | .282429091 | .263914771 | .204980661 | .446071552 | .165518084 |
| Outlier | weight | .267224173 | .252255638 | .199533494 | .438494919 | .160539288 |
| Outlier | task | .263747321 | .251343701 | .201283457 | .444049057 | .160777829 |

Mean per-seed task dev reduction vs baseline initial: .6984/1.8393/6.6092%; after joint3: -.0218/.4789/1.8284%; task beats baseline initial 9/9, joint3 7/9. Weight-only final reductions -.1712/-.0383/2.7222%; wins 5/9 overall, 3/3 outlier. Task slightly lower mean dX in Gaussian/outlier, worse in decay. All candidate zero-angle, continuous-product preservation (accepted moves), incremental residual parity and accumulated objective delta checks passed. Accepted weight moves 407, task 479. All 81 H8 refinement sweeps passed inherited consistency assertions. Runtime includes all 27 arms and is synthetic only.

Interpretation: modest initial task-aware rotation advantage mostly disappears after direct sign and scale refinement; weight-only rotation best on structured outliers after refinement. No justification yet for adding costly full-size rotation search by default. H8 remains primary candidate; H10 progressive residual construction next, compare final quality at identical rank/storage and matched sign sweep budgets, record time. Keep rotation optional pending real-matrix evidence. Prototype and raw report delivered separately; no full-size runner yet.
