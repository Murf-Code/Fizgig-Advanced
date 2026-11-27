# Muon numerical isolation: CPU synthetic tests, 2026-11-27

Round 4 A/A discrepancy motivated two CPU experiments. PyTorch 2.10.0+cpu, four threads, no CUDA. Executed and then repeated against exact Git blob matches of the optimizer sources at 6405c85: muon_me.py 95b9df5eeee48f2bad6c9ed0288e0a98f84c7b88; muon_me_compressed.py 20775f3d83f2ab7e7ddda94aab0b5ac42c63a3ed; memory_budget.py 61b9e07475588ca2b37700a341399e03badadb63. Results identical at printed precision. No production code changed.

## Gradient perturbation
Shapes 6144x64 and 36864x64, seeds 42/43/44, synthetic G=UV/sqrt(k*m*n), k=1/4/16/64, independent isotropic perturbation rescaled to 1.2345% of gradient Frobenius norm (matching measured tproj.1 first-step A/A gradient error). Random fixed down factor A (64x6144, Gaussian normalized by sqrt(6144)). With L=chol(AA^T), effective displacement norms computed from B L; no dense BA. FP32 compact multiplication, approximate to floating roundoff. Production NS casts to BF16; comparison uses same polynomial in FP32. 144 measurements, six cases per rank/iteration/precision. Median effective-update relative error %:

| Input construction rank | BF16 NS1 | NS3 | NS5 | FP32 NS1 | NS3 | NS5 |
|---:|---:|---:|---:|---:|---:|---:|
|1|6.14|70.61|432.16|6.02|69.41|793.68|
|4|1.84|28.09|259.38|1.79|27.37|306.12|
|16|1.39|12.84|131.71|1.35|12.42|134.17|
|64|1.30|3.61|12.61|1.26|3.45|12.09|

BF16 NS5 median cosine .116/.287/.579/.992 and perturbed/reference update norm ratio 4.307/2.699/1.613/1.003, respectively. Low-rank errors therefore include norm inflation, not just rotation. Near zero singular values the polynomial slope is a=3.4445; repeated application has high small-mode gain. Normalization and nonlinear saturation mean a^5 is not a global Lipschitz bound. Finite-precision rank may exceed constructed algebraic rank. FP32 precision alone does not remove sensitivity. Synthetic perturbation differs from real A/A perturbations, and actual tproj gradient rank has not been measured. Do not attribute CUDA backward variation to NS (which executes afterward).

## Identical-gradient replay
Actual MuonME and CompressedMuonME, one B factor per optimizer, fixed A, initially B=0, LR .0002 match_rms_adamw, momentum .95, Nesterov, NS5, weight decay 0, 192 MiB, block64. Same two shapes, three seeds; independently generated rank1 or full rank64 gradients per step, copied identically into all five optimizers, 13 updates. 60 optimizer trajectories, 780 optimizer calls, 180 sampled records. FP32 control and repeat exactly equal; all compressed modes exactly equal to FP32 at step1 (empty momentum). All sampled factors finite. Median effective update/cumulative displacement relative errors (%):
| Regime | Format | Step2 update / cumulative | Step13 update / cumulative |
|---|---|---:|---:|
|rank1|BF16|43.746 / 30.305|52.814 / 34.997|
|rank1|INT8|64.710 / 44.998|136.084 / 105.501|
|rank1|Split-Max|59.297 / 41.266|120.270 / 93.029|
|full|BF16|.865 / .512|.903 / .382|
|full|INT8|1.148 / .677|1.520 / .895|
|full|Split-Max|1.082 / .640|1.318 / .784|

This is fixed exogenous gradients, not closed-loop fitting or Krea2 training, and only B moves; A fixed. The data isolate state quantization from gradient variation. BF16 conservative relative to INT8; Split-Max reduces synthetic errors vs INT8 but does not eliminate low-rank sensitivity. No inference of visual learning degradation or production settings. Rank1 gradient draws are independent, not necessarily realistic recurrent gradients. FP32 refers to momentum storage; production NS remains BF16.

Next: acquire actual first-step full factor gradients and spectra, replay same exact gradients with frozen initialization and compare NS, then repeated fixed-state CUDA backward to locate upstream variation. Consider damped singular-value response / fewer iterations only as research arms with matched update norm and quality validation; no default changes. Local probe and raw JSON retained as user artifact, not committed in this documentation-only change.

## Deferred follow-up: relative error scaling

User observed that normalized full-rank and rank-1 replay bar charts retain nearly the same shape. Rank1/full-rank cumulative error ratios are approximately 91.6x BF16, 117.9x INT8 and 118.7x Split-Max. Normalizing to INT8 within each regime gives BF16 42.7%/33.2% and Split-Max 87.6%/88.2%. This motivates a hypothesis of partly separable codec distortion and conditioning/NS amplification, not a demonstrated scaling law: only two replay regimes, three codecs, and aggregated medians, with separate plotting scales. A future controlled rank 1/2/4/8/16/32/64 sweep with measured spectra and matched perturbations could test it. User chooses to defer this investigation in favor of the next major memory architecture task. Do not change optimizer defaults or claim real-model numerical parity.
