# Seven-rank Muon CPU sweep — 2026-11-27

User requested the deferred sweep before returning to activation architecture. CPU PyTorch 2.10.0+cpu, four threads; exact optimizer source hashes match 6405c85 (muon_me 95b9df5eeee48f2bad6c9ed0288e0a98f84c7b88, compressed 20775f3d83f2ab7e7ddda94aab0b5ac42c63a3ed, budget 61b9e07475588ca2b37700a341399e03badadb63). User archive muon-seven-rank-cpu-sweep.zip contains scripts, raw JSON, source copies and verification log. No production edits or CUDA test.

Sensitivity: 6144x64 and 36864x64, seeds 42/43/44 (derived seed seed*100+rank), synthetic Gaussian product gradients of construction rank 1/2/4/8/16/32/64, independent noise rescaled to 1.2345% Frobenius norm. Fixed A; effective displacement norm from B chol(AA^T), FP32 products, FP64 norms. 252 matched measurements. BF16 NS5/3/1, FP32 NS5, ridge polar via FP64 small Gram eigensolve, singular response sigma/sqrt(sigma^2+(tau||X||_F)^2), tau .02/.05. Median effective relative perturbation error (%):
|Rank|NS5|NS3|NS1|FP32 NS5|Ridge02|Ridge05|
|---:|---:|---:|---:|---:|---:|---:|
|1|432.68|69.88|6.12|790.12|60.79|24.40|
|2|304.77|34.96|2.77|379.55|42.79|17.20|
|4|260.05|28.24|1.83|304.52|29.77|11.99|
|8|172.91|17.17|1.53|181.74|20.41|8.28|
|16|130.80|12.74|1.39|133.13|13.39|5.53|
|32|83.94|8.04|1.33|82.72|7.87|3.48|
|64|11.82|3.63|1.30|11.31|3.23|2.16|

Replay: 6144x64, seven ranks x three seeds x 13 exogenous gradients, four storage modes, six strategies, 6552 synthetic B updates, 1512 records at steps 1/2/13. Fixed A, initial B zero; production codec and recurrence, .95 momentum, Nesterov, .0002 adjusted match_rms_adamw LR, block64, no decay. Identical first update within each strategy. Actual optimizer crosscheck rank1/64 seed42, 104 calls, exact alpha-add match; sweep multiply-add accumulation max discrepancy 5.49e-8 relative.

## Cumulative error after 13 updates, standard NS5
|Rank|BF16 %|INT8 %|Split-Max %|Split-Max/INT8 %|BF16/INT8 %|
|---:|---:|---:|---:|---:|---:|
|1|34.949|113.443|98.049|86.43|30.81|
|2|25.338|81.591|71.986|88.23|31.06|
|4|12.649|39.268|34.122|86.89|32.21|
|8|4.249|12.048|10.502|87.17|35.27|
|16|1.593|3.677|3.271|88.96|43.31|
|32|.524|1.193|1.052|88.13|43.94|
|64|.414|.907|.788|86.88|45.60|

Step13 median lookahead error BF16 .2966–.3133%, INT8 1.0643–1.1117%, Split-Max .9110–.9574%. Conditioning-sensitive amplification strongly supported. Split-Max/INT8 ratio stable, BF16 ratio changes: not an exact universal multiplier. Lookahead 90% energy ranks at step13 9/17/25/36/44/49/52; momentum mixture rank differs from construction rank.

## INT8 cumulative error vs each strategy's own FP32 reference
|Rank|NS5|NS3|NS1|FP32 NS5|Ridge02|Ridge05|
|---:|---:|---:|---:|---:|---:|---:|
|1|113.443|10.238|1.044|113.809|11.254|4.611|
|2|81.591|7.352|.903|81.401|7.422|3.156|
|4|39.268|3.842|.822|38.739|4.080|2.004|
|8|12.048|2.195|.792|11.457|2.062|1.375|
|16|3.677|1.410|.775|3.078|1.205|1.043|
|32|1.193|1.019|.767|1.035|.917|.883|
|64|.907|.850|.764|.817|.823|.814|

NS3 clean step13 cosine vs NS5 .948–.997, norm ratio .907–1.056 (per-rank medians); ridge05 cosine .928–.991, norm ratio .927–1.140. NS1 often much smaller norms (.467–.788). Lower compression error does not mean same optimizer or better training. No LR retuning, closed-loop fitting, useful weak-direction preservation, GPU timing, real gradient spectra or quality tests. NS3 simplest future candidate; ridge05 mathematical alternative. Park optimizer changes, continue bounded async activation offload and backward-memory investigation as next major milestone. GPU parity/quality/resume remain open.
