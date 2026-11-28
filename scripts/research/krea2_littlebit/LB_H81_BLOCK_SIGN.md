# LB-H8.1 — Blockwise sign refinement

H8/H9 real key14 showed whole-plane proposal interference (H8-B accepted 1948 flip events from 5,233,542 proposed events). H8.1 partitions each path/plane's latent directions into groups of 64 or 16 using deterministic permutations (seed 42,43 for two sweeps), ranks negative individual deltas, tries the entire group and halves the sorted prefix until the exact joint quadratic decreases. Updates the three residuals in place after each accepted group. B plane uses transposed weight residual and swaps forward/backward capture roles. All scales remain fixed during a sign sweep; each sweep is followed by the H7 row/column/latent refit. No SVD, rotation or QAT. Both candidates start from the same H7-control archive, hash/metadata/payload verified against completed H89 report, with raw/split and fit/development parity gates. Historical H8-B is copied from the verified-reference report, not rerun. Two final packed candidate archives with unchanged payload and JSON. Fixed two sweeps for this comparison.

For each oriented group let U be the fixed weighted left factor, V the proposed signed right-factor displacement, R=E,T=X E^T,Z=G E, H=k0 U^T R+k1(T U)^T X+k2(G U)^T Z. Joint delta = 2 <V,H> + <k0 U^T U+k2 (G U)^T(G U),VV^T> + k1 <U^T U,(X V^T)^T(X V^T)>. Update R+=UV,T+=(XV^T)U^T,Z+=(GU)V. Direct residual reconstruction and loss at end of sweep must match incremental/predicted objective within tolerance. No claim of global optimality.

CPU core actually tested in local torch2.10.0+cpu, 4 threads, two paths BF16 scales and signs, structured synthetic teacher plus Gaussian noise and approximately 10% corrupted initial signs. Small 7x13 rank5/128x512 rank34 tests passed across multiple group sizes, scale/sign validity and exact objective parity. Full key14 dimensions 1536x6144 rank597+597 fit2048 synthetic, objective .818141093205 -> .489704878889 (64,6.663548s,3021426 accepted events) or .476065235402 (16,8.154149s,2984123 accepted events). Max absolute residual discrepancy 4.55e-13/4.83e-13 respectively, predicted objective discrepancy <4e-16. Synthetic acceptance is not expected to transfer directly. Core Git blob e404b308792dc1c95627a75fcd036c11b3cc5e76 matches local tested file exactly; SHA256 eb69ec73a3c418ba902d5dd2cd25efe0fa21344f531d1abd059c7cb47c7d27b2. Runner commit 22aa3e1b9d71dfbb7ada005812257a8c82be8539 has not been executed against real archives here. Peak memory and total real runtime not measured.

Run block_sign_screen.py with same raw, stock, key as H89, --h89-report output_loras/littlebit-h89.json --out output_loras/littlebit-h81.json. The H7-control safetensors must be beside the H89 JSON under its original name. Upload new JSON; retain archives. Reused development only; no new confirmation, learning transfer or memory savings claims. H5 remains best binary benchmark. Hybrid parked.


## Completed real Krea2 H8.1 — 2026-11-27

Parsed complete user report littlebit-h81.json, SHA256 4730f879362488e2ee05ae1e6a9eb8f5bd42d949b63b0a819d818f4af54898e1. Protocol lb-h81-block-v1 completed 244.828s, torch2.15.0+cu134, 8 CPU threads. Key14 1536x6144, rank597+597, 1179348 bytes/.999745686849 bits. Fit64/2048 rows, reused dev8, confirmation8 unscored. Saved H7-control archive hash verified by runner; all four parity metric differences exactly zero. Historical H8-B reproduced from H89 report, not rerun.

| Arm/stage | Dev J | Dev fwd RMS | Dev dX RMS | Pooled fit J | Fit dX RMS | Weight RMS |
|---|---:|---:|---:|---:|---:|---:|
| H7 | .162078364650 | .261141226839 | .505926861260 | .164625209517 | .388394270742 | .622054899267 |
| H8.1-64 sweep1 | .162175968627 | .242416775416 | .515350409189 | .147566353710 | .327152487951 | .638955168783 |
| H8.1-64 sweep2 | .161057193138 | .232687283613 | .517659168103 | .139173924556 | .290503444845 | .649244060105 |
| H8.1-16 sweep1 | .161747905514 | .239260560028 | .515994375399 | .145921450162 | .321056576183 | .640624885603 |
| H8.1-16 sweep2 | .160119836696 | .228735409514 | .517609684827 | .138139337475 | .287129747394 | .650838000939 |

H8.1-16 is report winner but dev J improves only 1.2084% vs H7 despite 16.0886% pooled fit decrease. H8.1-64 improves .6300% dev/15.4601% fit. Both improve dev dX on only 1/8 captures vs H7; forward improves on all 8. H8.1-16 fit dX RMS drops 38.8394% ->28.7130% while dev dX rises 50.5927% ->51.7610%; weight RMS rises 62.2055% ->65.0838%. This is strong evidence of fit/development divergence consistent with overfitting, not proof that generalization is impossible. Historical H8-B dev .158739005691/fit .155912763797; H9-A dev .158287865520; H5 weighted50 QAT .140023255602 remains best binary dev. Fit and dev objectives use different weighting/aggregation; compare each to its own control, not raw gap as a directly matched metric.

| Arm | Sweep1 accepted/proposed | Sweep2 accepted/proposed | Total accepted | Acceptance |
|---|---:|---:|---:|---:|
| H8-B historical | 950/2680498 | 998/2553044 | 1948 | .0372% |
| H8.1-64 | 93743/2410652 | 88366/1768541 | 182109 | 4.3575% |
| H8.1-16 | 120236/2258249 | 95921/1548378 | 216157 | 5.6784% |

H8.1-16 ~110.96x more accepted flip events than H8-B (not unique flips). Sorted-prefix fallback and smaller groups both changed versus H8, so cannot isolate effect of group size alone across versions. All 16 substeps accepted, all 8 PCG solves converged (28–38 iterations), max sweep residual discrepancy 2.998e-15, predicted/direct J discrepancy <=2.776e-17. Measured sign sweeps 3.875/4.359s (64), 8.469/8.328s (16). No equal total-compute claim. Two archives recorded each 1179348 payload bytes, 1180172 file bytes; not independently opened here.

Decision: close H8.1 as a successful optimization engineering experiment but not a fidelity winner. Do not simply increase sweeps or further shrink groups. Consider a bounded regularization comparison (higher weight reconstruction penalty, e.g. .5/.25/.25, or fixed teacher-weight anchor) from shared H7, report common fit scoring and dev dX; alternatively prioritize pending H10 progressive primary/residual fitting with QAT benchmark. Lock candidate before independent fresh-bank confirmation; reused dev selection is exploratory only. No full-model, training transfer, actual VRAM or deployment evidence. Keep all archives; hybrid parked. No new runner in this results update.
