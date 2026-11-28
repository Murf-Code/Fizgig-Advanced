# LB-H — maximize pure LittleBit

User scope decision: pure binary first. LB-G hybrid is preserved, parked, and not expanded. See LB_G_HYBRID.md and RESEARCH_STATUS.md.

## H1 convergence screen (a737af51, regression tests 27bcd0dd)

One-bit ceiling, same matched two-path E3 initialization, total rank ceiling 2048, original .2 weight/.4 forward/.4 input-gradient pooled objective, BF16 rounded scales and SmoothSign100. Use exact same old fit/dev/prior-confirmation capture partitions as F/G; all scoring panels are previously inspected. Fresh capture validation is a separate required gate, not implemented here. Suggested pilot key14 only (hardest), then key0; MLP is optional later because its total rank cap underfills one-bit budget. No hybrid, no new codec, no trainer changes.

| Arm | Sign learning rate | Sign warmup | Schedule |
|---|---|---|---|
| H0 | none | — | E3 initialization |
| H1 | 1e-4 | 0 | F2 constant |
| H2 | 1e-4 | 20 | F5 constant |
| H3 | 1e-4 | 20 | F5 through 100, cosine to 5e-6 at 500, floor to 1000 |
| H4 | 5e-5 | 20 | constant |
| H5 | 1e-4 | 20 | F5 through 100, freeze signs from 101, scales continue |
| H6 | 0 | — | scale-only |

All six optimization arms 1000 updates, fixed checkpoints 0/20/50/100/200/350/500/750/1000, scale LR .001 unchanged. No extra sweep of ranks or beta yet. Select checkpoint by prior development only, record prior confirmation for selected candidate. Save in-progress history at each checkpoint and completed candidate immediately; report is not a resumable checkpoint or deployable artifact. Record exact tiled weight error, sign gradient norm and current LR, cumulative flips, net flips, scale statistics, payload and decoded size, theoretical MAC count. Compare first 100 updates to LB-F/G; explicitly check long-horizon gains, oscillation, and weight/action tradeoff. Do not interpret reduction in local error as LoRA learning retention.

## Checks and limits

On PyTorch 2.10 CPU, reconstructed the exact rate/optimize excerpt (4062 characters, FNV-1a 1319198947) and imported F components excerpt (299 characters, FNV-1a 2913438331), matched against committed code; used copied qat_core mathematical subset. Synthetic 37x31 dual paths ranks 7+3, five token probes: constant/cosine prefix identical through 100, freeze preserved sign shadow tensors and cumulative flip counts while scales changed through 105, schedule midpoint/floor correct. These checks executed on isolated excerpts, not full repository pytest. Committed tests also compare directly to F5; that complete suite and CUDA runner remain unexecuted here. No new independent capture bank, actual VRAM or portable LoRA results.

## Next gate

Review H1 results, then narrow sign schedules and use fresh capture bank before promoting any winner. Investigate unequal binary ranks, activation-aware initialization, latent rotation, sign-specific optimizers only where evidence warrants. Preserve scope: maximize pure binary, validate complete block and actual LoRA transfer, then return to hybrid.


## H1 completed GPU results — 2026-11-27

User littlebit-convergence.json, completed, 232.188 s, CUDA torch 2.15.0+cu134, key14 only, seven candidates, raw/capture hashes match LB-G, no archive supplied. All candidates use rank597+597, 1179348 packed B, .99974569 effective bits; optimized decoded BF16 factors 18372948 B and 9169920 theoretical MACs/token (97.17% dense). F2/F5 100-step objectives reproduce LB-G exactly.

| Arm | Selected step | Dev | Prior confirmation | Dev at 1000 |
|---|---:|---:|---:|---:|
| H0 | 0 | .196482 | .210136 | — |
| H1 | 100 | .161681 | .178545 | .194367 |
| H2 | 100 | .161586 | .177727 | .184091 |
| H3 | 100 | .161586 | .177727 | .176659 |
| H4 | 200 | .159367 | .175588 | .171938 |
| H5 | 100 | .161586 | .177727 | .163012 |
| H6 | 100 | .183758 | .197204 | .207813 |

H4 best selected checkpoint improves F5 by 1.373% dev and 1.204% prior confirmation objective; prior confirmation RMS .419032, raw forward .219933 and dX .550278, versus E3 .330163/.557911. Weight error .608102 versus E3 .598044 (worse). No arm improves after 200 at measured checkpoints; H3 and H5 reduce late degradation but do not beat their shared step100. Element-count weighted cumulative flips (weights .4/.1/.4/.1 for A/B of equal rank paths) H1 .25060 at100 -> 8.46816 at1000; H3 .16896 -> 7.14044; H4 .43128 at200 -> 7.08405 at1000; H5 .16896 unchanged from100 to1000, confirms sign freeze. H1 net flips at1000 13.247%, far smaller than cumulative events. H6 fixed shadow signs but signed scale multiplier min -.133621 at1000, max 1.921392; effective factor signs can change through negative scales despite fixed binary codes. No near-zero fraction at logged checkpoints does not exclude crossing zero between checkpoints. H5 weight/action deterioration despite no additional shadow flips, and H6 late deterioration without any shadow flips, show oscillation is not the sole possible explanation. Fit minibatch losses use variable sampled rows and are not directly comparable; no deterministic full fit loss series in report, so overfitting is a hypothesis, not established.

Decision: retain H4 and F5 as pure-binary reference, do not repeat 1000-step sweeps on old captures. Next priority fresh diverse calibration with locked fit/dev/untouched confirmation and separate fixed-panel fit forward/dX diagnostics. Then targeted pure LittleBit objective/initialization study for key14 (backward fidelity is limiting): compare pooled versus per-capture normalization, balanced versus backward-emphasized objective, with weight-preserving control and short 100–200 step schedules; isolate each change, consider positive scale parameterization and unequal rank allocation afterward. Need independent review before major math changes. Do not treat H4 as universal winner or revive hybrid research. No trainer/code changes during result review, no deployment or portability/VRAM claim.


## H2 — objective-weight ablation (2026-11-27)

Runner binary_qat_objective.py (ca674382), tests checks/test_littlebit_objective.py (6cb71246). Pure binary E3 two-path initialization, one-bit ceiling, H4 schedule (.00005 sign LR, 20-step scale-only warmup, .001 scale LR, 200 updates). Same pooled-energy denominators, row sampling, SmoothSign100 and BF16 packed scoring. Fixed weights W/F/B: .2/.4/.4 control, .2/.2/.6 backward60, .2/.1/.7 backward70, .5/.25/.25 weight50, 0/0/1 backward-only diagnostic. H0 initialization reference. No other changes; positive scales and alternative normalization deferred. Fixed fit and development BF16 action error by capture, exact tiled weight error, sign/scale diagnostics at 0/20/50/100/150/200. Common balanced development objective selects checkpoint and arm; do not select by fit or confirmation, or compare weighted training losses across arms. Old panels are exploratory. Expect H2-control to reproduce H4 at 200 subject to numerical determinism; additional 150 checkpoint can select differently.

Optional --fresh-stock points to another completed V5 stock folder, with original raw SHA256, distinct captured item IDs (at least three), and unique capture hashes disjoint from the old bank. Generate using existing LittleBit Test Mode with a genuinely separate dataset (unique file stems, no reused/near-duplicate source images), let stock complete, then use its stock directory. Keep original --stock as fit/dev source; score the fresh bank on preselected checkpoints only after selecting the overall development winner; do not retune on fresh scores. IDs/hashes are necessary but not sufficient for independent data provenance. If omitted, fresh_capture_bank=null and results remain exploratory; no new calibration was generated here. No separate new stock-only queue or fresh-fit split implemented. First run key14 only, then narrow finalists. Fresh data remains a required gate before promotion.

Static source inspection completed; regression tests added for H4 control parity, finite gradients on all weights, and fresh-bank overlap/hash checks. Complete pytest and CUDA execution not performed in this environment. No production, hybrid, or previous runner changes. No new independent reviewer feedback, deployment or portable LoRA claim.
