# LB-F — local QAT stabilization

## Evidence from LB-E

User report `littlebit-binary-qat.json`, protocol binary-qat-v1, CUDA torch 2.15.0+cu134, 563.469 seconds, 62 candidates, three projections, 100 updates/arm. Fit captures steps 3/4; previously inspected scoring captures steps 2/1. E4 improves both scoring objectives vs E2 in 10/10 pairs, but weight error in only 4/10. E5 worsens all three metrics vs E3 in 10/10. E6 improves development in 6/10 and both scoring panels in 3/10 vs E3; weight error worsens in all 10. E6 vs E5 improves both scoring panels in 10/10. Key0 1-bit SmoothSign E6: dev .03389395 vs E3 .06185954; prior confirmation .05429558 vs .07902648; weight relative error .34843773 vs .31867779. SVD8 at same ceiling: .02457421/.09192431, rank153 vs 1194 total binary rank. MLP up0 .5-bit SVD8 .02739644/.00591595, E3 .09502661/.086651 (rounded); binary rank cap skipped distinct MLP 1-bit SVD8 in E, already screened in D3. All objectives are local mean squared relative forward/backward errors, not learning retention.

## LB-F protocol

Run one real projection first (block 0 attention key), then expand only if stable. Same source hashes and fit partition checks as E. Direct Q2/Q3/Q4/Q6 controls, SVD8 at each .5/1 effective-bit ceiling independent of binary rank dedup. Charge packed scale bytes, report actual bits, decoded storage and theoretical MACs. No codec or trainer changes.

- F0: E3 initialization, no updates.
- F1: scale-only, signs frozen.
- F2/F3/F4: SmoothSign100 sign LR .0001/.0005/.001; independent scale LR .001.
- F5: 20 scale-only steps then F2 settings.
- F6: 20 primary-only updates, reinitialize residual against actual hard rounded primary error, then 180 joint updates (reset Adam). Rank allocation remains equal.
- F7: fixed quantized SVD8 codes and learned BF16 latent gains; charge 2 bytes per latent gain, reduce rank if required. Activation-only pooled objective; this is not full factor QAT.

Binary arms use common seeded row permutation stream, 256 rows, unbiased full weight-energy denominator, pooled fit target energy (zero/nonfinite aggregate rejected), .2 weight/.4 forward/.4 input gradient. Separate sign/scale gradient clipping. FP32 fit, actual hard packed BF16 scoring. Checkpoints at 0/20/50/100/200 (F6 starts at 20 after residual construction); select by fixed prior development objective, score selected checkpoint on prior confirmation. Initial checkpoint eligible; improvement cannot be assumed. Full weight error computed in tiles at checkpoints. Record cumulative/net sign flips, gradient norms, scale multiplier extremes and near-zero fraction, SmoothSign saturation proxy. F6 flip statistics reset after primary phase. This changes normalization and optimizer relative to E6, so F4 is not an exact replication. Fit/evaluation precision gap remains. Both scoring captures were inspected in earlier research; neither is fresh independent confirmation.

Validation: locally executed exact optimization excerpt from d169b511 (3870 characters, FNV-1a 4049480548) with independently copied existing Path/actions helpers on CPU torch 2.10; scale-only freeze, callback autograd, warmup, deterministic repeat, zero-energy guard passed. Added repo regression tests; full checkout pytest, complete runner, CUDA, actual archive and LoRA training not executed. No new independent review of F implementation yet. Next: inspect selected checkpoints and separate forward/dX errors; fresh captures across timesteps/images/token types before claiming generalization; hybrid SVD8 + binary residual LB-G after stabilization. Preserve prior E runner and negative results.


## Completed LB-F GPU pilot — 2026-11-27

User report `littlebit-qat-stabilization.json`, b3de63404c2572f47f089890e38fcf5315562e53, protocol binary-qat-stabilization-v1, 226.5 s, CUDA torch 2.15.0+cu134, 22 candidates, key0 only. Raw, archive and four capture SHA256s match LB-E. Fit steps 3/4, prior scoring steps 2/1. All 12 optimized binary arms improve both prior scoring objectives vs respective E3 initialization at their selected checkpoints. Objectives below are mean squared relative forward/dX error; not learning retention.

| Arm | .5-bit dev / prior conf | 1-bit dev / prior conf |
|---|---|---|
| F0 | .08085389 / .11846529 | .06185954 / .07902648 |
| F1 | .05202418 / .08160517 | .03132532 / .04728452 |
| F2 | .03358429 / .05503262 | .01510494 / .02579067 |
| F3 | .04383114 / .08602451 | .02405090 / .03976174 |
| F4 | .04064533 / .07345782 | .02997510 / .04219981 |
| F5 | .03408130 / .05969315 | .01461367 / .02512366 |
| F6 | .03097206 / .05594744 | .01485359 / .02545772 |
| SVD8 / F7 | .05741004 / .12383907 | .02457421 / .09192431 |

F5 1-bit: rank 597+597, 1179348 packed bytes, .9997457 effective bits; step100, weight error .27707647 vs .31867779 F0; development relative fwd/dX .0413363/.1658875, prior confirmation .0422357/.2201442. Direct Q2 (2.25 bits) .04192738/.05959194 objectives, Q3 (3.25 bits) .01160663/.01751086, Q4 .00351161/.00531331, Q6 .00018363/.00027225. F5 beats Q2 on both panels but not Q3; storage advantage does not imply VRAM/speed gain (binary decoded factors 18372948 bytes, 97.17% dense-equivalent MACs). F2 and F6 near F5 at 1 bit; F6 best .5-bit dev, F2 best .5-bit prior conf. F2/F5/F6 select step100 at both caps and worsen development by step200. F1 scale-only improves without any sign change, but F2/F5/F6 improve further. F7 selects step0 both caps; scalar latent gains alone did not improve SVD8. Scale multiplier near-zero fraction zero at all logged checkpoints. At 1-bit F5 step100 weighted net sign differences 2.82%, cumulative flips .1471 per element; step200 4.60%, .7002 and dev .018107. F4 step200 18.60% net, 1.9229 cumulative per element, worse dev .029975. Weighted by actual factor element counts; cumulative events are not unique flipped signs. Observed oscillation supports smaller LR/decay test, not proof of causality.

Decision: retain F2/F5/F6 as provisional finalists, F1 as control, deprioritize high sign LR and scalar-only SVD8 gains. Next gate: repeat narrow finalists on key14 and MLP up0; fresh capture panels with diverse images/timesteps/token types required before selecting general policy. LB-G same-byte SVD8 primary + binary residual allocations 25/50/75% SVD8 with matched pure endpoints, optimize with low sign LR, report theoretical MACs and decoded bytes as well as local error. No fresh independent confirmation, complete block test, archive, runtime peak VRAM, or portable LoRA result. No code changes in this result review.
