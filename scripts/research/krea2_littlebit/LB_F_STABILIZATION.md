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
