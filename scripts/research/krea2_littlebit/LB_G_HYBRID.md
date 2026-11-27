# LB-G — SVD8 primary plus binary residual

## Hypothesis and evidence

LB-F completed on key0 (user littlebit-qat-stabilization.json, 226.5 s). F2/F5/F6 selected step100 at both .5/1 bit caps. At 1 bit F5 dev/prior confirmation .01461367/.02512366 vs F0 .06185954/.07902648, SVD8 .02457421/.09192431. All twelve optimized binary arms improved both prior panels. Pure binary still costs 97.17% dense-equivalent MACs at 1 bit on key0. F7 SVD8 latent gains selected step0 at both caps. See LB_F_STABILIZATION.md.

## Protocol

- Run all three existing projections: key0, key14, MLP up0. Same raw/capture/archive hash verification and split as LB-E/F.
- .5 and 1 effective-bit ceilings. G0 pure binary (F0 initialization, F1 scale-only, F2 .0001 sign LR, F5 20-step warmup, F6 primary-first); G25/G50/G75 fixed SVD8 primary charged to 25/50/75% target bytes plus one binary residual charged to remaining budget; G100 pure SVD8. Direct Q2/Q3/Q4/Q6 controls. SVD8 quantization uses actual packed-decode factors. Residual initialized against decoded SVD8 product, not continuous SVD.
- Hybrid F5-style training 100 steps, 20 scale-only warmup, sign LR .0001, scale LR .001; frozen SVD8 branch. Same .2 weight/.4 forward/.4 input-gradient objective with original teacher-energy denominators. FP32 fit; CPU BF16 hard packed-decode scoring through summed paths, autograd input cotangent. Match row stream and budget ceilings. Binary-only endpoints shortened to 100 steps (F1 had selected 200 in F). F6 uses 20 primary + 80 joint updates. No new sign-LR sweep.
- Fixed evaluation checkpoints 0/20/50/100, F6 starts at 20; select minimum prior-development objective. Prior confirmation is previously inspected, not independent. Record initial and selected hybrid fidelity, cumulative/net flips, scale health, actual bytes, ranks, decoded factor bytes and theoretical MACs.
- Rank ceiling min(m,n,2048) across both branches. This may underfill MLP 1-bit G25/G50 (estimated .680015/.899357 effective bits), and pure binary MLP rank allocation repeats at 1 bit; deduplicate binary only. Compare actual bytes rather than claiming exact equal storage. Key0 1-bit G25/G50/G75 projected MAC ratios .76823/.55257/.33040 vs .97168 pure binary; no measured GPU speed or VRAM claim. A fixed SVD8 primary is deliberately not joint mixed-precision QAT.

## Validation and next gate

Implementation aa2129e9, added regression tests 192e2825. Independent CPU arithmetic checks of the same rank/byte formulas for 12 real-shape allocations passed; synthetic FP32 fixed-primary forward/dX identities passed. Full repository checkout unavailable in current environment (DNS failure); committed test suite and complete CUDA runner not executed here. Run regression tests locally before GPU screen. No fresh captures, archive generation, full-block tests, portable LoRA, or production changes. Next gate: compare actual error/bytes/MAC Pareto fronts on all three projections, then collect fresh independent capture panels before broader selection; joint SVD8 QAT, sign decay, asymmetric rank split and full-block distillation remain potential follow-ups, not implemented.
