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


## Completed LB-G GPU screen — 2026-11-27

User report littlebit-hybrid-qat.json, 96ace540da4629f28f43435009c0e206f8c978b4, hybrid-svd8-binary-v1, CUDA torch 2.15.0+cu134, 871.297 s, 61 candidates (22/22/17), three projections. Raw/archive/capture hashes match LB-F. All 18 hybrid arms selected step100 (last available), all logged near-zero scale fractions zero. Same two fit captures (3/4) and two previously inspected scoring captures (2/1); no independent confirmation.

| Projection / ceiling | G0 best dev | G25 dev / prior conf | G50 dev / prior conf | G75 dev / prior conf | SVD8 dev / prior conf |
|---|---|---|---|---|---|
| key0 .5 | F6 .030972/.055947 | .025696/.043093 | .028732/.047129 | .032023/.079992 | .057410/.123839 |
| key0 1 | F5 .014614/.025124 | .014174/.024477 | .009867/.019843 | .011297/.024335 | .024574/.091924 |
| key14 .5 | F6 .261251/.281041 | .258785/.289546 | .275733/.302276 | .301284/.331742 | .363866/.391862 |
| key14 1 | F5 .161586/.177727 | .179807/.189180 | .182737/.196684 | .212350/.228635 | .275223/.292787 |
| MLP0 .5 | F5 .049221/.043998 | .031614/.013511 | .018904/.005141 | .018685/.004309 | .027396/.005916 |
| MLP0 1 | deduplicated .46582-bit arm | .015019/.004311 | .009231/.002508 | .007551/.001997 | .011096/.002492 |

G0 column selects on development only; .5-bit confirmation best pure binary F2 key0 .055033, F2 key14 .274227, F1 MLP .040639. MLP1 G25/G50 actual .680015/.899357 effective bits, not equal byte; G75 .999780. Key0 1-bit G50 r76+603, packed 1179430 B, 10446006 decoded BF16 factor B, 55.26% dense MACs, weight error .228297. F5 r597+597, 1179348 packed B, 18372948 decoded B, 97.17% dense MACs, weight error .277076. G50 improves dev 32.48%, prior conf 21.02%, MAC count 43.13% vs F5. G75 r115+291 uses 33.04% dense MACs, 6252102 decoded B; dev/conf both slightly better than F5. G50 vs Q3 (3.25 bits) dev better, prior conf worse; prior conf forward/dX 2.125/19.808% vs Q3 2.146/18.591%. Key14 F5 still has prior conf forward/dX 22.12/55.37%, direct Q2 13.20/32.99%, Q4 3.64/10.25%. Key14 weight relative error .60954 F5 vs .59804 F0; forward improvement dominates. MLP1 G75 r418+1106, 12580140 packed B, 68712612 decoded B, 34.11% dense MACs, prior conf fwd/dX 5.351/3.362%, vs SVD8 12.49% dense MACs and objective .002492. MLP G75 beats SVD8 both panels at .5/1 bit but costs more compute. Key0 G50/G75 and MLP G75/SVD8 form useful fidelity/compute alternatives. Do not equate archive bits with physical VRAM or theoretical MACs with wall time.

Decision: stop repeating broad local sweeps on the same four captures. Next gate fresh diverse stock capture bank, locked fit/development/untouched confirmation partitions; predeclare narrow candidate set (key0 G50/G75, key14 direct Q2/Q3/Q4 and binary comparator, MLP G75/SVD8; include stock INT8 and Q6 references). Expand representative projection families and depths; test full transformer block output/dX and LoRA effective update parity before full model training. Adaptive heterogeneous precision allocation is a hypothesis, not a validated policy. No archive generation, actual runtime peak VRAM, training transfer, fresh capture test, or production change in LB-G. No code changes in result review.
