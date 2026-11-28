# Heterogeneous frozen-base compression — approved follow-up

2026-11-28 user approved retaining LittleBit as a candidate for sensitive-to-family
heterogeneous compression, after the current QH full-model training pilot. This
reopens the parked LB-G/LB-I direction as a scheduled research milestone, not a
replacement for the ongoing QH preparation. Do not interrupt current archives
or repeat broad tuning on previously inspected development data.

## Evidence and hypothesis

Same H3 bank, block14 attention key, raw BF16 references, development RMS
relative forward and input-gradient error:

| Candidate | Effective bits | Forward | dX |
|---|---:|---:|---:|
| LB-H10 joint-reset | .999746 | 21.174% | 48.399% |
| Q2 H64 G64 | 2.25 | 13.368% | 31.993% |
| targeted Q2/Q3 | 2.499512 | 11.251% | 23.070% |
| Q3 H64 G64 | 3.25 | 6.801% | 17.757% |
| Q4 H64 G64 | 4.25 | 3.640% | 9.865% |

Sources LB_H10_PROGRESSIVE.md, QH_ALLOCATION.md and H4 rank-reference report.
Earlier key0 LB-F5 at .999746 bits achieved 4.134%/16.589% on its earlier
development capture and 4.224%/22.014% on previously inspected confirmation;
its matched direct Q2 action objective .04192738/.05959194 versus F5
.01461367/.02512366. On the later H3 bank key0 Q2 7.250%/25.986%, targeted
mix 3.793%/14.687%, Q3 2.177%/13.698%. Do not treat the cross-bank numbers as
matched comparisons. LB-G hybrid SVD8 + binary residual at approximately
1 bit: key0 G50 .009867/.019843, pure F5 .014614/.025124 on same older
panels; theoretical MAC ratio .55257 vs .97168. MLP0 G75 .007551/.001997,
SVD8 .011096/.002492. Key14 binary and hybrid remain poor. See
LB_F_STABILIZATION.md and LB_G_HYBRID.md. No full-model learning or physical
VRAM result for these optimized binary or hybrid candidates.

Hypothesis: preserve low-sensitivity matrices using one-bit binary factors
or hybrid SVD8/binary; protect other matrices with direct Hadamard Q2/Q3,
Q4/Q6 or stock INT8 as necessary. Average eligible storage below 2 bits is
an aspiration only, not a mandatory target. Do not force one codec globally,
or infer matrix sensitivity solely from depth, shape, Frobenius error or
a single capture. Eager factor decoding can negate savings.

## Ordered work

1. Complete current QH preparation and actual model training comparison.
   Include stock INT8, existing Q6 Hadamard, Q4 Hadamard (prepare separately),
   uniform Q3/Q2 and mixed Q2/Q3. Same raw source, split, initialization,
   rank32/alpha32, optimizer/LR/EMA settings and sample seeds. Short stability
   first, then matched longer learning comparisons of promising arms.
   Evaluate adapters on the same stock base; compare to no-adapter and stock
   training, not simply local reconstruction or training loss. Keep Q4 and
   Q8 as essential controls. Find minimum stable swap counts at .25/.5/.75/1
   MP only; physical dedicated/shared memory, transient scratch, warm step
   time and quality. No resolution above1MP. Do not require all arms fit
   at same swap count for residency, but also compare at matched swap count
   when possible for speed and learning isolation.
2. Lock binary/H10, hybrid G50/G75 and SVD8 candidates, direct Q2, Q3,
   task-targeted Q2/Q3, Q4/Q6 and INT8. Preserve existing archives. Collect
   a genuinely independent calibration/evaluation bank with distinct
   source images and diverse timestep/token coverage. Extend coverage over
   all eight projection families and depths, ultimately all224 eligible
   matrices. Bound capture storage by streaming per-family/layer groups;
   separate fit/development/untouched confirmation. Fit only on calibration,
   select on development, confirm locked choices on untouched cases. Prior
   H3 confirmation was consumed by H3; do not reuse it as fresh. Avoid
   exhaustive optimizer sweeps; start narrow with locked recipes.
3. Build measured per-matrix candidate table: packed bytes incl masks/scales,
   forward/dX errors and cosine, portable LoRA gradient/update agreement,
   decoded scratch and peak VRAM, latency. Reject nonfinite or unsupported
   operators. Compare equal-byte policies and weight-only baseline. Use
   full-model QH learning outcomes to inform provisional fidelity bounds;
   no universal local threshold until validated. Choose nondominated
   candidates, then solve global constrained allocation with per-family
   layout and memory/runtime constraints. Compare lowest-byte feasible
   policy and conservative policy; include direct-only policy as ablation.
   Evaluate whole block and full denoiser before longer training.
4. Existing compact_loader requires identical specs across 28 blocks.
   QH row masks preserve this, heterogeneous codec/rank choices do not.
   First consider uniform per-family formats/counts across blocks and
   actual measured common layout. For genuinely per-matrix choices, design
   heterogeneous swap support with shape/dtype/length-safe transfers and
   explicit buffer capacity accounting; no silent bypass of layout guard.
   Padding to common slot sizes must count against resident and transfer
   memory. Test actual 28-block swap equivalence, checkpointing and LoRA
   export/reload. No persistent dense weight or decoded factors that erase
   the gain; use bounded tiling or fused kernels only after parity.
5. Save a reusable hash-identified full eligible projection archive; original
   raw required until noneligible weights are packaged. Compare heterogeneous
   vs strongest QH, Q4 and stock INT8 on same stock-base LoRA evaluation.
   Report actual minimum swaps, dedicated/shared GPU peaks, step times,
   archive bytes and learning quality. Prefer practical VRAM gains over
   sub-2-bit headline. Hugging Face distribution only after stability and
   redistribution review.

## Independent review and current status

Ask Astra to challenge allocation objective, cross-layer error accumulation,
binary factor scratch, heterogeneous classic-swap invariants, calibration
provenance and equal-budget comparison. Source:
https://github.com/Murf-Code/Fizgig-Advanced/tree/research/krea2-littlebit
QH-4B results commit 659437bbd43b48d11bcf4bb67c4ae7892dd365f3;
QH runtime ledger 58ee693b29934f4ca00df17f76ae8007626a1e37;
LB-H10 results 72ac33bf1cb916596b3746352afc70b0cfcb864a;
LB-G findings 077dbbc5649a789005ecee5a3cb34c44a3a46760.
No independent reviewer response yet. This document records the approved
follow-up, not implementation of heterogeneous allocation, new results,
a trained model, or a new test queue. Keep current QH preparation intact.
