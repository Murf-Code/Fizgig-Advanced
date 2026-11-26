# Muon-ME isolated CUDA results — 2026-11-25

Source: user-pasted console output from `muon_cuda_bench.py --rank 64 --output muon-cuda-report.json`, benchmark commit dc0fb89. The full JSON report was not supplied at time of this record; the companion JSON is an exact numerical transcription of the five printed summary dictionaries, not a replacement for the original report. Device/version metadata and per-step records are not established by this console summary. Known test design: 528 extrapolated Krea 2 rank-64 FP32 factor shapes, synthetic externally supplied gradients, two warmups and six measured steps. All arms finished and reported finite weights and momentum.

| Arm | Warm extra MiB | Total allocated GiB | Total reserved GiB | Median step ms | Relative weight discrepancy (%) |
|---|---:|---:|---:|---:|---:|
| Reference | 1358.160 | 3.9876 | 4.3223 | 65.984 | — |
| Reference repeat | 1358.160 | 3.9876 | 4.3223 | 66.213 | 0 |
| Shape-local | 750.625 | 3.3932 | 3.7598 | 65.890 | 0 |
| Chunk 128 MiB | 60.000 | 2.7095 | 2.7773 | 56.958 | 0.09455 |
| Chunk 32 MiB | 26.141 | 2.6868 | 2.7148 | 221.111 | 0.13797 |

128 MiB versus reference: 95.582% lower warmed scratch, 1.27817 GiB lower peak allocated, 1.54492 GiB lower peak reserved, 13.679% shorter optimizer step. Reference repeat time differs by 0.346%; only one repetition of each ME arm, no statistical confidence interval. Shape-local is bitwise identical to reference and reduces warm extra by 44.732%, while chunked arms retain bitwise identical momentum under the fixed supplied gradients but not weights. Relative discrepancy is normalized by reference cumulative endpoint displacement from initial weights, not weight norm or per-step error. Maximum absolute weight difference 6.12368e-6 at 128 MiB, 5.93183e-6 at 32 MiB. The 32 MiB arm saves another 33.859 MiB of warmed scratch at 3.882x the 128 MiB step time. Prefer 128 MiB as the initial CUDA candidate; shape-local is a reproducibility control, 32 MiB a memory-pressure fallback candidate only after calibration. Budget is an estimated explicit scratch cap, not allocator guarantee.

## Gate and next work

Isolated CUDA feasibility passed for this configuration. No conclusion yet about training loss, visual quality, full training peak VRAM, other ranks/devices, quantized state compatibility, or training speed. Do not add savings from independent phases. Obtain original report for device/version and per-step analysis. Next: opt-in real Krea 2 matched training control versus Muon-ME 128, optionally shape-local control; preserve existing adaptive TSD queue. Consider actual high-resolution rank-16/32 fit and calibrated state compression separately; no silent setting changes or production defaults.

Research register supersession: Muon-ME is now CUDA-tested in an isolated optimizer benchmark, not integrated into trainer. Earlier CPU-only status remains historical.
