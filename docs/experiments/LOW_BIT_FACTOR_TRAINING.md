# Direct packed LoRA factor training — initial CPU experiment

Research proxy only. Run: python scripts/research/matrix_geometry/low_bit_training.py results.json
PyTorch 2.10.0 CPU, one thread. 64 assertions passed; 84 trajectories (7 configurations x 2 regimes x 6 seeds), 240 updates each. Rank 8, 96 input/64 output dimensions, minibatch 32, noisy synthetic linear target, Adam equations with no weight decay, matched cosine schedule within each regime (initial LR 0.02 growth, 0.002 refinement; final 10%). No independent LR search. Not an exact diffusion training implementation.

Direct variants retain integer codes and FP32 block scales between steps, no persistent FP32 factor master; compute, gradients and Adam moments remain FP32. INT4 is physically packed into nibbles. Group size 64, symmetric dynamic max-abs scale, deterministic nearest or stochastic rounding. QAT controls use FP32 master factors and deterministic fake quantization in forward and evaluation. Stochastic rounding RNG is independent of the data RNG. This is not QLoRA or quantization of the frozen base. The reference for direct stochastic training is Zhao et al., https://proceedings.mlr.press/v304/zhao26b.html; not a reproduction of their method.

## Measured mean normalized squared effective-weight error, last 20 steps

| Case | Growth | Refinement | Factor bytes | Adam moments bytes |
|---|---:|---:|---:|---:|
| FP32 | 0.000010196 | 0.000004268 | 5120 | 10240 |
| INT8 nearest | 0.001673377 | 0.006754760 | 1360 | 10240 |
| INT8 stochastic | 0.000287165 | 0.000227463 | 1360 | 10240 |
| INT4 nearest | 0.850902218 | 0.070044693 | 720 | 10240 |
| INT4 stochastic | 0.069855919 | 0.064999695 | 720 | 10240 |
| INT8 QAT | 0.000126705 | 0.000134919 | 5120 | 10240 |
| INT4 QAT | 0.039991397 | 0.040979442 | 5120 | 10240 |

INT8 SR improves markedly over nearest rounding but remains worse than FP32, particularly refinement. INT4 direct is not a quality-ready candidate. Storage savings: 73.44% INT8 and 85.94% INT4 factor bytes only; with FP32 moments 24.48% and 28.65%, excluding gradients, scratch, allocator and metadata. No GPU VRAM result. Packed-byte equality statistic is not a measure of stalled effective updates because scales change and INT4 shares bytes.

Checks cover finite round trips, shape, exact packed-byte counts, zero blocks, RNG replay of packing, one-scale error bound, and stochastic tiny-update expectation with fixed scale anchor. Full optimizer checkpoint/resume and standard LoRA export not tested. No autograd integration; analytic gradients in small CPU proxy. Direct live training requires recomputation or bounded per-layer dequantization in backward; simply replacing storage while autograd retains all dense factors may erase peak-memory gains. Next: per-rank group layouts and scale drift, precision promotion, FP16/BF16 baseline, compressed momentum, actual GPU kernel and resume/export verification. Avoid changing live trainer/queue until verified.
