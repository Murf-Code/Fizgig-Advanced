# LoRA matrix geometry research

Isolated from release and Global Replay. No trainer behavior changes in this first increment.

## Initial laboratory
- Compact QR/SVD diagnostics, exact adapter difference norm and offline spectral balancing/truncation.
- Synthetic linear regression screening: AdamW, batched Muon, PoLoRA, LoRA-TSD with and without scalar balancing. Zero-up initialization, identical data and initial factors, explicit LR sweep.
- CPU/CUDA CLI; JSON records individual failures rather than silently substituting optimizers. The diagnostics check full-product references only on small matrices. BF16 tests use rounded values with FP32 decomposition; not BF16 optimizer validation.

Run: `python scripts/bench_lora_matrix_geometry.py --device cpu` from the repo root. On CUDA use `--device cuda --rank 64`. This is not the diffusion benchmark. Synchronization and diagnostic overhead are intentionally included in timings; peak allocation is process-wide, not isolated optimizer state.

## Gates before diffusion testing
1. Verify compact calculations, exact update cap including quadratic term, zero/rank-deficient behavior, Muon equivalence, optimizer resume and memory.
2. Audit source for paired parameter eligibility, EMA and dropout. Do not change live factor basis without consistent optimizer/EMA state handling.
3. Implement separately validated Riemannian preconditioning, spectral balancing, Riemannion and RoM prototypes using primary papers. Do not label a generic preconditioner as a reproduced paper.
4. Screen on multiple synthetic shapes, seeds and conditioning regimes. Select candidates based on convergence, stability and cost. Do not infer image quality from synthetic loss.
5. Build Krea 2 comparison: rank/alpha 64, five epochs, all eligible blocks, fixed seeds/data/holdout/samples, no replay/selective/adaptive LR/EMA/rank dropout. Tune learning rates on separate calibration data; report equal-step and equal-wall-time comparisons. Exclude SOAP, Prodigy, Adafactor. Export JSON, readable log and sample ZIP.
6. Offline rank 32/16/8 compression, preserving alpha/r scaling and testing same prompts/seeds plus held-out quality. Never overwrite original adapters. Rank reduction and better learning are different outcomes.

No GPU evidence yet. The balanced-factor helper is an offline mathematical primitive, not a live optimizer. No claim of improved quality or file size.
