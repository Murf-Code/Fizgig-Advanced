# Matrix Geometry memory architecture — CPU stages 1 and 2 (2026-11-25)

Research baseline: eedc1319b57aa3cb537061beab9bf4523388a321. Standalone CPU research, not production integration; independent of Selective Global Replay.

Reproducible sources, raw JSON reports, logs and Windows launcher are in [stage 1](../../scripts/research/matrix_geometry/memory_cpu_stage1/) and [stage 2](../../scripts/research/matrix_geometry/memory_cpu_stage2/). Stage 1 was also independently rerun by the user in the Fizgig Windows venv: 24 momentum arms, loader continuation, malformed checkpoint, unique-storage accounting, phase budget and activation tests all passed.

Stage 2: 25 momentum recurrence arms (1000 updates), 30 fixed-factor geometry comparisons, 36 short and 36 extended synthetic closed-loop fits plus 24 half-precision controls, 10,000 budget simulation requests. Mock phase instrumentation passed. See the raw reports for individual seeds, metrics and assumptions.

Findings: INT8 block64 reduces outlier error relative to block256; Split-Max64 protects a single largest outlier but multiple outliers remain problematic. Synthetic learning convergence is worse for compressed candidates at 600 updates, with much smaller relative gaps after 1800 updates. BF16 momentum is a conservative candidate (~447 MiB estimated persistent saving if FP32 baseline); Split-Max64 (~640 MiB) and INT8 block64 (~657 MiB) remain research candidates. These are payload estimates, not measured training VRAM. No equivalence or real-image quality claim.

Next: phase-resolved CUDA instrumentation, isolated bounded checkpoint activation offload with H2D ring correctness, isolated BF16/INT8 state comparisons and real checkpoint resume. Test separately before combinations. No trainer defaults changed. Keep rank64 genuine 1 MP on 12 GiB as an unverified objective.
