# Fizgig Matrix Geometry memory lab — CPU stage 2

Independent, read-only research for `eedc1319b57aa3cb537061beab9bf4523388a321`. No repository modifications. Not Selective Global Replay. No CUDA execution or production integration.

## Run
Double-click `run_cpu.bat` on Windows; the window stays open and saves `cpu-results-v2.log`. The launcher uses the user's Fizgig venv path; edit it if needed. Alternatively, from this folder in PowerShell:

```powershell
& 'H:\Stable_Diffusion\Fizgig\venv\Scripts\python.exe' -u .\run_cpu.py 2>&1 | Tee-Object -FilePath cpu-results-v2.log
```

The script prints each completed experiment group, writes `cpu_results_v2.json`, and returns an error on failed assertions. `--steps 1200` extends the closed-loop runs. Default 600 steps. PyTorch is required. `run.log` is the included run's output.

## What is measured
- Five codecs: symmetric INT8 with 256 or 64 values/scale; split-maximum INT8 with 64 or 256 values/scale; and block64 with up to 5% FP32 fallback blocks. The split stores the largest magnitude value and its index exactly and quantizes the rest. Two outliers in a block can defeat it. The hybrid retains the original quantized backing and adds exact copies for a bounded set of blocks ranked by normalized coordinate reconstruction error. No GPU performance or loader claim for either new codec.
- 25 exogenous 1,000-step EMA recurrence arms, five gradient distributions; mean relative error for the last 100 steps; weak coordinates defined by 0 < |reference momentum| < 1, with 1e-9 floor.
- 30 fixed-factor geometry comparisons (rank 16 and 64, three dense gradient scenarios, 160 iterations, last 40 analyzed), comparing Muon-style FP32 momentum against each compressed state. BF16 five-step Newton–Schulz coefficients match the research implementation. Actual effective update is B dA + dB A + dB dA, not just the linear tangent. Weak projected error uses the lowest quartile of non-negligible singular directions of the FP32 update (threshold 1e-5 of leading singular value), projected onto matching U and V; not a full subspace distance metric. Fixed factors isolate state error; no diffusion model.
- 36 closed-loop low-rank matrix fitting runs (two ranks, three seeds, six methods, identical zero-B initialization and decaying LR). Uses analytical gradients, identical scalar LR for both factors, no weight decay, no trainer-specific LR scaling; not the complete Fizgig Muon optimizer or image quality evaluation.
- Dynamic driver-free budget simulation, 10,000 randomized requests, explicit refusal when one unit cannot fit, shrinking immediately and growing after three favorable samples. Mock phase peak coordinator preserves the whole-run peak across resets and rejects resets inside active phase. This is not actual CUDA telemetry integration.

All numerical metrics are descriptive, not statistical equivalence tests. Storage estimates exclude padding, allocator rounding, scratch and metadata beyond explicitly stored tensor payloads. FP32 state must be the actual baseline for the estimated savings.

## Proposed integration — not applied
- M3.7/M3.8/M3.9/M3.10: before GPU, validate real factor traces, multiple outliers, singular-direction behavior, codec serialization, and packed-state restoration of any selected novel codec. Measure allocation lifetimes of argsort, gather, scatter and per-block copies. A faster custom kernel is not justified without measured quality and scratch evidence.
- M4.4: use dynamic pressure policy only at safe boundaries; keep pending persistent allocations and outstanding transfer commitments disjoint from already allocated bytes. Do not retry partially mutated optimizer steps. A refusal is not a recommendation to change rank or precision automatically.
- M4.5: adapt PeakLedger to a CUDA backend with explicit device (max_memory_allocated, max_memory_reserved, reset_peak_memory_stats); route every existing epoch/preview/geometry reset and reader through the single owner in `families/train.py`. Use passive mode by default, only enable phase resets in diagnostic mode. Real phase markers must surround actual forward/backward/update calls, including accumulation and preview. No resets in overlapping scopes; flush maxima before every reset. CUDA allocator enqueue intervals do not equal hardware execution residency; correlate with event traces and external WDDM/NVML telemetry. Existing `PeakOwner` in memory_core is an earlier untested scaffold, not an integrated solution.

Next GPU gate remains isolated original INT8 block64 and bounded sync activation offload at swap22. Novel codecs are CPU research candidates only; evaluate before adding complexity. No repository modifications.

### Additional controls and telemetry
`python run_extended.py` repeats all 36 closed-loop arms for 1,800 steps using the same schedule shape over the longer horizon. `python run_half.py` tests BF16 and FP16 storage with FP32 reconstruction and arithmetic, for 600 and 1,800 updates. These alternatives are not integrated in Muon-ME. FP16 may underflow for small gradients; BF16 has a much wider dynamic range. `python phase_probe.py` exercises the separate PhaseProbe adapter with mock allocator counters, testing passive/diagnostic mode, exceptions, and reset protection. Its CUDA backend has not been executed. All these commands save separate JSON outputs. The main report's budget section was rerun with stronger plateau/shock coverage after the original execution (see budget_note). A conservative hysteresis policy can refuse some requests that fit the instantaneous snapshot; this is deliberate, not a claim of optimal throughput.
