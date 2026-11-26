# Fizgig memory research — local CPU lab

Reference commit: eedc1319b57aa3cb537061beab9bf4523388a321. No repo changes. Independent of Selective Global Replay.

Run from this directory with a Python environment containing PyTorch:

    python run_cpu.py
    python matrix_memory_cpu_verify.py --repo H:\Stable_Diffusion\Fizgig

The second command requires the repository at the specified commit and exercises the actual existing QuantMuon prototype; it was not executed here. `run_cpu.py` was executed on PyTorch 2.10 CPU. `cpu_results.json` contains measured results from 24 fixed-seed, 1000-step momentum recurrence arms. These are exogenous synthetic gradients, not closed-loop Muon training or a quality result. Stochastic rounding uses a dedicated seeded generator for these tests only. The `PackedMomentum` class is an EMA-SGD state/loader harness, not Muon; its direct loader validates records before publishing and does not call the base loader, but is not a drop-in replacement and does not implement optimizer load hooks or transactional optimizer steps. GPU state replacement may overlap old/new storage. Codec finite checks synchronize on CUDA and are for verification, not performance. `PeakOwner` is an untested CUDA integration scaffold: every existing reset site must be routed through it to preserve global statistics. Its intervals reflect host enqueue boundaries and do not establish GPU phase residency. No CUDA tests executed.

## Integration proposals (not applied)
- M1: route existing epoch, preview and geometry-fit peak resets/readers through one owner; use passive counters in ordinary runs; trace only selected diagnostic updates. Capture phase boundary at actual driver forward/backward and optimizer calls, not inferred wall times. Add model/optimizer unique storage inventory and explicitly tracked host buffers. NVML and WDDM sample separately; do not treat shared memory as proof of paging.
- M2: add opt-in Krea 2 eager-only bounded checkpoint helper around the existing `checkpoint(block, combined, tvec, freqs, attn_params, use_reentrant=False)` branch; leave wait/submit outside checkpoint. Reject compiled, replay, and unsupported layouts for first GPU gate; ensure both activation and weight dependencies before recompute. CPU cap counts logical snapshots, not total host allocation; apply a separate available-RAM check. No async implementation yet. Do not enable by default.
- M3: use validated INT8 codec and direct packed-state restore pattern in QuantMuon, retaining its unquantized current momentum for NS. Do not replace the Muon math with the EMA-SGD harness. Validate stable parameter identities, full checkpoint schema, and absent versus uninitialized state. Measure packing scratch, and reject a single unit exceeding a strict budget. Add real factor gradient/momentum trace replay and effective BA update metrics before quality claims. Stochastic RNG must be included in resume contract if adopted.
- M4: phase_plan and scratch_plan are pure accounting references. All phases must include all simultaneously live allocations exactly once; allocator reserves are separately observed, not blindly added to live bytes. No automatic runtime changes, fallback retry, rank or LR changes. Host pinned budgets and transfer scheduling remain to be implemented.

## Proposed GPU run order (not an executable queue)
1. ME128 rank 64, 0.25 MP, swap 22 passive control and short diagnostic trace.
2. Same settings swap 16 diagnostic, stop on sustained pressure; then swap 21 and 20 probes only if safe.
3. Swap 22 bounded synchronous activation offload; compare outputs, input and factor grads, RNG, ring slot ordering, transfer counters, peak and timings.
4. Swap 22 INT8 momentum, independently; check persistent storage after warmup, pack/unpack peak, checkpoint save/resume, effective BA and heldout loss.
5. Combine only after individual gates; lower swap cautiously.
6. True 1024x1024 rank 16, then 32, then 64 if clean; record actual bucket/token count.

No production changes justified. No GPU queue or trainer patch supplied here. Existing standalone offload helper is restricted to immutable inputs and ordinary backward; actual CUDA ring correctness and memory savings remain unverified.
