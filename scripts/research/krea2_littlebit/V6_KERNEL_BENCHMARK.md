# V6 stage 1 — Q6 compute grouping

V5 GPU proof of concept completed; source evidence: user-held littlebit-report(1).json and littlebit-results.zip, group littlebit-20261127-035136-abc368. 224 Q6 projections, 9,496,166,400 targeted packed bytes vs 12,161,859,584 native INT8 bytes (21.92% reduction). Clean matched 16-swap timing 3.898 vs 38.204 s/update. Q6 timing at 12 swaps finished (not proof of minimum feasible swap count). At 16 swaps Q6 allocated 7.68 GiB vs 8.74 GiB stock. Three holdouts, twelve updates, single seed; no interpretable retention ratio. Q6 adapter technically evaluated on three deployment bases. Shared WDDM allocations alone do not establish paging.

Grok review highlighted launch and GEMM fragmentation. Code already rotates X once per forward and accumulated dX once per backward; do not rotate per weight tile. The archived Q6 format interleaves codes and scales in 128-row storage tiles. Larger compute groups must respect those boundaries, not simply call the existing decoder with larger rows. Large BF16 MLP projections reach 192 MiB decoded, excluding unpack intermediates.

New standalone kernel_bench.py uses the existing V5 archive, verifies original source SHA256, computes an archive SHA256 fingerprint (not authenticated integrity for legacy archives), and sweeps grouped decode and batched unpack plus larger GEMMs. Real weights for eight projections in block 9; synthetic 1280-token BF16 inputs, seed 42; stock W8A8 control; same Q6 reference for forward and input-gradient comparison; warmups and synchronized wall-clock timing. Reports allocated/reserved peaks and per-operation incremental allocation. No native loader, training or archive changes. BF16 partial-sum rounding differs with GEMM grouping; bitwise gradient parity is not required. Compare errors and investigate before full-model integration; no automatic promotion. Allocator cap is not WDDM residency enforcement. OOM recorded for timed arms; initialization failures can terminate the benchmark. No CUDA, Triton or full-model validation executed by implementer. Run regression tests on checkout.

Example from repo root:

```powershell
.\venv\Scripts\python.exe scripts\research\krea2_littlebit\kernel_bench.py --raw "PATH_TO_RAW_BF16" --archive "output_loras\littlebit_comparisons\littlebit-20261127-035136-abc368\q6-had\compact.safetensors" --out output_loras\littlebit-kernel-v6.json
```

Next: select Pareto candidates by timing, scratch and numerical agreement; test real captured inputs and full-model matched swaps/allocator budgets; only then consider fused kernels or changed precision. No production integration.
