# Muon-ME isolated CUDA gate

Run from repository root with Fizgig closed:
```powershell
.\venv\Scripts\python.exe scripts\research\matrix_geometry\muon_cuda_bench.py --rank 64 --output muon-cuda-report.json
```

No model or dataset needed. 528 real rank-16 factor shapes extracted from the Krea 2 checkpoint header, extrapolated to rank 64; synthetic FP32 parameters and externally supplied gradients. Five sequential arms: reconstructed BatchedMuon reference (matches step equations from b299a617), reference repeat, shape-local unbounded, 128 MiB, 32 MiB. Two warmups and six measured steps per arm. Record first-step total peaks and warmed extra allocator peak, total reserved peak, synchronized step time, finite values and final weight/momentum agreement. Compare against reference repeat before interpreting differences. CUDA driver memory snapshots are not a sampled device-wide peak. No quality, full training VRAM or speed claim. OOM recorded, no automatic setting changes. Output JSON atomically updated after each completed arm. CPU smoke (15 small tensors, two warmups/two measured steps) completed five arms; bitwise identical weights and momentum, no CUDA available for verification. CPU smoke does not exercise effective chunk splitting. Full-size CUDA gate required before trainer integration.

## Prior matched PoLoRA result recovered

Completed matrix-geometry-report(2).json: rank64, 0.25 MP, 22 swapped blocks, 29 steps each, same seed/LR/holdout. 128 MiB vs unchunked: CUDA allocated 8268686848 vs 9362822144 bytes; reserved 8808038400 vs 11744051200 bytes; sampled NVML 10.50 vs 11.98 GB as logged; step 6961.9 vs 8019.5 ms. ~1.02 GiB allocated and 2.73 GiB reserved reduction, ~13.2% lower step time. Both finished; baseline validation only. This is whole-training peak, not optimizer-only peak. No paired final quality proof. No need to repeat this exact test now.
