# Experimental stream calibration profiles

Profiles are opt-in, local, workload-specific recommendations. This branch does
not automatically modify the normal trainer, WIP, or production preferences.

## Full calibration

From the Fizgig environment, use real Krea 2 checkpoint and matching cached
latent/text inputs. Allow sufficient disk space for reference gradients and
updates. Replace the paths with your actual paths:

```powershell
python scripts/diagnose_stream_sync.py --suite calibration --steps 5 --warmup 2 --swap 22 --rank 32 --precision int8 --optimizer muon --model "H:\models\krea.safetensors" --cache "H:\dataset\cache" --profile-dir "$env:LOCALAPPDATA\Fizgig\stream-profiles"
```

A profile is saved only if the same qualified strategy wins the repeated
real-GPU benchmark, exact parity checks and memory checks. Synthetic and CPU
runs cannot create profiles. The checkpoint and actual frozen input tensors
are hashed; hashing a large checkpoint adds an initial disk read.

Run the same command with `--lookup-profile` to check whether a profile matches
and whether current free RAM and VRAM meet its measured requirements. This does
not load the model onto the GPU or train. Use `--reuse-profile` instead to
repeat only the saved candidate, staged ring 2 and classic baseline. A miss
runs the normal calibration suite. Reuse still rechecks numerical parity,
timings and memory; it does not skip verification.

Profiles match the GPU identity, driver, OS, CPU, RAM/VRAM, PyTorch/CUDA, exact
tracked-clean source commit, checkpoint content, cached input content, seed,
rank, precision, swap count, optimizer, learning rate, and measured step
schedule. Changes require a new calibration. Missing hardware identity or a
dirty tracked source tree prevents profile creation, without preventing an
ordinary diagnostic.

## Scope

The current profile workflow covers Krea 2, LoRA, batch size 1, checkpointed
training, auto SDPA, BF16 or INT8, AdamW or Muon, with actual cached inputs. It
does not establish parity for all production datasets, Global Replay, resume,
MOSS-Image, LoKR or arbitrary compile/attention configurations. A profile is a
local recommendation for its measured workload, not a hardware-wide guarantee.
The normal trainer is intentionally not yet wired to apply it.
