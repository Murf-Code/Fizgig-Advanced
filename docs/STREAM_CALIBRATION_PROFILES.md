# Experimental stream calibration profiles

Profiles are opt-in, local, workload-specific recommendations. This branch does not automatically apply a profile to normal training or modify WIP or production preferences.

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
## Normal training (experimental)

In Preferences > Training memory > Stream strategy (experimental), select
the measured staged or fully pinned strategy manually. Default preserves the
existing staged two-slot behavior. Changes apply at the next Start, and are
ignored for classic swap, forward-only inference, CPU, unsupported families,
and Global Replay. The GPU ring remains capped by the classic block-weight
VRAM budget, and host pinning still goes through the live RAM safety
preflight. A fully pinned request can fall back to staged transfers; the host
staging pool can shrink independently of the GPU ring.

For CLI training, the equivalent option is
`--send_only_strategy {default,staged3,staged4,direct2,direct3,direct4}`.

Do not treat a diagnostic profile as automatically applicable to a normal
training dataset. The diagnostic's workload identity does not establish
equivalence to an arbitrary production training run. Automatic profile
application remains gated on broader lifecycle and real GPU validation.
