# Krea 2 Global replay

An opt-in Krea 2 LoRA training path. It keeps the **same global flow-matching objective as ordinary Krea 2 training**; only how the gradient is computed changes.

## Enable it

In Other Options, set **Training method** to:

- **Standard** (default): normal training, exactly as before.
- **Global replay**: the method below. A **Selective block check** tick (off by default) appears under it.

On the command line:

```text
--family_option training_method=global_replay --family_option selective_block_check=1
```

Leave out `selective_block_check=1` for the check off (the default). The test build's `training_method=selective_global_replay` and `selective_global_replay_h2d` still work and mean Global replay with the check on. Its other methods (Blockwise flow, Residual-path hybrid) are no longer offered: a run naming one logs a note and trains Standard.

## What it does

1. Fizgig samples noise and a timestep on the usual Krea 2 schedule and builds the usual target.
2. It runs the frozen input/text setup, then the 28 main transformer blocks one at a time with no autograd graph kept across blocks. Each block's input is copied to CPU.
3. The loss is measured at the real `dit.last` readout, the same prediction used by normal training and validation.
4. Backward starts at the readout and replays the blocks in reverse, each receiving the gradient from the block after it. The LoRA gets the exact chain-rule gradient of the global loss, not 28 separate local losses.
5. Validation and previews use the regular full-model path.

When blocks swap, it uses the H2D-only (send-only) block swap: each block is replayed and backpropagated before its ring slot is reused, so no device-to-host swaps are needed. `torch.compile` and gradient checkpointing are off for this path, because the replay already bounds live activations.

## Selective block check

The check is off by default. With the tick on:

1. All blocks train for the first 8 optimizer updates.
2. At each check (update 8, then every 8), with all blocks active, Fizgig scores each block's movement since its previous check against the fresh global-loss gradient at the check, before that update's step: `cos(-g_now, theta_now - theta_last_check)`. Negative means the block's recent updates carried it past where the loss now points.
3. Blocks with a negative smoothed score skip their LoRA-factor gradients until the next check; all blocks are woken for the update before each check. If every block scores negative, the least negative stays active.
4. The scores, active blocks and counter are saved in `training_state.json` and restored on resume. The first check after a resume only re-baselines.

A skipped block still runs forward and still passes the gradient back to earlier blocks, so the saving is only its small LoRA-gradient work. The check is a noisy first-order test, not a guarantee.

The test build scored each step against the same gradient that produced it. Any normal optimizer (SGD, Adam, Muon) always steps downhill against its own gradient, so every score was positive and no block was ever skipped ("28/28 blocks active" at every probe).

With the tick off: every block trains on every update, with no checks and no scoring overhead.

## Requirements and limits

- Krea 2, standard LoRA, batch size 1 (use gradient accumulation for a bigger effective batch).
- Trains all 28 main transformer blocks; text-fusion and input/timestep/output adapters stay frozen. Custom block subsets are rejected.
- Rejected: fine-tuning, LoKR, sliders, DOP, masked/prior loss, rotated INT8 backward, training attention maps, shared/helper training.
- Saved block inputs live in CPU RAM; at larger sizes that can be several GB.

## PoLoRA

PoLoRA works with Global replay. Global replay gives every LoRA pair the same gradient Standard training does, which is all PoLoRA needs.

- **Check off (default):** all 28 blocks get their gradients on every update, and PoLoRA steps every down/up pair as it does under Standard.
- **Check on:** a skipped block's pairs have no gradients for that update. PoLoRA leaves those pairs out of the step entirely: their weights, momentum, Q/P curvature diagonals and sigma_max warm starts stay exactly as they were, with no momentum decay. The pairs that did get gradients get the same update they would get if every pair had trained. A pair with only one of its two gradients is an error, because PoLoRA's two sides are coupled.
- A block that wakes after being skipped resumes with the momentum it had when it was paused. The check's scores were tuned on Muon runs, so it may rank blocks differently under PoLoRA.

## Results so far

Chris's A/B (same dataset, Muon, block swap 16, int8, 0.25 MP), with the check never skipping a block: Global replay ran about 2.5 s/step with an 8.8 GB peak and 0.139 validation loss, against Standard's 3.7 to 4.4 s/step, 11.0 GB and about 0.144.
