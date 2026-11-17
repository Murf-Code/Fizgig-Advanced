# Krea 2 Blockwise Flow and Selective Global Replay (experimental)

These are opt-in experimental Krea 2 training paths. Blockwise flow and selective replay use activation storage and reverse replay; Residual-path hybrid uses approximate residual identity-path credit between periodic exact replays. All retain the **same global flow-matching objective as ordinary Krea 2 training**. None is a complete implementation of the Dust, Noise, or NoProp research algorithms.

## Enable it

In the Krea 2 training options, set **Training method** to one of the seven choices:

- **Standard backprop** (default).
- **Blockwise flow (experimental)** — classic bidirectional block swapping when blocks are offloaded.
- **Selective global replay (experimental)** — Blockwise flow plus adaptive selection of active LoRA blocks.
- **Blockwise flow H2D (experimental)** — the same Blockwise flow algorithm using H2D-only block swapping.
- **Selective global replay H2D (experimental)** — the same selective algorithm using H2D-only block swapping.
- **Residual-path hybrid (experimental)** — approximate identity-path credit on most updates, with periodic exact global replay.
- **Residual-path hybrid H2D (experimental)** — the same hybrid algorithm using H2D-only block swapping.
- **Residual-path hybrid H2D — selective A (experimental)** — B on every approximate update, plus A every fourth approximate update, with H2D-only block swapping.

The Blockwise flow CLI option is:

```text
--family_option training_method=block_local
```

Standard backpropagation remains the default.

## What it does

1. Fizgig samples noise and a timestep using the existing Krea 2 schedule, constructs `x_t = (1 - t) * x_0 + t * noise`, and builds the existing target `noise - x_0` in Krea 2's patch-token layout.
2. It runs the ordinary frozen input/text setup, then processes the main transformer blocks one at a time. Each block's input activation is copied to CPU; the next block receives a detached activation, so no 28-block autograd graph is retained.
3. The forward loss is measured at the **real `dit.last` Krea 2 readout**, against the ordinary flow-matching target. This is the same prediction objective used by normal training and validation.
4. Backward starts at the real frozen final readout, obtains the gradient for the final transformer output, then replays the main blocks in reverse order. Each block receives the gradient from the next block, so the LoRA receives the chain-rule gradient of the global Krea 2 loss. This is not 28 independent local objectives.
5. Validation and previews continue to use the regular full-model Krea 2 path. Temporary warm-up heads are not included in the exported LoRA.

The main LoRA optimizer (including supported choices such as Muon, PoLoRA and LoRA-TSD) updates only LoRA parameters. The method stores block inputs on CPU and replays blocks during backward. Classic bidirectional block swap is supported; `torch.compile` and gradient checkpointing remain disabled for this path because the replay already controls activation lifetime.

### Blockwise flow H2D

Choose **Blockwise flow H2D (experimental)** or pass:

```text
--family_option training_method=block_local_h2d
```

This is the same Blockwise flow algorithm and global loss, not a different local-loss method. It selects Fizgig's existing `LoRAStreamOffloader`: frozen base Linear weights keep their CPU masters and stream host-to-device through a reusable GPU ring. Setup creates those CPU masters once; training steps do not swap block weights back device-to-host. During Krea 2 manual replay, only one block's local autograd graph is live and its backward completes before another block can reuse that ring slot, so this replay path does not need gradient checkpointing for H2D safety. The normal full-model H2D path still has its own checkpointing requirements. If a selected weight layout cannot be streamed safely, the offloader logs the reason and falls back to classic swapping. H2D is a transfer strategy; it does not alter Blockwise Flow's objective, block order, or optimizer behavior.

### Blockwise flow H2D

Choose **Blockwise flow H2D (experimental)** or pass:

```text
--family_option training_method=block_local_h2d
```

This is the same Blockwise flow algorithm and global loss, not a different local-loss method. It selects Fizgig's existing `LoRAStreamOffloader`: frozen base Linear weights keep their CPU masters and stream host-to-device through a reusable GPU ring. Setup creates those CPU masters once; training steps do not swap block weights back device-to-host. During Krea 2 manual replay, only one block's local autograd graph is live and its backward completes before another block can reuse that ring slot, so this replay path does not need gradient checkpointing for H2D safety. The normal full-model H2D path still has its own checkpointing requirements. If a selected weight layout cannot be streamed safely, the offloader logs the reason and falls back to classic swapping. H2D is a transfer strategy; it does not alter Blockwise Flow's objective, block order, or optimizer behavior.

## Optional head-only warm-up

The Krea 2 Other Options section exposes two controls when Blockwise flow is selected:

- **Blockwise head-only warm-up** (`--block_local_head_warmup_steps`): for this many optimizer updates, temporary per-block velocity heads are calibrated while every LoRA parameter is frozen. The LoRA optimizer is not stepped, so optimizer momentum and decoupled weight decay cannot alter the adapters. The LR scheduler and LoRA EMA do not advance. These heads are then frozen and are not used by the exported-model objective.
- **Blockwise LoRA activation ramp** (`--block_local_lora_ramp_steps`): after head warm-up, multiply the LoRA optimizer's current learning rate by `1/N`, `2/N`, …, `1` for N optimizer updates. This composes with the selected optimizer (including Muon), scheduler, and per-image LR multiplier. Set either control to `0` to disable that phase.

The temporary heads are a calibration phase only. Once the LoRA activates, the training loss switches to the real global Krea 2 prediction loss and the heads stop updating. Expect the displayed loss scale to change at that transition because the two phases report different objectives. The default head learning rate remains `1e-3` and the heads are excluded from LoRA export.

Both counts are optimizer updates after gradient accumulation, not image/micro-batch steps. For the earlier freeze/ramp experiment, 40 head-only updates followed by a 40-update LoRA ramp is a supported schedule. These settings are saved in resumable state; changing them when resuming is rejected to avoid silently changing the schedule.


## Selective global replay

Choose **Selective global replay (experimental)** in the same **Training method** control, or pass:

```text
--family_option training_method=selective_global_replay
```

This method uses the same exact global Krea 2 flow-matching objective and one-block-at-a-time reverse replay as Blockwise flow, but adaptively skips LoRA-factor gradient calculation for blocks whose recent optimizer updates point uphill relative to their own global-loss gradients. It does not invent independent per-block losses.

How selection works:

1. All 28 blocks train for the first 8 optimizer updates.
2. On the eighth update, Fizgig measures each block's actual parameter displacement against its gradient using the directional cosine `-dot(gradient, parameter_delta) / (||gradient|| * ||parameter_delta||)`. A positive score indicates that the optimizer displacement was locally downhill; a negative score indicates an uphill displacement.
3. Blocks with positive smoothed scores remain active. Blocks with negative scores have their LoRA parameters temporarily set non-trainable, so their adapter-factor gradients are not calculated. Blocks with no measurable score remain active; if every measured block is uphill, the least-uphill block remains active so training does not stall.
4. Every eighth optimizer update, all blocks are reactivated for one probe update. Their scores are refreshed and the next selective window is chosen.
5. The active block set, per-block scores and probe counter are saved in resumable training state.

**Important compute boundary:** a skipped block still runs in the forward pass, and backward still propagates the loss gradient through that block so earlier active blocks receive the correct chain-rule gradient. What is skipped is the gradient calculation for that block's LoRA factors, not the block's entire backward pass. This should reduce adapter-gradient work, but it will not eliminate the transformer's full backward cost or produce a 28x speedup. Selection is a first-order local test of the optimizer's actual parameter displacement, not a guarantee that the next minibatch's loss will decrease.

Selective global replay requires batch 1 and standard LoRA. It is not compatible with PoLoRA in this build because PoLoRA requires a gradient for every down/up pair; the selectively inactive pairs intentionally have `grad=None`. Muon, AdamW, and LoRA-TSD use the standard missing-gradient skip behavior and are supported by this path. Blockwise head warm-up and LoRA activation ramp controls apply to Blockwise flow only.

### Selective global replay H2D

Choose **Selective global replay H2D (experimental)** or pass:

```text
--family_option training_method=selective_global_replay_h2d
```

This is Selective global replay with the same H2D-only offloader described above. Its adaptive probe cadence, directional scores, inactive-block behavior, and global-loss reverse replay are unchanged. H2D changes only how frozen base weights are staged between CPU and GPU; it does not copy DUST logic or use independent block losses.



## Residual-path hybrid

Choose **Residual-path hybrid (experimental)** or pass:

```text
--family_option training_method=residual_path_hybrid
```

This is a separate, approximate credit-assignment method. It is inspired by residual identity-path credit and Dust-style forward-only learning, but it is **not** an implementation of the Dust or Noise research algorithms.

On ordinary hybrid updates:

1. Krea 2 runs one normal forward through the complete transformer with autograd disabled for the transformer blocks. The driver captures the low-rank `A x` activations for each block's `attn.wo` and `mlp.down` LoRA, together with the branch gates.
2. The real final readout and flow-matching loss are evaluated. Autograd is used only to calculate the loss derivative with respect to the final transformer output; no transformer-wide graph or block reverse replay is created on these updates.
3. That readout signal is approximated as travelling unchanged along the residual identity paths. At each block, the attention and MLP residual gates scale the signal used to calculate the corresponding output-projection LoRA `B`-factor gradient. `A` factors and other LoRA projections receive no gradient on these approximate updates.
4. Every Nth optimizer update, the method switches to the existing exact global-loss block replay. That calibration update computes ordinary chain-rule gradients through all trainable LoRA modules, including both A and B factors, and corrects drift from the approximation.

The **Exact calibration interval** setting controls N (`--residual_path_exact_every`); the default is 8 optimizer updates and the minimum is 2. The interval counts optimizer updates after gradient accumulation, not micro-batches. With gradient accumulation, all micro-batches contributing to a scheduled calibration update use the exact path.

This design removes transformer-block backward/replay on the approximate updates, but it is not completely derivative-free: the final readout still uses a small input derivative, and the scheduled calibration updates still use exact block replay. Identity-path credit ignores the Jacobians of intervening attention and MLP branches, so its per-block signals are approximations rather than exact global gradients. It may trade convergence quality for lower compute; only a real Krea 2 training run can establish whether it is faster at a matched validation quality.

The two original Residual-path hybrid choices modify only the LoRA B factors on `attn.wo` and `mlp.down` during approximate updates. Exact calibration updates refresh all configured block LoRA factors. The additional **Residual-path hybrid H2D — selective A** choice keeps that B update on every approximate step and also updates the corresponding A factors on every fourth approximate update. Exact calibration steps do not count toward the four-update A cadence. On a selective-A update, the original full-width inputs to the two target projections in each block are copied asynchronously into pinned host RAM, then transferred back one projection at a time for the A-gradient calculation. This avoids retaining all 28 blocks' full-width inputs in VRAM, but adds host-memory use and D2H/H2D transfer traffic on those updates; actual overhead depends on token count, PCIe bandwidth, and resolution.

Use standard LoRA, batch size 1, and a missing-gradient-aware optimizer. Muon and AdamW are supported; PoLoRA and LoRA-TSD remain rejected because most approximate updates still omit A-factor gradients and their paired-factor update rules require both A and B gradients together. Blockwise head warm-up and LoRA activation ramp do not apply. The same unsupported features and loss restrictions as the other block-local methods apply.

### Residual-path hybrid H2D

Choose **Residual-path hybrid H2D (experimental)** or pass:

```text
--family_option training_method=residual_path_hybrid_h2d
```

This selects the same residual-path hybrid algorithm with the H2D-only offloader described above. H2D changes only how frozen base weights are staged between CPU and GPU; it does not change the credit estimate, selected LoRA projections, exact-calibration cadence, or training objective. If the selected frozen-weight layout cannot be streamed safely, the offloader can fall back to classic swapping.

### Residual-path hybrid H2D — selective A

Choose **Residual-path hybrid H2D — selective A (experimental)** or pass:

```text
--family_option training_method=residual_path_hybrid_h2d_selective_a
```

This is the H2D hybrid with one additional approximate-update rule:

- Every approximate update calculates and applies the output-projection B gradients for `attn.wo` and `mlp.down` in every main block.
- Every fourth approximate update also calculates and applies the matching A-factor gradients. Exact calibration updates do not count toward this four-update cadence.
- Every scheduled exact global-loss calibration update continues to refresh both factors for all configured block LoRAs.

For an A update, Fizgig caches the full-width inputs to those output projections in pinned host RAM during the forward. After the readout credit is available, each input is brought back to the GPU one projection at a time to form `dL/dA = (g B)^T x`. The activation copies are asynchronous where CUDA/pinned memory is available and are synchronized once before the A-gradient calculations. This keeps the extra full-width activations out of persistent VRAM, but increases CPU RAM use and adds D2H/H2D traffic every fourth approximate update. It is expected to be slower on those updates than B-only hybrid; the trade-off is additional adapter capacity, not a guaranteed convergence improvement.

## Requirements and current limitations

- Krea 2 only; standard LoRA network type only.
- Batch size must be 1. Gradient accumulation can be used for a larger effective batch.
- Selective global replay does not support PoLoRA; use Muon, AdamW, or another optimizer that safely skips missing gradients. Residual-path hybrid supports Muon and AdamW; it rejects PoLoRA and LoRA-TSD because most approximate updates omit A-factor gradients, which paired-factor optimizers do not support.
- These methods train all 28 main transformer blocks together. Custom block subsets are not supported.
- Text-fusion and input/timestep/output LoRA modules are left frozen in this method.
- Fine-tuning, LoKR, slider training, DOP, masked/prior loss, rotated INT8 backward, training attention maps, and shared/helper training are rejected.
- `torch.compile` and gradient checkpointing are disabled for these manual block paths. The three non-H2D experimental choices use classic block swapping; the four H2D-labelled choices explicitly request H2D-only offloading and log a fallback to classic swapping if the frozen weights are not streamable.
- The CPU must have enough memory for saved block inputs. At larger image-token counts, keeping all 28 inputs can consume multiple GB of system RAM. This trades activation VRAM for CPU memory and repeated block execution; it is not guaranteed to be faster.
- CPU tests check exact block-replay gradients on toy DiTs, the residual-path hybrid's approximate B-factor updates across 28 toy blocks, and its H2D forward-only pass followed by exact replay. These are algorithmic checks, not a full 28-block production Krea 2 CUDA run; GPU stability, transfer overlap, speed, and matched-quality convergence still need to be benchmarked.

## Saving and resuming

The Blockwise flow training-state architecture tag is versioned separately from the earlier independent-local-loss prototype. A resumable run with head warm-up includes the ordinary `lora.safetensors` and `optimizer.pt`, plus:

- `block_local_heads.pt`
- `block_local_optimizer.pt`

Keep both additional files when resuming before or after the configured head warm-up. Runs with no head warm-up do not need these auxiliary files. Selective global replay stores its selector state (`active_blocks`, `scores` and `optimizer_steps`) in `training_state.json`; Residual-path hybrid stores its calibration interval, approximate-update LR multiplier, optimizer-update counter, and approximate-update counter there so the cadence and LR behavior survive resume. The selective-A H2D variant also stores its A-update interval (4). Older Residual-path hybrid v1/v2 states remain resumable; approximate-update counts are reconstructed from the optimizer count and exact-calibration interval when older state files lack that counter.

## How to evaluate it

Compare the ordinary global Krea 2 validation loss and same-seed previews over training. Do not use the temporary-head calibration loss as a quality metric. Use the same dataset split, seed, rank, resolution, precision, timestep settings, optimizer and training budget for a standard-backprop baseline and each experimental mode. For Residual-path hybrid, compare time to reach a matched validation loss, not just seconds per optimizer update. Also compare preview quality, LoRA file size, peak VRAM, CPU RAM, and wall-clock time.


## Residual-path hybrid approximate-update LR

Residual-path hybrid exposes an independent **Approximate update LR multiplier** (default `2.0`). On the optimizer-step boundary, the currently scheduled LR is multiplied by this value only when the next update uses residual identity-path credit. The exact global-loss calibration update remains at the unmodified scheduled LR. The multiplier composes with the trainer's existing per-step LR adjustments and scheduler, and the live LR plot/postfix reports the effective rate for the upcoming update. It is available in both original Residual-path hybrid choices and the selective-A H2D variant.
