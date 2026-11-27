# Selective Global Replay research — frozen closeout (7G.1)

Status: experimental development complete. Preserve this branch and its experiment protocols. Production candidate was ported separately to WIP in 7040a58ae0abff813ab8034973c3b44a41e0dc01. No research harness, importance predictor, exact solver or provisional optimizer conversion is required by that port. No additional exploratory arms planned.

## Question and method

Global replay preserves the full loss and propagates block-input gradients through all transformer blocks. Existing Selective additionally suppresses adapter-factor gradients for blocks selected by the legacy displacement/negative-gradient cosine check, with all-block warmup and reactivation probes every eight updates. Suspension is not removal of a block from forward or backward replay. We investigated whether a better selection rule or learning allocation could preserve learning while reducing computation. Separate matrix-geometry and LittleBit base-compression projects are outside this closeout.

The final experiment retains the legacy policy and multiplies only Muon's post-orthogonalization additive learning term by min(1.5, sqrt(28 / active_count)). The count-based scaling is nominal, not exact effective-weight motion matching. Full-block updates have multiplier one. Skipped factors are not decayed and their momentum buffers are retained. Active decay, scheduler and native optimizer rate are unchanged relative to the existing policy; exposure to decay differs from full training. No extra forwards or per-step matrix decomposition are needed.

## Research progression and negative results

Earlier comparison protocols: GLOBAL_REPLAY_COMPARISON.md, GLOBAL_REPLAY_EXPERIMENT_5A.md, GLOBAL_REPLAY_EXPERIMENT_5B.md and GLOBAL_REPLAY_EXPERIMENT_5C.md. These investigated baseline replay, high-gradient and random selection, causal mask controls, learning-rate sweeps, and decay controls. The archived early reports are not all present in the final review; do not treat their protocol descriptions or remembered rankings as reverified quantitative results. This distinction also applies to synthetic momentum/allocation trials. Existing Selective in 7G is the legacy displacement policy, not an earlier adaptive-gradient arm.

* 6C.2k-C/D (SELECTIVE_DUAL_EVALUATION.md, SELECTIVE_MULTI_CHECKPOINT.md): local same-parent intervention comparisons on native W8A8 versus weight-only INT8. Six probes across seeds 42/43 and steps 4/8/12 in D had exact registered-state rollback and movement matching. Mean target gain (parent minus candidate MSE, times 1e-6) for uniform, static block 26, static block 13 and adaptive low-gradient/movement: native 13.31, -2.73, 27.39, -9.18; weight-only 0.90, 5.49, 8.65, 4.26. Adaptive low-gradient and movement chose the same block at all six probes. Near-uniform Muon movement made the movement/gradient ratio largely inverse gradient magnitude. Weight-only prediction displacement was about 92% smaller; it is a different numerical instrument, not a ground truth. Decay-only showed extremely small effective motion yet substantial native prediction change. No robust selector or preservation conclusion.
* 6F.2 (SELECTIVE_HISTORICAL_REVERSION.md): 38 historical reversion endpoints per seed per evaluator, source after step 3 and parent after step 11, 12/12 matched targeted/uniform controls, exact rollback. No block had a beneficial full single-block reversion across both seeds and both evaluators. Native cross-seed rank correlation about -0.037, weight-only -0.137; all-block early reversion worsened target loss across all four seed/evaluator combinations. Individual effects did not add linearly; nonadditivity is not proof of coadaptation. Reject a permanent block ranking inferred from this small panel.
* 6F.3 (SELECTIVE_PREDICTOR_VERIFICATION.md): six probes, 168 block records, 96 panel forwards. Tested activation input/output cosine, residual norm, actual additive Muon update alignment, first-order training utility and gradient RMS. No consistent expected-sign predictor across both seeds. Activation rankings were very stable, largely architectural. First-order training utility is not validation utility. Freeze learned importance heuristics rather than ship them.
* 7C/7D (SELECTIVE_JOINT_ATTENUATION.md): four 13-update seed-42 arms (full, rotating half-strength, movement-matched uniform, compensated rotating). 26/26 local matches passed, max relative error about 4.22e-8. Mean scales uniform .790521 and compensated 1.264908, close to the nominal RMS approximation. Mean final EMA losses full .182343828, rotate .182550908, uniform .182520797, compensated .182526344. Raw compensated was marginally lower than full; EMA was not. The QR solver proved local matching but added substantial time; no production solver or continuous attenuation. Local matching does not match whole trajectories.
* 7G (SELECTIVE_FINAL_7G.md): four arms x two seeds, 65 updates each. Unrestricted, legacy existing, alternating 14/28 true suspension, and the same alternating schedule with sqrt(2) learning-term scaling. Matched initialization and first balanced/compensated native direction/momentum audits passed both seeds. Fixed suspension degraded EMA; compensation recovered part of the deficit, not all. User's corrected visual preference was full, existing, compensated balanced, then balanced. This is subjective, not blinded scoring.
* 7G.1 (SELECTIVE_DYNAMIC_7G1.md): two new runs with legacy selection plus bounded count compensation; archived full/existing controls reused. All 130 updates completed and initialization hashes match controls. The actual run used native LR .00045 before the later AdamW-reference conversion commit. The latter is unverified by these runs.

## Final design and exact results

Krea 2 raw BF16 base loaded native W8A8 INT8, RTX 4070 Ti 12 GB, rank/alpha 32/32, 224 LoRA modules/448 factors, 13 training images and two fixed held-out images (image (144).png and image (182).png), 496 nominal resolution, batch/accumulation one, five epochs/65 optimizer updates, seed 42 or 43, validation seed 42, four noise levels, EMA .98/ramp 10 over all factors, BatchedMuon native LR .00045 with match_rms_adamw, momentum .95, Nesterov, five Newton-Schulz steps, weight decay .01, 22 blocks' worth swapped (24 streamed through 2-slot H2D ring). Four fixed prompts (two triggered, watercolor without trigger, unrelated landscape), seeds 1234–1237, 512x512, 20 preview steps with Turbo LoRA strength .7, baseline plus each epoch. Same logged recipe and initial factors; not a content hash proof of unchanged dataset or model.

| Arm | Seed | Raw validation | EMA validation | Step ms | Mean active blocks | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| Full | 42 | .180731854 | .180683892 | 3198.4 | 28 | 10.57 |
| Existing | 42 | .180668998 | .180810161 | 3172.2 | 22.1846 | 10.49 |
| Balanced | 42 | .181650803 | .181811962 | 3156.3 | 14 | 10.68 |
| Balanced compensated | 42 | .181390872 | .181502878 | 3130.8 | 14 | 10.36 |
| Dynamic compensated | 42 | .180236207 | .180318888 | 3109.4 | 22.6 | 10.39 |
| Full | 43 | .180126413 | .180271322 | 3109.6 | 28 | 10.43 |
| Existing | 43 | .180096732 | .180279290 | 3150.9 | 26.0615 | 10.36 |
| Balanced | 43 | .181634193 | .181875179 | 3096.2 | 14 | 10.14 |
| Balanced compensated | 43 | .181278800 | .181541860 | 3092.8 | 14 | 10.24 |
| Dynamic compensated | 43 | .180050676 | .180312851 | 3122.5 | 23.7692 | 10.29 |

| Mean of paired seeds | Raw | EMA | Step ms | Adapter-gradient fraction |
|---|---:|---:|---:|---:|
| Full | .180429134 | .180477607 | 3154.00 | 100% |
| Existing | .180382865 | .180544726 | 3161.55 | 86.154% |
| Balanced | .181642498 | .181843570 | 3126.25 | 50% |
| Balanced compensated | .181334836 | .181522369 | 3111.80 | 50% |
| Dynamic compensated | .180143442 | .180315870 | 3115.95 | 82.802% |

Mean EMA over epochs 1–5, respectively:
- Full: .182577537, .182107368, .181369103, .181020204, .180477607
- Existing: .182593856, .182188056, .181648752, .181182648, .180544726
- Balanced: .182570680, .182507403, .182313481, .182059021, .181843570
- Balanced compensated: .182590714, .182359806, .182113275, .181740385, .181522369
- Dynamic compensated: .182524759, .182109823, .181600118, .180914380, .180315870

Dynamic compensation: seed 42 mean 1.125527, max 1.283378, 50/65 compensated, zero cap hits; seed 43 mean 1.097747, max 1.5, 36/65 compensated, one cap hit. All full probes remain at one. Lowest mean raw and EMA among these arms; raw lower than both existing/full on both seeds, EMA wins seed 42 and marginally loses seed 43. Mean EMA difference vs existing -0.000228856 (~0.127%); vs full -0.000161737 (~0.090%). 17.2% less adapter-gradient work than full, but only 1.21% mean step-time savings vs full (1.44% vs existing). Not a meaningful throughput breakthrough and below the predeclared 10% speed goal. Dynamic policy changes its later masks, so this is a policy-level effect rather than a fixed-mask rate comparison. The two dynamic runs took 1965.91 seconds of summed run wall time. All ten final arms finished; 650 optimizer updates and 240 previews. The two final ZIPs pass CRC verification and contain 192 and 48 PNG previews respectively. Both final reports were re-read directly for this closeout.

## Decision

Retain full and existing as supported baselines. Port dynamic compensation only as an opt-in WIP feature for Krea 2 + Selective + built-in Muon; no default changes. No AdamW compensation claim, no automatic 1.8x conversion in production. Archive fixed balanced suspension, continuous attenuation, exact QR matching, importance predictors and alternative momentum rules. The current evidence is one small dataset, one model/precision/optimizer, two seeds and two held-out development images. Repeated use of that same panel makes it a development set, not an independent test. No statistical significance claim or generalization to other models/ranks. Qualitative previews are not a blinded preservation/likeness benchmark. Corrected user preference is recorded for 7G; no formal preference ranking was provided for 7G.1. Report these limitations in any paper. No need for another exploratory training queue. Only production smoke tests (including pause/resume) and a research manuscript remain.

## Reproducibility manifest

Keep the original user-side archives and reports together (do not delete or overwrite).
- 6C: global-replay-report(8).json, global-replay-results(7).zip, global-replay-comparison(8).log
- 6C.2k-D: global-replay-report(9).json, global-replay-results(8).zip, global-replay-comparison(9).log
- 6F.2: global-replay-report(10).json, global-replay-results(9).zip, global-replay-comparison(10).log
- 6F.3: global-replay-report(20261126-210854).json, global-replay-results(10).zip, global-replay-comparison(20261126-210854).log (the earlier 205044 snapshot is incomplete)
- 7C/7D: global-replay-report(20261126-213310).json, global-replay-results(20261126-213310).zip, global-replay-comparison(20261126-213310).log
- 7G: global-replay-report(20261127-045922).json, global-replay-results(20261127-045948).zip, global-replay-comparison(20261127-045921).log (code f8b0f7d06b9a542e25ff6168c68e438e81b8ecd8)
- 7G.1: global-replay-report(20261127-054958).json, global-replay-results(20261127-055006).zip, global-replay-comparison(20261127-054958).log (native LR recipe c29ceed1ee1544384813c56cb8e3feab35032cf8)
- 7G.1 later reference-rate code 6d0b03bff07876b01981b5cfcfe5ddcff1c54361: provisional mapping .00025 -> .00045, no completed conversion run.

Raw image archives remain private/local, not checked into GitHub. This is a development closeout, not a finished peer-reviewed paper. Early missing reports should be recovered before a full historical paper is written. Credit the original Fizgig project and upstream contributions, and identify AI-assisted code, analysis and review transparently in the manuscript.