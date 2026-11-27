# Fizgig Advanced — shared research status

Last reconciled: 2026-11-27. Canonical ledger: [WIP/RESEARCH_STATUS.md](https://github.com/Murf-Code/Fizgig-Advanced/blob/WIP/RESEARCH_STATUS.md). Branch copies are snapshots; consult the canonical ledger and branch-specific records before planning experiments.

## How to use this ledger

- This is a cross-chat index, not a substitute for protocols, raw results or source inspection.
- Status labels distinguish **closed research**, **measured pilot**, **active**, **proposed**, **deferred** and **production compatibility pending**.
- Before starting an experiment, check existing branch records and prior negative results. Explain what is genuinely new and what is held fixed. Do not repeat failed hypotheses without a reason or a changed condition.
- Keep research branches isolated. Only port validated changes with explicit approval. Record exact commits, configuration, results, failure logs, limitations and next gates after each experiment. Never claim a test ran unless it did.
- Raw user images, checkpoint binaries, private paths and logs stay out of Git. Archive names identify evidence retained by the user, not files guaranteed to be in Git.
- Independent review (Astra/Grok) is encouraged for new mathematics; reported third-party tests are not locally reproduced results.

## 1. Selective Global Replay — closed Muon research; opt-in port

Owner: `test/selective-movement-forks` / `archive/selective-global-replay-7g1`, frozen at `165ccabcd5343453197f6f70a50189aee450f0ab`. Original production candidate port to WIP: `7040a58ae0abff813ab8034973c3b44a41e0dc01`.

Primary source: [Frozen closeout (7G.1)](https://github.com/Murf-Code/Fizgig-Advanced/blob/165ccabcd5343453197f6f70a50189aee450f0ab/docs/experiments/SELECTIVE_RESEARCH_CLOSEOUT.md). Earlier comparison protocols live on the frozen branch. Do not reopen that experiment series merely because an independent reviewer proposes similar measurements.

| Experiment | Evidence / disposition |
|---|---|
| 1–5 | Replay baselines, high-gradient/random selection, mask controls, LR and decay sweeps. Earlier numerical archives are incomplete in the closeout; protocols are not proof of a specific ranking. |
| 6C.2k-C/D | Local movement-matched interventions, native W8A8 vs weight-only INT8; six probes, seeds 42/43, steps 4/8/12. Adaptive low-gradient/movement picked the same block each time. Static block 13 had the highest mean target gain among the listed D arms in both evaluators. No robust adaptive selector established. |
| 6F.2 | Historical reversion, 38 endpoints per seed/evaluator. No individual block benefited from full reversion across both seeds and both evaluators; all-block reversion worsened loss in all four combinations. No permanent block ranking. |
| 6F.3 | Six probes, 168 block records, 96 panel forwards. Activation cosine/residual, actual Muon additive proposal alignment, first-order training utility and gradient RMS did not produce a consistent expected-sign predictor across seeds. Historical reversion is not instantaneous attenuation; do not claim every conceivable shuffled-control design was tested. |
| 7C/7D | Rotating attenuation and movement-matched uniform/compensated arms. 26/26 local matches; exact solver added time, no compelling EMA benefit. Not a production solver. |
| 7G | Fixed alternating 14/28 suspension degraded EMA; compensation partially recovered deficit. User's qualitative preference: full, existing, compensated balanced, balanced. |
| 7G.1 | Legacy dynamic selector plus bounded count compensation; best mean raw and EMA among final arms, raw improved both seeds, EMA improved seed 42 and slightly regressed vs full on 43. |

Final comparison: Krea2 native INT8, 13 train / 2 development holdouts, rank/alpha 32, 65 updates, two seeds, Muon native LR .00045. Mean EMA loss: full .180477607; existing .180544726; balanced .181843570; compensated balanced .181522369; dynamic compensated .180315870. Dynamic compensated mean step time 3115.95 ms vs full 3154.00 ms (1.21% faster), with 82.802% of adapter-gradient work. Not a significant throughput or generalization claim. Repeated holdout is a development panel.

Keep the legacy policy and optional learning-term multiplier `min(1.5, sqrt(28 / active_count))`; no compensation on full-block updates. Preserve native optimizer state and decay semantics. No permanent block importance or exact movement solver in production. Full GPU/resume and broader compatibility gates remain, as recorded in [compensation notes](https://github.com/Murf-Code/Fizgig-Advanced/blob/test/selective-optimizer-support/docs/SELECTIVE_COMPENSATION.md). The provisional 1.8x AdamW-to-Muon reference conversion is not an established universal conversion or production requirement.

## 2. Selective multi-optimizer compatibility — measured pilot

Owner: `test/selective-optimizer-support`. Experiment implementation `5d198f24e3097c4956429712f16ccf7175792ba1`; [queue and audit source](https://github.com/Murf-Code/Fizgig-Advanced/blob/5d198f24e3097c4956429712f16ccf7175792ba1/src/fizgig/web/optimizer_match.py). Evidence: user archive `ademamix8bit-full.zip`, embedded `optimizer-match-report.json`, fixed baseline and epoch samples; report re-read for this ledger.

Fourteen of sixteen attempts completed (910 updates, five epochs, seed 42, 13 training images / two holdouts, rank/alpha 32, native INT8). All seven completed Selective pairs have matching input hashes, passed recorded local-state and reactivation compatibility checks. No external compensation.

| Optimizer | Mean active / 28 | Path length retained | Next action |
|---|---:|---:|---|
| AdEMAMix 8bit | 16.35 | 79.3% | Verify native decoupled decay then test moderate compensation. |
| Paged AdEMAMix | 17.37 | 81.4% | Same approach, verify actual native backend. |
| Lion 8bit | 23.00 | 89.8% | Calibrate unrestricted LR first; weak visual learning. |
| PoLoRA | 18.54 | 84.2% | Strong visual learning; test modest native LR scaling separately. |
| LoRA-TSD | 13.51 | 65.6% | Strong visual changes despite reduced movement; avoid forcing path-length parity or raising its cap. |
| Automagic v3 | 26.15 | 96.7% | Underlearned; calibrate native starting rate and record real controller rates. |
| Automagic v4 | 24.32 | 97.5% | Same; preserve native adaptation. |
| Prodigy | — | — | Both attempts failed at startup; explicitly excluded for now. |

Path length is the sum of effective weight update Frobenius norms, not net displacement or learning quality. Compatibility status is recipe-specific; it is not a quality or production approval. Subjective visual review against untrained and full controls: AdEMAMix/PoLoRA/TSD learned clearly; Lion Selective underlearned; Automagic samples nearly unchanged. User observed Automagic LR stayed at or below about .0000025 until epoch five; this is user observation, not independently reconstructed internal-rate telemetry. A .00025 starting rate is an experimental candidate only. Holdout is too small for statistical claims. Original artist training images not supplied for independent fidelity assessment. Research instrumentation adds CPU transfer/hash/geometry overhead; optimizer-only timing is not end-to-end speed.

## 3. Optimizer LR calibration and SOAP — active, separate from selection

Owner: `test/selective-optimizer-support`; [harness](https://github.com/Murf-Code/Fizgig-Advanced/blob/test/selective-optimizer-support/src/fizgig/web/optimizer_match.py). Current working native baselines: AdamW .00025, Muon .00045, PoLoRA .02; not universal effective-learning conversions. Compare effective movement using exact scaled BA differences, repeat controls and visual/validation outcomes.

- SOAP memory research: chunked/offloaded/compact experimental variants, with compact variance transport replacing unsafe variance reset. Full covariance estimates are recipe-specific, not GPU measurements.
- Pilot `optimizer-match-report(1).json` re-read: AdamW, AdamW-repeat, SOAP512, compact32 all completed 13 updates. AdamW repeat gate failed, so no automatic LR recommendation. Both SOAP arms started near AdamW movement after initialization but movement declined late; compact showed a sharper post-refresh decline. No validated quality equivalence.
- Five-epoch SOAP sweep implemented at `a09034bf26d0d11f4b0d606b846635127678a45f`: SOAP512 .00025/.0004, compact32 .00025/.0005, fixed samples, 65 updates per arm. Results not available in this audit; review incoming reports before modifying the recipe.
- Do not conflate memory feasibility, calibration, optimizer-state safety and quality. Keep Muon/PoLoRA baselines intact while testing SOAP. Full CUDA integration and resume gates remain.

## 4. Gauge-invariant selection — proposed narrow experiment only

Owner: planned on `test/selective-optimizer-support`, not implemented or approved for production at this snapshot. Independent Grok and Astra reviews in the project identify a coordinate-dependence weakness of the legacy factor-space cosine. Grok's original rank-r historical displacement derivation was incorrect in general; do not implement it.

Astra's corrected proposal: with W=sBA and factor gradients g_A=s B^T G, g_B=s G A^T, identify Z=P_B G+G P_A-P_B G P_A. Historical displacement D=s(BA-B0A0) may have rank 2r. For a block use direct-sum cosine of -Z and D, and observability rho = sqrt(sum ||P(D)||_F^2 / sum ||D||_F^2), where P is the current differential-space projector. Thin bases and 2r cores avoid dense matrices. Rank-deficient factors, missing/tiny signals, consistency of reconstructed overlap, finite-precision gauge conditioning, and zero displacement require explicit unknown handling; zero is not equivalent to unknown under the legacy policy. Astra reported 100 synthetic dense-reference cases, max discrepancy 6.3e-13; not independently reproduced here.

**Next gate:** independently reproduce dense-reference, gauge, rank-deficiency and cancellation tests; record latency/scratch. Compare the new score and observability with legacy in matched PoLoRA/LoRA-TSD runs, same probe schedule, no compensation, same data/seeds/samples. Preserve legacy as control. Do not repeat 6F proposal/utility/reversion or 7C exact-movement experiments unless new evidence justifies it. The new geometry is a cleaner retrospective metric, not proof of predictive utility. No claim of superiority, GPU implementation or new code from this ledger.

## 5. Frozen-base compression (LittleBit) — active, independent

Owner: `research/krea2-littlebit`. [README](https://github.com/Murf-Code/Fizgig-Advanced/blob/research/krea2-littlebit/scripts/research/krea2_littlebit/README.md), [Round 2 protocol](https://github.com/Murf-Code/Fizgig-Advanced/blob/b1fa54a56945353a5a30b26da1653e6bc23ddcc0/scripts/research/krea2_littlebit/PEER_REVIEW_V4.md). Keep native LoRA export and same-stock-base evaluation. Earlier external rank-512 transfer failed on synthetic projection evidence; not a diffusion result. Use actual activation/cotangent screening, held-out evaluation, matched stock reference, and full runtime memory/speed accounting. 96% retention is an aspiration, not a gate; suppress ratios when stock gain unstable. Do not vendor separately licensed pilot material or claim legal clearance. No full-model GPU result verified in this audit; check current branch before changing queue.

## 6. Matrix geometry and optimizer memory — active, separate

Owners: `research/lora-matrix-geometry` (geometry/resolution) and `research/polora-kernel-efficiency` (kernel/state/memory).
Sources: [geometry experiments](https://github.com/Murf-Code/Fizgig-Advanced/blob/research/lora-matrix-geometry/docs/experiments/LORA_MATRIX_GEOMETRY.md), [matrix register](https://github.com/Murf-Code/Fizgig-Advanced/blob/research/polora-kernel-efficiency/docs/experiments/MATRIX_GEOMETRY_REGISTER.md), [PoLoRA kernel research](https://github.com/Murf-Code/Fizgig-Advanced/blob/research/polora-kernel-efficiency/docs/experiments/POLORA_KERNEL_EFFICIENCY.md).
- Muon-ME isolated rank64 CUDA scratch and timing improvement measured; subsequent six closed-loop runs finished. Rank64/16 swap fit but severe shared-memory pressure and slowdown. Do not infer a quality win or repeat completed sweep.
- PoLoRA-ME chunking/state-view experiments, TSD-ME compact clipping, budgeting, activation offload, quantized momentum, rank allocation, initialization and rotation-compatible EMA are tracked in the matrix register with separate gates. CPU prototypes are not integrated CUDA results.
- Do not blend geometry, low-bit frozen-base compression, Selective policy and optimizer LR calibration changes into one uncontrolled experiment.

## 7. Shared priorities and update protocol

1. Preserve the closed Muon result and archived evidence. Production compatibility tests are separate from exploratory training.
2. Complete pending SOAP calibration and native optimizer compatibility/compensation in isolated arms; exclude Prodigy.
3. Pursue only the narrow new gauge-invariant score experiment if its correctness and cost gates pass. Keep existing selection behavior as default.
4. Continue independent LittleBit and matrix-memory work under their own protocols. Check latest branch register before queueing.
5. Add new entries here with date, branch/commit, evidence pointer, measured result, limitations, decision and next gate. Mark superseded or negative results instead of deleting them. Keep canonical WIP ledger current and synchronize active branches; historical archive branches remain frozen.

Shared caveat: small datasets, repeated holdouts, few seeds and subjective fixed samples support feasibility and hypothesis testing, not general claims. Review the original source before citing exact numbers. Credit original Fizgig, upstream contributors, independent reviewers and user GPU testing.
