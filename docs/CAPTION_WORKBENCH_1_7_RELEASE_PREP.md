# Fizgig Advanced 1.7 — Internal release preparation notes

**Internal checklist, not public release notes.** Prepared from the Workbench release-notes draft on 2026-11-23. Keep out of the reader-facing 1.7 announcement. No audit, GPU test, or attribution item below is automatically cleared by publication of the draft.

**Branch:** `test/caption-workbench`

**History review:** 654 Workbench-base-to-branch commits inventoried. Source features are documented in [expanded release notes](RELEASE_NOTES_WORKBENCH_DRAFT_2026-11-23.md).

**Artwork:** Four supplied screenshot WebP assets are embedded in `docs/RELEASE_NOTES_advanced-1.7.md` and `docs/RELEASE_NOTES_WORKBENCH_DRAFT_2026-11-23.md`. Confirm links render on the exact release head after changes.

### Explicitly not shipping / intentionally handled differently
- **Krea 2 Training Text Enhancer:** **not included**. The experimental training-conditioning integration was introduced in [57d7b18a](https://github.com/Murf-Code/Fizgig-Advanced/commit/57d7b18a969825e39de73c5c477a64740ee9de9d) and then deliberately **reverted** in [021af047](https://github.com/Murf-Code/Fizgig-Advanced/commit/021af047c0584e269bd948f527779d7f3ba888a8). Do not advertise it as a feature. It was derived from upstream [#393](https://github.com/shootthesound/Fizgig/pull/393) by @spaghettified26, which credits the original Krea2T-Enhancer by Capitan01R. Existing **Text-fusion calibration** is a different feature and remains.
- **Upstream Studio queue:** not copied wholesale because Advanced has a different state/queue architecture.
- **Native Qt dual-renderer path:** removed after successful GPU-on WebView2 evaluation; do not present as an optional user mode.
- **Feature parity:** not every TagBench behavior or every upstream Fizgig feature is integrated. User-requested deliberate omissions must remain documented rather than silently treated as bugs.

## Branch reconciliation: checked 2026-11-23
- `master`, `fizgig-advanced`, and `integration/upstream-9.8-review` are ancestors of Workbench; they have no additional branch commits to merge.
- `wip-test` has five commits absent from Workbench history. Existing [upstream integration plan](UPSTREAM_9_8_INTEGRATION_PLAN.md) records matching Text-fusion calibration implementation by content on inspected heads; **do not overwrite newer Workbench code merely to unify history**.
- `fix/krea2-calibration-encode-prompts` has two divergent commits; compare behavior with the corresponding later Workbench fixes before any import.
- `krea2-training-text-enhancer` has one divergent feature commit, **intentionally excluded** as above.
- Branch ancestry alone cannot prove feature parity with upstream master. The selective upstream feature matrix and acceptance gates live in [UPSTREAM_9_8_INTEGRATION_PLAN.md](UPSTREAM_9_8_INTEGRATION_PLAN.md).

## Release blockers / verification checklist

The following are **not** cleared merely by a passing focused unit suite:
1. Broad-suite residual failures and the original source compile blocker documented in [whole-branch audit](CAPTION_WORKBENCH_FULL_BRANCH_AUDIT_2026-11-22.md). Re-run after latest fixes; historical numbers are not the current result.
2. Full current-head CI verification, including Krea 2 calibration, masking, save/pause/resume, stage handoff and Studio metadata.
3. Disposable live CUDA run with edited captions, pixels, masks, deletions and restores across epoch re-encode boundaries and resumed checkpoints.
4. Actual AMD/ROCm MIOpen and memory tests; do not substitute CPU regression tests or simulated VRAM planner values.
5. Interactive desktop acceptance for responsive themes, imports, duplicate edits, undo, Waterfall/Constellation selection, two Studio tabs, primary hot swap, and PNG setup restoration.
6. Final code-level provenance pass for upstream PRs and borrowed implementation details, stable contributor links, and exact release version/date.

**Publication rule:** remove this DRAFT label only when branch reconciliation, CI, GPU checks, and credits are finished. Do not use unverified claims such as “all upstream features included” or “tested on all GPUs”.

