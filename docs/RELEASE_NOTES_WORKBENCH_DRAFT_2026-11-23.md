# Fizgig Advanced — Caption Workbench release notes (DRAFT)

**Status:** Working draft, not a released build or production sign-off.  
**Integration branch:** [`test/caption-workbench`](https://github.com/Murf-Code/Fizgig-Advanced/tree/test/caption-workbench)  
**Compiled:** 2026-11-23. Recheck against the branch HEAD, CI, and live hardware before publication.  
**Scope:** Changes made on Advanced/Workbench since the `wip-test` base, including selective upstream Fizgig 9.8.x adaptations. This document is curated, not an assertion that *every* upstream feature was imported.

## Highlights — features to describe in the public release

### Caption Workbench / dataset management
- **Overhauled Captions tab:** a three-pane Tags workspace with an image rail, center preview/editor, and separate **Image Tags** and **All Tags** panels. Panels resize and the layout expands to available window space.
- **Three linked ways to explore:** Tags, image-frequency **Waterfall**, and interactive tag-co-occurrence **Constellation**. Image selection, range selection, and active-image handoff are shared between views.
- **TagBench-inspired tag workflows:** autocomplete, immediate text search, Boolean AND/OR/NOT filtering, double-click text editing, tag reordering and sorts, context-menu operations, customizable colors, selected-image and multi-image changes, keyboard navigation, and tag/caption exports.
- **Safer editing:** automatic saves for supported edits, grouped session undo/redo, review locks across covered writers, confirmation before destructive image/tag operations, dataset removal/restore journal, and explicit refresh.
- **Duplicates and bulk captions:** duplicate review/merge, bulk tag operations, selected-image AI captioning using existing local captioners, user-confirmed overwrite/append workflows, and ZIP / image-plus-caption imports using the existing uploader.
- **Image editor integrations:** crop and object removal in the compact editor, with the existing local **LaMa** removal backend preserved rather than importing TagBench's remover. Image Prep's watermark editor retains its glass-overlay workflow.

### Live training / LoRA improvements
- **Dataset changes during training:** supported caption and image edits are detected, queued, and re-encoded at epoch boundaries. Cache writes are atomic; removed files and failed refreshes are guarded against stale training. This is still awaiting live GPU acceptance tests.
- **Krea 2:** Standard and **Global replay**, with opt-in selective block checking, and the existing separate **Text-fusion calibration** system. Global replay includes PoLoRA and masked-training compatibility work.
- **Training controls and checkpoints:** Save Now, improved Pause/Stop confirmation, resume step/progress and compatible cache reuse, optional loss-based best checkpoint, image validation and loss views, plus clearer save/restore states.
- **More training configuration:** per-subfolder weights and repeats, validation image groups/hold-outs, revised optimizer and Adaptive LR handling, training method visibility/compatibility, and mask-related safeguards. Verify model-specific compatibility before promising universal support.
- **Model/conditioning safeguards:** offline-first cached tokenizers, faster Krea 2 text-encoder release, safe model-header reads, and robust error handling for unsupported captioner models.

### Studio / model experimentation
- **Faster primary-LoRA switching:** selectively adapted from upstream **#394**, with failure rollback, load-session tracking and regression coverage. Do not describe this as an imported upstream queue: Advanced's queue architecture is different.
- **Saved render setups:** save the setup associated with each rendered image, not a later state of the controls; retain PNG setup metadata. Multi-tab ownership and stale-tab guardrails were adapted from upstream **#395**.
- **Edit-model references:** preserve each reference image's own Reference MP / aspect sizing; Studio Reference MP control for supported edit models.
- **MiniMax H3 Image speed boost:** optional half-resolution early render steps; **off by default**.
- **Experimental Frankenstein Studio:** H3 model blending, experimental low-VRAM streaming planning, and compatible-family checkpoint mixing. Experimental model mixing is **not** cross-family compatibility or a promise that 12 GB hardware is verified; CPU planner checks do not replace live GPU proof.

### UI, responsiveness and memory
- **One GPU-accelerated Chromium image UI:** the superseded native Qt image-renderer experiment was removed. WebView2 runs with GPU acceleration on by default; GPU disable remains only a troubleshooting preference.
- **Windows acceptance:** on the user's Windows hardware, all six interactive GPU-on WebView2 checks were reported smooth: Caption Workbench, tag search/edit, Waterfall/Constellation, image editor, Image Prep/Samples, resize/fullscreen. This is qualitative testing, not measured FPS or a cross-platform guarantee.
- **Reduced UI work while idle:** avoid Constellation layout when not in Constellation, lazy/deferred thumbnail and gallery loading, bounded All Tags and Waterfall nodes, cached thumbnail decoding, and more efficient tag search.
- **Constellation benchmark:** one synthetic 1,200-caption graph-construction workload fell from about 192 ms to 15 ms with identical edges. This is not an overall application or CPU-utilization benchmark.
- **UX polish:** responsive panels, more compact editing controls, revised color/theme consistency, confirmation states, fullscreen scaling, and improvements to scrolling, selection and state preservation.

## Fixes and safety improvements
- Prevent incorrect selected-only AI caption operations from silently expanding to caption the whole dataset.
- Roll back failed image/caption/mask file moves and preserve mask subfolder paths; fail closed on corrupted removal journals.
- Preserve review locks through supported legacy caption and duplicate-merge operations.
- Guard against stale cross-folder caption loads, stale Studio browser tabs, and cross-tab destructive actions.
- Protect dataset sync from mask sidecars being treated as training pictures and from superseded Image Prep originals being admitted.
- Correct Undo/Redo busy-state handling and cross-view selected-image handoff.
- Keep requested stop behavior across subprocess boundaries and training-stage handoffs; preserve Pause semantics. Live interruption/resume validation remains open.
- Reject GGUF/non-safetensors model and captioner inputs early, including bounded safetensors-header checks.
- Handle video-only training datasets without inaccessible remembered masked-training/element-boost settings.
- Improve AMD ROCm MIOpen database path fallbacks and an **opt-in** `FIZGIG_MIOPEN=0` workaround. No AMD GPU smoke-test claim.

## Upstream source and PR attribution

Fizgig is authored upstream by **[Peter Neill / shootthesound](https://github.com/shootthesound)**. This Advanced branch selectively ports and adapts upstream work; it does not simply copy the upstream Studio queue or replace Advanced's Caption Workbench.

| Original source | How it relates to Workbench |
| --- | --- |
| [Fizgig upstream master](https://github.com/shootthesound/Fizgig) and 9.8.x development | Source for selected compatibility, safety, Studio, MiniMax and AMD work. Audit actual source before final release. |
| [#388](https://github.com/shootthesound/Fizgig/pull/388) | Captioner safetensors-only guard; adapted with bounded header checks. |
| [#402](https://github.com/shootthesound/Fizgig/pull/402) | Video-only folder/mask training launch check. |
| [#397](https://github.com/shootthesound/Fizgig/pull/397) | Offline-first tokenizer and partial-cache recovery. |
| [#370](https://github.com/shootthesound/Fizgig/pull/370) | Changed-resolution resume/cache protection; already represented in Advanced, not separately duplicated. |
| [#389](https://github.com/shootthesound/Fizgig/pull/389) | AMD MIOpen workaround / database path handling. Credit contributors named on upstream PR, including @scryptio where applicable. |
| [#394](https://github.com/shootthesound/Fizgig/pull/394) | Primary-LoRA Studio hot swap; adapted to Advanced's single-worker architecture. |
| [#395](https://github.com/shootthesound/Fizgig/pull/395) | Multi-tab Studio ownership/isolation. |
| [#398](https://github.com/shootthesound/Fizgig/pull/398) and [#399](https://github.com/shootthesound/Fizgig/pull/399) by [@spaghettified26](https://github.com/spaghettified26) | Pillow safety and PNG metadata performance were reviewed during performance work. **Do not claim these exact PR changes shipped until integration is code-verified.** |
| [TagBench / TagGUI](https://github.com/) | UI/workflow inspiration for captioning, tag browsing, Waterfall and Constellation. **Replace this generic citation with the exact project's canonical repository and author before publication**. This is an inspiration acknowledgement, not a claim of wholesale code import. |

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
