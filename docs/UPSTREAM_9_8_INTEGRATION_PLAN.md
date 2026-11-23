# Upstream Fizgig 9.8.x → Advanced Caption Workbench: selective integration

**Review snapshot:** 2026-11-23. Our baseline: `test/caption-workbench` at
`ebd390876e13d9b6239f8facd75b17eb09a3d4ef`; upstream
`shootthesound/Fizgig@master` at `43c88d8277291bece7f4d220568e3fe375093b50`.
GitHub compare at review time: **223 commits upstream-only**, **814 commits on our Workbench side** since the
shared ancestor `aa199708e6be3dc4703288c6901ed1f56bb91d65`. Counts are a snapshot, not the proposed merge size.

Our older `fizgig-advanced` branch was synced through upstream **9.7.6**. The current upstream includes
9.7.7–9.8.5 and additional work after the 9.8.5 notes. Do **not** run a blanket merge of upstream master into `test/caption-workbench`. The user selected
**direct, small, tested pushes to `test/caption-workbench`**, the current development source of truth.
The initial staging batch passed CI and was fast-forwarded into Workbench without a history rewrite.
Use small atomic commits with upstream attribution, check the live head before every push, and retain the
staging PR only as historical review context.

## Invariants we must preserve

1. Caption Workbench's Tags / Waterfall / Constellation UI, editing semantics, virtualization, undo/review
   safeguards, thumbnail caching, and *one Chromium-backed image UI*, with GPU-enabled WebView2 by default.
   Do not reintroduce a native Qt image renderer or a second state/sync layer.
2. Advanced's dataset image+caption change detection and next-epoch re-encoding, including manual changes,
   removed images, cache preflight and resumed runs. A change to a caption must not silently reuse old encodings.
3. Training extras: Krea 2 global replay, Text Fusion and calibration where selected, selective block check,
   per-folder weights/repeats, progress/steps-left, validation and best checkpoint, PoLoRA, training masks,
   manual Save Now and pause/resume cadence. Preserve existing control defaults and confirmation interactions.
4. Advanced's LoRA/optimizer/mask compatibility, LaMa removal, image-prep editor, two-step destructive
   controls, and source attribution. Avoid overlapping feature switches when the architecture already
   has an answer.

## Decision matrix (not an automatic import list)

| Upstream area | Decision | Integration and conflict notes |
| --- | --- | --- |
| Captioner refuses non-.safetensors files, #388, `56729e1` | **Integrated into Workbench** | Adapted in `web/captions.py` with stricter bounded header read; tests in `checks/test_captioner_header_guard.py`. Leaves Workbench caption APIs alone. |
| Video-only folders with a remembered Mask/Element boost tick, #402, `5337483` | **Integrated on staging branch** | Adapted `families/launch.py` and `form.py`, with upstream's regression cases. Important: disabled for *clips-only*, enabled when a trained subfolder contains photos. No mask renderer changes. |
| Non-safetensors model headers checked centrally, `9ac00cd` | **Integrated into Workbench** | Adapted the upstream shared header reader into existing model planners, quantization, MiniMax, and both loaders. No Advanced metadata writing replaced; added regression tests and CI. |
| Offline-first Anima / ACE-Step tokenizers and SDXL config, #397, `1437792` | **Integrated into Workbench** | Ported Anima, ACE-Step, SDXL, cached helper checks and updater, keeping Advanced SDXL timestep spread. Offline/partial-cache tests added to CI. Next-epoch data-change recache remains a separate lifecycle to test live. |
| Resume at new size with Cache disabled (#370), and stop request hung (#379 notes) | **Priority 1: verify then fix** | These hit reliability and run lifecycle. Advanced already changed resume and cache behavior; compare exact paths, do not overwrite wholesale. |
| MiniMax H3 / low-VRAM previews and Studio fixes; Krea 2 ROCm memory/TE release | **Partially integrated** | Krea 2 encoder releases weights to meta instead of copying to RAM, with file-backed inference guard and tests. Remaining AMD/ROCm and memory planner differences require hardware validation. |
| Radeon expandable segments / MIOpen launcher paths and override (#389) | **Priority 1 for AMD, isolate** | Preserve Windows/Linux separation and user overrides; test on Windows AMD rather than assuming path layout. |
| Studio save setup, primary-LoRA quick swap, queue and cross-tab isolation (#394/#395) | **Priority 2: selective backend integration** | Good performance/correctness improvements, but substantial shared state and `web/static/app.js` changes. Integrate API lifecycle first, UI second, preserve current Workbench/Studio panel layout. |
| Edit-model references retain their own shape at Reference MP, `43c88d8` | **Integrated into Workbench** | Adapted the sizing/reference-key implementation and un-greyed Studio Reference MP with upstream tests. |
| Multi-line captions and “Add as another line” (#300) | **Defer architecture review** | Can conflict with our tag-based editor, selection, AI recaption, tag ordering and next-epoch text re-encode. First define whether a training item has one caption string or per-line variants; ensure save/reload/undo tests cover both. |
| Per-folder numeric repeats from folder names (#300) | **Do not duplicate UI** | Advanced has explicit per-folder weights/repeats. May optionally import naming conventions into that existing model, but **never** create a competing repeat counter. |
| Caption jobs in run queue, pod dataset download | **Defer** | Useful but involves job scheduling, path security and selected-image behavior; design around our Workbench, not upstream's older Captions tab. |
| Studio without primary LoRA, automated save, early/last preview frames, responsive output panes | **Evaluate individually** | Keep if it solves a concrete gap and integrates cleanly; not an excuse to import a competing UI. |
| MiniMax H3 Image fine-tuning and Speed boost | **Speed boost integrated** | Added upstream H3 Image speed boost (Off by default; 1–4 half-size early steps) to Samples and Studio, with tests. Fine-tuning remains separately gated. |
| Frankenstein model-blending experiment | **Experimental integration in Workbench** | H3 Image + H3 Video live blend Studio mode with 16GB+ upstream behavior and experimental 12GB int8 block streaming; independent CPU checkpoint mixer for structurally compatible MOSS, Anima, Krea 2, SDXL/Illustrious. See [mixing guide](EXPERIMENTAL_MODEL_MIXING.md). Not cross-family, not training, no GPU live validation yet. |
| New native file picker | **Already represented** | Upstream itself credits Fizgig Advanced; avoid porting our old implementation back over our own. |
| Any native Qt image rendering alternative | **Reject** | Previous controlled Windows test found GPU-enabled WebView2 as smooth as the browser; one shared rendering architecture is the accepted decision. |
| Upstream release notes / branding / controls | **Curate, never wholesale replace** | Credit upstream PRs and original authors for copied code. Do not replace Advanced's names, default choices or its UX with upstream variants. |

## Integration procedure

1. **Inventory against exact SHAs.** For each candidate commit read upstream patch + our current implementation, not
   just its release headline. Tag as already present / missing / conflict / deliberate non-adoption.
2. **Backend first.** Port the minimum code, with upstream commit and contributor attribution, directly on Workbench.
   Where our behavior differs deliberately, preserve it and add a regression test explaining why.
3. **Keep control surfaces coherent.** Use existing settings and shared app state; introduce no unnecessary mode
   toggles, duplicate tab, dialog, cache, new renderer, or second queue for the same job.
4. **Test every boundary.** Lint/syntax and unit tests; Caption Workbench JS tests; model selection, header,
   dataset editing→recache at next epoch, masks with all supported training methods, save/pause/resume,
   global replay, validation and optimizer continuity.
5. **Hardware smoke test.** Windows GPU-enabled WebView2, browser, NVIDIA training; AMD on compatible hardware;
   confirm resource use when the relevant tab is inactive. CI alone does not prove these.
6. **Promote deliberately.** Each atomic commit is pushed to Workbench and checked with branch CI.
   A CI pass is not a substitute for Windows/GPU/AMD hardware smoke tests; fix regressions on Workbench.

## Separate branch divergence to handle deliberately

`wip-test` contains **five newer Text Fusion calibration commits** not in the Workbench branch (manual/automatic
target probing), affecting `families/train.py`, `krea2/direction_edit_v2.py` and their checks. They are *our*
branch's features, **not** upstream Fizgig changes. Reconcile them independently after the upstream safety batch;
do not mix them into a bulk upstream merge.

## Workbench integration progress and QA

- [x] Review initial fixes on staging, then fast-forward them into Workbench
- [x] Harden `captioner_problem` for GGUF / oversized headers and add tests (upstream #388)
- [x] Bring video-only Mask/Element boost launch behavior into line with the UI and add tests (upstream #402)
- [x] First batch passed GitHub Actions; subsequent integration commits have per-commit CI
- [x] Port upstream's identified model-header readers into the bounded shared reader; audit any new readers separately
- [x] Port offline tokenizers and partial-cache recovery with automated tests
- [ ] Live training next-epoch caption/image re-encode after cache changes
- [x] Change Krea 2 encoder release to avoid a CPU weight copy
- [ ] Review stop/resume/cache behavior against upstream without disturbing Advanced's existing fixes
- [x] Integrate Studio and Royale edit-reference shape and MP control
- [x] Add optional MiniMax H3 Image speed boost (off by default)
- [ ] Evaluate Studio primary LoRA swap, output save metadata, and cross-tab isolation: upstream's latest Studio queue is not present in the Workbench baseline, so direct cherry-picks are unsafe
- [ ] Windows/Linux GPU and live training smoke tests (unit CI alone is insufficient)

**References:** [upstream commits](https://github.com/shootthesound/Fizgig/commits/master/),
[upstream releases](https://github.com/shootthesound/Fizgig/tree/master/docs),
[Advanced Workbench](https://github.com/Murf-Code/Fizgig-Advanced/tree/test/caption-workbench).
