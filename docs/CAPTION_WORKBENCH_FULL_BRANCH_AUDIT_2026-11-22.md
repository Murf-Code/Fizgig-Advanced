# Fizgig Advanced — full branch audit
**Audit date:** 2026-11-22  
**Branch:** `test/caption-workbench`  
**Comparison:** `wip-test` (the original feature branch base)  
**Disposition:** **Not approved for production merge.** Continue working on this isolated branch.

## Scope and methodology

The original branch comparison returned **179 commits ahead, 0 behind** and **39 changed files**, within a tree of **926 entries (515 Python files and 4 JavaScript files)**. Additional commits below were made during this audit, so re-run the GitHub compare view for the latest totals. This is an audit of the *whole repository* using broad static checks and tests, plus targeted source review of all major modified paths—not a claim that every line of the 926 entries has been manually inspected.

Reviewed: commit history and changed-file inventory; trainer epoch loop; image/caption/reference/mask live re-encoding; bucket admission; caption and image queues; rollback/recovery; saved/removed image journals; persistent review locks; all Caption Studio edit routes and older caption writers; AI captioning/translation; duplicate merge; UI undo/redo, selection, tag parsing/analytics, import workflow; CI/test coverage; HTML static IDs; model-tokenizer integration; and native CPU LaMa object removal preservation.

Relevant GitHub Actions:
- Focused regression suite, expanded to include batch and regularisation training: [107 passed, 1 skipped](https://github.com/Murf-Code/Fizgig-Advanced/actions/runs/45022414621) on `e839aedf9`.
- Whole-source and broad test inventory: [Branch-wide audit run 45021848837](https://github.com/Murf-Code/Fizgig-Advanced/actions/runs/45021848837). The suite excluded one test module whose import is blocked by a baseline syntax defect; it reported **1309 passed, 18 failed, 52 skipped, 20 warnings, 2 subtests passed**. Three of the failures were new `Watch.removed_keys` assumptions and have since been repaired and added to focused CI. The other 15 have **not** been fully cleared; they must not be represented as passing.
- [Manual branch-wide audit workflow](../.github/workflows/branch-audit.yml): checks Python compilation, JavaScript syntax, duplicate HTML IDs, static undefined Python symbols, UI ID references and broad tests. The workflow is manual to avoid costly overlapping runners on every push. It reports historical failure counts but does not conceal them.

## Critical or high-priority defects found and repaired

| Severity | Area and issue | Audit repair and verification |
|---|---|---|
| **P0** | `server.remove_from_dataset()` moved an image and caption then looked for `masks/<stem>-masklabel.png` at the dataset root. Failed removal could leave partial filesystem moves with no journal. | Removal now preplans all files, records original mask subdirectories, rolls back on failure and restores the original relative path. Added masks-subfolder roundtrip and injected-move-failure regression tests. |
| **P0** | Explicit AI `only` selection was filtered to known folders; when all paths failed the filter, `only or None` could launch captioning on **the entire dataset**, including with overwrite enabled. | `selected_media()` now rejects empty, missing, out-of-dataset, symlink-escaped and unsupported selections, and never silently converts them to caption-all. Added regression tests. |
| **P1** | Mid-run new-image admission could treat `*-masklabel.png` as an image when captions were optional; likewise it could include originals superseded by Image Prep's `resized/` copies. | Use training's existing mask and superseded-stem filters in `loss_watch._new_image_sources()`. An initial CI test reproduced the mask bug; subsequent focused CI passed. |
| **P1** | Review locks could be bypassed through the older Write All Captions route, Problem Images caption save, or alternate caption-extension routes. | Add shared review-ledger locking and stem-based checks; preserve actual reviewed files. Review-locked saves are blocked atomically and covered by tests. |
| **P1** | Duplicate-tag merges checked review locks and snapshots before writing but did not hold the review lock through multi-keeper commits. | Lock validation and commit together, recheck source snapshots immediately before the first write. Preserve existing rollback behavior. |
| **P1** | A corrupt `.fizgig_caption_removals.json` could be silently treated as empty when recording a new removal, thereby replacing existing history. | Fail closed for journal mutation; preserve corrupt input. A failed removal rolls back changed files and the new test checks this. |
| **P1** | Restoring a sidecar mask enqueued `*-masklabel.png` as a training picture update. | Exclude mask filenames from restored-image notification; mask scanner handles mask updates separately. |
| **P1** | `studioTravel()` re-rendered while `history.busy` was still `true`, leaving Undo/Redo disabled after the first operation. | Clear busy state before reload and render. |
| **P1** | Overlapping `loadCaptions()` requests could repaint previous-folder content after switching datasets. | Generation-based response checks and guarded assignment; do not render superseded fetches. |
| **P1** | `Watch` batch policy referenced `removed_keys` unconditionally even on minimal/legacy `Watch` instances. | Added safe default for policy checks. All three related broad-suite failures are now covered by the focused batching/regularisation CI. |
| **P2** | “Export wildcard” exported a sorted list of unique tags, not TagBench's full-caption-per-image wildcard. | Implemented a complete caption-per-line export preserving dataset caption entries, linefolding and duplicates; added Node regression tests. |

All changes were pushed incrementally to `test/caption-workbench`; no production or `wip-test` merge was made.

## Broader regression results and existing technical debt

A full Python compile (`compileall src`) finds **one syntax error**, in `src/fizgig/krea2/direction_edit_v2.py:541`: an `IndentationError` after an `if` on line 540. This file is **unchanged from `wip-test`**. It blocks `checks/test_krea2_direction_edit_v2.py` collection and causes tests that parse the entire source tree to fail. It should be fixed in a separate baseline-repair change; do not falsely call the broad suite green.

The broad suite executed after excluding that collection blocker. Its 18 failures grouped as follows:

| Group | Failures | Current interpretation |
|---|---:|---|
| New loss-watch batch/regularisation assumption | 3 | **Repaired** and added to expanded CI |
| Tests indirectly blocked by the pre-existing malformed direction-edit module | 2 | Baseline syntax blocker |
| Old style-score / sample-pair interface expectations | 5 | Unchanged underlying `sample_pairs.py`; requires compatibility test/code reconciliation |
| Tag-shuffle expectations (variant count, wiring, form text) | 4 | Requires separate investigation; not cleared |
| Static close-up mask note, Krea2 global-replay settings, LoRA metadata, samples API fixture | 4 | Requires baseline comparison/reconciliation; not cleared |

These total 18. The **15 unresolved failures** are not necessarily all caused by this branch, and there is not enough evidence to certify that they were all already failing on `wip-test`.

The repository-wide undefined-name check (`ruff F821/F822/F823`) reports **eight warnings** spread over:
- `src/fizgig/krea2/model.py`: `BlockSwapConfig`
- `src/fizgig/krea2/utils.py`: `Qwen3VLConditioner` (two)
- `src/fizgig/minimax/refmod.py`: `dit`, `decoder`, `turbo`
- `src/fizgig/scripts/cache_text.py`: `text_embedder`
- `src/fizgig/scripts/minimax_cache_text.py`: `encoder`

The affected files have the **same Git blob SHAs on `wip-test`**, so these are pre-existing warnings and are not automatically regressions introduced by the caption branch; some might be runtime errors and need targeted verification. All UI JavaScript sources parse under Node. The primary HTML has **1,023 static IDs with zero duplicate IDs**. A JS-to-HTML scan flagged 16 unresolved literal ID lookups; several (including `studioLiveStatus`) are created dynamically; these are investigation leads, not confirmed missing controls.

## Still incomplete or not certified

1. **GPU active-training validation is outstanding.** The branch's CPU-only tests cannot certify real CUDA/ROCm VAE/TE recaches during a run, shape-changing crops, paired references, mask edits, new-image admission, mid-epoch changes, resumes, or failures during allocation/re-encoding. The code intentionally keeps the existing fixed-step bucket schedule. A changed image with a new aspect ratio can still be represented in its former bucket.
2. **Visual UX testing is outstanding.** No interactive running desktop session was available to check dark/light themes, panel proportions (1440/1024/768/390px), native editor mounting, keyboard focus, drag/drop, constellation pan/zoom, pointer selection and accessibility. A parsed JS file does not establish browser correctness.
3. **History is not universally unified.** Saved captions, Studio bulk edits, legacy caption actions, committed image crop/remove and image removals use one session-level history. Uncommitted mask brush strokes still use Image Prep's separate canvas history. Async AI-caption batch writes and all duplicate-merge operations are not guaranteed to be a single reversible Studio history transaction. History is session-only, not persisted across a restart.
4. **Review-lock protection is not a universal filesystem guard.** The reviewed ledger protects covered UI/API writers, not direct disk edits, external tools, or every editor that could mutate a dataset. Some long-running image editing calls can race with a review being marked after the initial check; this requires a concurrency test before sign-off.
5. **Restore journal may be partially reconciled if destination name collisions already exist.** That is correct for the legacy browser's partial-restore workflow, but the unified Undo action should represent incomplete restores explicitly, rather than blindly marking the entire operation reverted.
6. **TagBench parity is not complete.** Waterfall, constellation, search, sorting, autocomplete, duplicate review, drag/drop, model captioning and local LaMa reuse exist; full TagBench category automation, all brush history actions, tags-to-prose transformations and some preset behaviors still need explicit parity/acceptance tests.
7. **Uploader safety needs hands-on tests.** Importing into an existing named folder may replace same-name files after confirmation; ZIP, interrupted chunk recovery, nested paths, caption extension pairing and large datasets need real browser/server integration tests.
8. **Synchronization concurrency needs stress tests.** Queue writes and source metadata snapshots must survive two edits within one encoder cycle, concurrent review changes, external filesystem writers, failed cache replacements, and program restarts. The source scanner uses file size/mtime rather than content hashes and can miss metadata-preserving edits.

## Recommended pre-merge gate

**Block the merge** until:
- The remaining 15 broad-suite failures are classified on both branches; fixes affecting training/captions get targeted regression coverage.
- A disposable two-epoch GPU smoke run proves source caption edit, latent edit, mask edit, remove/restore, and restart/resume each produce the expected next-epoch caches and unchanged optimizer-step policy.
- An interactive UI test covers both themes, narrow layouts, keyboard navigation and the native LaMa editor.
- Multi-action history, duplicate merge, review/AI races, import collisions and interruptions are exercised under a live server.
- Tests for queue rollback, partial restore, failure injection and stale-write rejection remain green.

**Audit outcome:** significant P0/P1 bugs were fixed and the focused suite is green, but the branch is **experimental**. A completely green targeted CI is not equivalent to production readiness or an exhaustive manual inspection of all repository files.
