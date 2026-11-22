# Caption Studio implementation audit

Test branch: `test/caption-workbench` (based on `wip-test` at `0e9fbf7`).

This is an experimental implementation, not yet signed off for production GPU training.

## Included in this branch

- Native three-panel caption workbench using Fizgig's existing media/caption APIs and theme tokens.
- Search, pagination, missing-caption filter, draft preservation, text undo/redo, tag chips, drag-to-reorder tags, tag index, selection, and bulk tag changes.
- Tag operations preserve prose after the first caption line. Artist-strip recognizes `@` tags and `artist:` tags.
- Existing captioners, Image Prep tools, dataset ZIP routes, and image editing APIs are reused instead of adding Civitai/Buzz dependencies.
- Direct caption edits queue text updates; each epoch also scans source-file metadata to detect edits from bulk tools and external writers.
- Edited still images queue latent refresh at the next boundary. The original training bucket shape is retained.
- Removed images are excluded from subsequent steps; a mixed batch with a removed image is skipped to avoid stale training.
- Text and image cache replacements are atomic. Failed updates stay queued, and interrupted `.processing` files can be recovered.

## Audit findings fixed

- Earlier draft did not actually land on GitHub; commits are now on this separate test branch.
- Disabled per-image loss tracking must remain disabled even though dataset synchronization stays active.
- Source lookup was quadratic for large folders; directory entries are now indexed once per scan.
- Caption drafts could be lost during navigation; current in-memory drafts are preserved between images.
- TagBench's first-line-tag / subsequent-prose convention is respected.
- Image editing in an active run now uses CPU inpainting where applicable.
- Failed VAE refresh must not train old cached pixels; the affected image is excluded until refreshed.
- Interrupted encoder calls must not strand claimed caption/image changes.

## Verification

GitHub Actions runs Python compilation for changed training/web modules, JS parsing, and
`checks/test_caption_studio_sync.py`, `checks/test_caption_live_images.py`,
`checks/test_caption_formats.py`, `checks/test_image_edit.py`.

The repository-wide test suite cannot currently collect because `src/fizgig/krea2/direction_edit_v2.py`
has pre-existing syntax errors on `wip-test`. This file is left unchanged on the test branch.

## Important limitations / remaining validation

1. **Real GPU run not verified.** Smoke-test caption edits, crop, remove, undo, and restore in active training. Confirm the following epoch reads new TE and VAE caches and that the expected number of optimizer steps is preserved.
2. **New images added to a folder are not inserted into an already-created batch schedule.** They require a new/restarted training run. Existing image changes are supported.
3. **Edit-pair/reference latent caches cannot be safely rebuilt by the still-image refresh path.** Such images are excluded after edits until a full recache/new run.
4. **Crop and mask/face-box alignment needs visual verification with masked training** before considering live cropping production-safe.
5. **This is a focused integration, not feature-for-feature TagBench parity.** TagBench's duplicate-finder, category-color system and wildcard export were not ported. Fizgig's native caption models, picture editor and dataset ZIP remain available.
6. **Browser visual and accessibility testing is still needed.** JavaScript parsing and code review do not substitute for hands-on browser testing.

Keep this on the test branch until those validations pass. Do not merge into `wip-test` or release it on the strength of static checks alone.


---

## P1/P2 continuation — 2026-11-22

**Status: implementation pushed to `test/caption-workbench`, not production sign-off.** This section supersedes
the older "features missing" statements above. The older paragraphs remain as a record of the initial audit.
The base is `wip-test`; changes are isolated to the test branch. Do not merge until live GPU/UI verification.

### P1 editing, review, and history

- Shared session-level chronological undo/redo covers saved Studio captions, grouped Studio tag operations,
  Studio picture crop/removal steps, moved/removed dataset images, and legacy Find/Replace, Add Text,
  Static Caption, and Song Caption operations. The backend checks caption snapshots before reverting; grouped
  edits validate every target and review lock before writing. Unsaved Studio drafts block an unsafe history step.
- Tag ordering supports original order, reverse, A–Z, Z–A, dataset frequency and family tokenizer piece
  counts, with configurable first/last pinned tags and first-line/prose preservation.
- Multi-selection includes shift-ranges, ctrl/cmd-click, all matching files, selected-page toggles, and inversion.
- Persistent review locks are enforced by direct save/removal, Studio batch writes, AI worker at write time,
  legacy static/song/add/replace operations, translation at commit, image edit undo/redo,
  duplicate merges, and detail-crop mirroring. A review ledger that cannot be read blocks writes.
- Separate local image editor brush/mask history still handles **uncommitted canvas strokes**. Only committed
  crop/remove image steps are integrated into the Studio-wide chronological history. External writers
  and other non-Caption Studio interfaces do not automatically create Studio history entries.

### P2 advanced TagBench experience

- Three exploration modes: searchable tag index, frequency waterfall, interactive co-occurrence
  constellation (pointer pan, scroll zoom, keyboard tag activation). Uses local data, no cloud analytics.
- Actual per-model tokenizer piece count sorting where a local tokenizer is available; 500-tag request cap
  avoids large downloads/requests. Tags autocomplete from the dataset index.
- Draggable, keyboard-accessible panel separators persist proportions; responsive collapse for narrow screens.
- Keyboard shortcuts include workspace undo/redo, select all, image navigation, and search focus,
  while preserving existing Image Prep canvas shortcuts.
- Drop or choose paired images/captions and ZIPs; reuse the existing resumable uploader. Imports target
  a named Training Data folder; existing-file collisions require user confirmation.
- Selected-image AI caption workflow supports missing-only, replace, or append text as an additional line;
  uses Fizgig's existing local/installed captioners and respects review locks. AI work is asynchronous:
  its completion is **not** yet an atomic grouped Studio history step.

### Automated checks

CI now compiles the modified Python service, dataset writer, training modules, AI worker, translation worker
and tokenizer helper, parses all UI JavaScript, exercises pure tag-order/constellation code, the native LaMa
wiring checks, review locks, duplicate management, atomic batch edit conflicts/rollback, and the original
live-dataset synchronization tests. Check the *latest branch commit* in GitHub Actions; a passing previous
commit does not establish the next commit's status.

### Manual verification required before merge (not yet performed)

1. **Live GPU training:** run at least two epochs on a disposable dataset, change a caption twice within one
   epoch, crop and mask a picture, remove/restore it, verify next-epoch TE/VAE/mask refresh, actual output
   captions/pixels and optimizer-step count. Repeat with a resumed LoRA.
2. **Synchronization races:** edit two images in quick succession; lock a review during AI work; change
   captions during translation; interrupt image re-encoding; simulate missing sidecars and stale metadata;
   confirm no old cache is trained after a failed refresh. Test masked, paired, multiple-subfolder and
   new-image admission scenarios.
3. **Browser visual check:** light and dark themes at 1440px, 1024px, 768px, and 390px widths; resize
   all panels; pan/zoom the constellation; operate the watermark/LaMa editor mounted in Studio; use keyboard
   only; verify tab switching and caption draft retention.
4. **Importer smoke test:** upload ZIP, paired caption and picture files, a folder containing nested paths,
   mixed media, a duplicate filename and an interrupted 8-MB chunk; confirm source dataset safety.
5. **History smoke test:** caption save, multi-selection bulk, picture crop/remove, dataset removal and
   restored/redo state; modify the same caption outside Studio and confirm undo rejects the stale snapshot;
   lock an image and confirm all relevant mutation routes refuse changes.
6. **Regression sweep:** test caption file encodings, detail-crop mirroring, duplicate keeper merges,
   resume-cache behavior and local LaMa inpainting (never replace with TagBench's CPU removal).

**Hardware/UI limitation:** GitHub Actions' available checks are CPU-only. No real active CUDA/ROCm
trainer or interactive desktop browser session was available through this integration, so visual review
and live GPU validation are *not verified*. The repository-wide suite has an existing baseline
collection problem in `src/fizgig/krea2/direction_edit_v2.py` and requires separate repair.
