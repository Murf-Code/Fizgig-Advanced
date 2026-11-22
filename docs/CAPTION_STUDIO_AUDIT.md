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
