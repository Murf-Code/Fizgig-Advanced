# Caption Studio — TagBench-style duplicate-image review

Experimental feature on `test/caption-workbench`. Based on the original user-supplied
`tagbench-build 1.2.zip` (`src/dupe-match.ts` and `src/Duplicates.tsx`).

## Ported TagBench concepts

- **64-bit difference hash (dHash)**, a **4×4 color grid**, and exact file matching;
  similarity differences are `max(Hamming difference, colored-grid distance)` unless
  the input bytes are identical. The comparison runs on the CPU.
- **Strictness 0–16**; default is 4. A larger number is **more lenient**, not
  more strict. Adjusting the slider reuses decoded fingerprints (no re-scan).
- **Connected duplicate groups**: transitive groups are possible, so not every
  pair in a group is necessarily within the chosen difference threshold.
  Inspect the thumbnails rather than deleting based on group membership alone.
- **Keep one or multiple pictures**: default to the most richly tagged image,
  but allow any combination with at least one keeper.
- **Optional copy tags from dropped images to every keeper**, with case-insensitive
  deduplication, first-line tag handling and preservation of the keeper's
  additional lines. Changes can be previewed before application.
- **Human-controlled whole-picture removal**: separate action to move unkept
  images to the dataset's `removed/` folder, along with their captions and
  masks. There is no automatic deletion as part of tag merging. The previous
  removal journal allows restoration after the app restarts.

## Fizgig adaptations

- Scans are performed by a separate CPU-only worker on the backend. They do
  **not** load a training or image-generation model and don't use GPU VRAM.
  A scan has a 30,000-photo input cap and a 150,000-pair result cap; the UI
  reports a truncated scan.
- Applies the existing Caption Studio review locks to caption edits and to
  whole-picture removal. Unsupported prose captions are not merged into tags.
  Stale captions fail instead of overwriting edits made after the preview.
- All merged captions are written atomically per file, with best-effort
  rollback if a multi-keeper merge encounters an unexpected failure. Live
  training is notified to refresh changed text caches at the next epoch.
- **Fizgig's local LaMa object-removal tool is retained without replacement.**
  The duplicate-image feature compares images and may move complete image files
  to `removed/`. It **does not** inpaint objects. When editing pixels,
  Caption Studio still uses `web/image_edit.py -> lama_fill ->
  fizgig.dataset.lama.inpaint()`, choosing CPU while training holds the GPU.
  No TagBench CPU object-removal implementation was imported.
- Photographs are read upright and can reside in secondary dataset folders.
  Masks, watermark sidecars, removed folders, and out-of-dataset paths are
  excluded. The worker checks file sizes and modification times while hashing;
  re-run Scan after editing images.

## Acceptance testing before production

1. Scan a dataset with resized copies, slight color changes, distinct
   images, mixed image formats and masks; verify group boundaries and slider.
2. Keep two or more images; preview copying tags to each, then apply. Confirm
   every keeper's updated caption and all donor images/captions are preserved.
3. Toggle copy tags off and move only checked unkept pictures to `removed/`;
   confirm a second run can restore them without overwriting existing files.
4. Verify reviewed/locked images cannot have captions overwritten or be moved
   out of the dataset. Prose stays unchanged, and stale edit previews reject.
5. Exercise a running LoRA training session: edited keeper captions must be
   refreshed at the next epoch; whole-picture removals must exclude stale
   items before the next training step.
6. Open an image in Caption Studio and use **Remove Tool**. Confirm the actual
   local LaMa model is used (on CPU during training), not TagBench's remover.
7. Test large datasets, secondary subjects, themes, keyboard controls, and
   narrow windows. The unit tests cannot replace a live browser/GPU pass.

Status: feature implemented on the test branch with CPU regression and UI
wiring checks. It should not be merged into the default branch until browser
and GPU integration have been exercised.
