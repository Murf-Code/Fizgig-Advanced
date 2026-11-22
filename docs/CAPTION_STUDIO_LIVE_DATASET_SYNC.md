# Caption Studio — live dataset synchronization (test branch)

Implemented on `test/caption-workbench`. This is an experimental training path and **not a promise that every
media/dataset configuration can be hot-swapped**. Changes apply at the **next epoch boundary**, not during a
partially prefetched batch.

## Supported paths

| Change | Handling |
| --- | --- |
| Existing still's caption (.txt or dataset-specific caption extension) | Detect on-disk or queued edit; re-encode text. The source is checked before the encoded text is installed. |
| Existing still image edited, removed or restored | Re-encode its VAE latent into an atomic cache file; exclude stale or removed pixels from future steps until refreshed. A changed aspect keeps the active run's existing bucket; the source size metadata is updated. |
| Existing masked still | Detect added, changed and removed loss masks; validate and update the dataset item at epoch boundary. Invalid masks block the image. Masks themselves are read on the latent grid and are not VAE encoded. |
| Existing paired-reference still | Detect before-image changes and re-encode control latents. Reference-aware edit text uses the driver's `encode_text_with_references`; slider text stays plain when its initial text cache has no reference image metadata. Keep the picture excluded until both caches are current. |
| Newly added still, including secondary subjects | VAE and text encode before admission, then add to the active bucket's rotating pool **only if the selected bucket already exists**. Each epoch's original batch/step count and scheduler length remain unchanged. A fixed-size random rotation means a newly admitted item is eligible from the following epoch but **may not be drawn in every individual epoch**. Up to 64 candidates are considered per boundary. |
| H3 existing video/audio source edited | Native MiniMax H3 clip/audio encoder refresh; preserve packed audio, clip still and paired clip latents where applicable. Requires configured audio VAE for audio-bearing sources. Does **not** load the large preview decoder to refresh H3 media. |
| Multi-dataset monitoring | Resolve each active item's owning dataset block rather than relying on the primary folder. Holdout and regularization datasets stay excluded from new-item admission. |
| Failed/interrupted cache refresh | Keep pending work in atomic queues, report it through the UI and exclude stale image or caption items until successful. A successful caption refresh does **not** unblock stale pixels. |
| Visibility | Captions tab polls `/api/captions/live_status` for unscheduled files, blocked items, invalid masks, pending text and image updates and unsupported media edits. |

## Restrictions and required restarts

- A new image needing a **new bucket resolution** cannot be admitted without changing the active sampler/step
  schedule. It remains unscheduled and is reported. Changing its aspect/crop or restarting with a prepared cache
  permits a different bucket. Existing image edits keep the original bucket and can crop more than a fresh run would.
- Newly arrived **video/audio** files are reported as not scheduled; the H3 media live refresh currently covers
  edits to media already present at launch, not dynamic insertion of new clips/voice items.
- Families other than MiniMax H3 do not support live video/audio recaching here; they fail closed.
- MiniMax H3 reference-distillation teacher caches (`...teref*.safetensors`) are a separate multi-image
  dependency graph. Updating those teacher/reference associations requires a cache rebuild and a restart.
- Validation holdouts and regularization images cannot be silently admitted while training.
- Source changes preserving **both file size and nanosecond modification time** may evade the fast scanner.
  A content digest of every large image/video on every epoch is not enabled because of its disk I/O cost.
- The algorithm preserves the current optimizer's **fixed step count and trainable bucket dimensions**.
  It does not introduce extra optimizer steps when more pictures arrive.
- The in-run system is limited by the driver's model-specific encoder paths and available VRAM. If a
  source changes again during encoding, it must retry at the next boundary rather than commit a stale result.
- Tests in `checks/` exercise queue races, images, masks, paired images, caption safety, multi-dataset key
  mapping, H3 audio/clip writer paths, and the fixed-size sampler on CPU using stub encoders.
  **No end-to-end training run on a real GPU has been performed here.**

## GPU acceptance matrix before merging

1. Start a real still-image run; edit a caption, edit an image, edit twice before the next boundary, and
   delete/restore a source. Confirm the correct cache tensor changes, the running loss skips an invalid cache,
   and the subsequent epoch uses the new values.
2. Add a compatible picture to a running existing bucket and confirm that it participates in later epochs
   while the optimizer step count and saved-epoch timing are unchanged.
3. Exercise a secondary subject folder, custom caption extension, repeat count, masks and detail crops.
4. Exercise an edit pair: replace the before-image and caption; ensure latent controls and reference-aware text
   match and the item is not used before both succeed.
5. Exercise H3 clip (with and without audio), H3 voice, clip pair and clip-still. Verify the packed audio and
   video tensor shapes and that the loader reads the refreshed files.
6. Kill an encoding task mid-boundary and resume. Confirm processing files are recovered and newer edits win.
7. Verify memory and pause/stop/restart on Windows/NVIDIA and Linux/ROCm.

All of these tests should pass before changing the experimental branch to a production/default branch.
