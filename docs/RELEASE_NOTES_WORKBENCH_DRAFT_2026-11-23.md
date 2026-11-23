# Fizgig Advanced — Caption Workbench release notes (DRAFT)

## The highlights

**Caption Workbench is a whole new way to work with your training pictures.** The Captions tab is now a visual workspace: browse your dataset, edit pictures, work with tags or natural-language captions, find duplicates and see how individual images are learning, all without jumping between separate tools.

**Edit pictures while a LoRA trains, and Fizgig re-caches what changed.** Crop an image, remove an object with local LaMa, change a caption or update a supported mask; at the next epoch boundary the trainer can re-encode the changed image or text instead of starting the whole cache over. Eligible new pictures can join an active run too. Unsupported changes are flagged rather than quietly used with stale caches. *Full live GPU acceptance is still pending.*

**Pick your verification and sample pictures where you're already looking at them.** Right-click images in Tags or Waterfall to add or remove them from Validation Images or Sample Images, including multiple selections. Sample picks can use the picture's own caption as the preview prompt, while leaving your manually written prompts alone.

**See which images are learning and which are stuck.** With Loss Status on, a second colored outline identifies images the training loss watcher considers **Stuck**, **Learning**, **Easy**, **Fully mined** or in another monitored state. Sort or filter by those results to find pictures worth revisiting. Selection highlighting stays separate.

**Tags finally have a visual language of their own.** Related tags receive consistent colors across Image Tags, All Tags and Constellation. Right-click any tag to choose a preset or custom color, or return it to Auto. Edit comma-separated tags quickly, switch to a natural-language caption field, and keep prose intact when sorting or changing tags.

**Explore your dataset as pictures or as relationships.** **Waterfall** lays the images out in a full-width gallery, preserving their different shapes. **Constellation** turns tag frequencies and co-occurrences into a colorful interactive map; click a tag to reveal its closest relationships. Selections carry back into the Tags editor.

**Find near-duplicates before they waste training effort.** Compare likely copies, choose one or several to keep, preview copying useful tags from the others, and remove only the pictures you explicitly choose. Removed images can be restored, and reviewed pictures are protected.

**The training and Studio tools have grown too.** Krea 2 keeps its new **Text-fusion calibration** and **Text fusion only** training, with Global replay improvements. Optional **mid-epoch best-loss checkpoints**, clearer step counts, faster compatible Studio LoRA switching and experimental model-blending tools round out the release. The separate experimental *Training Text Enhancer* is **not included**.

The full feature explanations, screenshots, bug fixes, optimization notes and upstream credits follow.

**Status:** Working draft, not a released build or production sign-off.  
**Integration branch:** [`test/caption-workbench`](https://github.com/Murf-Code/Fizgig-Advanced/tree/test/caption-workbench)  
**Compiled:** 2026-11-23. Recheck against the branch HEAD, CI, and live hardware before publication.  
**History review:** 654 Workbench-base-to-current commits were inventoried and the user-facing features curated. This draft is not a substitute for live GPU sign-off.
**Scope:** Changes made on Advanced/Workbench since the `wip-test` base, including selective upstream Fizgig 9.8.x adaptations. This document is curated, not an assertion that *every* upstream feature was imported.

## Screenshot gallery — Caption Workbench (assets pending repository upload)

**Four real application screenshots supplied 2026-11-23.** The original PNGs and a ready-to-render Markdown gallery are packaged in `Fizgig_Workbench_Release_Notes_Screenshots.zip` alongside the release-note draft. **These image files have not yet been checked into this public GitHub repository**. The descriptions below are part of the draft, but the four image embeds must be activated after placing the PNGs in `docs/assets/screenshots/`. Do not publish with missing images.

### Tags — image and caption editor
<!-- After PNG upload, activate: ![Caption Workbench Tags editor](assets/screenshots/caption-workbench-tags.png) -->

**Screenshot: `caption-workbench-tags.png`.** Three-panel image and caption workspace: left dataset thumbnails, a large selected macaw preview with Remove/Crop and Undo/Redo, plus Image Tags and the searchable All Tags panel. This illustrates the TagBench-inspired integrated editing workflow.

### Waterfall — image-first dataset exploration
<!-- After PNG upload, activate: ![Caption Workbench Waterfall](assets/screenshots/caption-workbench-waterfall.png) -->

**Screenshot: `caption-workbench-waterfall.png`.** Full-width masonry-style dataset gallery that preserves varying image aspect ratios. Illustrates dense visual browsing without leaving Caption Workbench.

### Constellation — tag frequency and co-occurrence
<!-- After PNG upload, activate: ![Caption Workbench Constellation](assets/screenshots/caption-workbench-constellation.png) -->

**Screenshot: `caption-workbench-constellation.png`.** Full-canvas colored tag graph with frequency-sized nodes and relationships, plus search and zoom controls.

### Constellation — focused tag relationships
<!-- After PNG upload, activate: ![Caption Workbench Constellation selected-tag focus](assets/screenshots/caption-workbench-constellation-focus.png) -->

**Screenshot: `caption-workbench-constellation-focus.png`.** Selecting **raindrops** highlights linked tags and connections, dims unrelated nodes, and displays a ranked related-tag list. Distinct from the general graph overview.

**Verification note:** These screenshots document interface appearance and workflows; they do not constitute performance benchmarks or live GPU-training validation.

## What's new — a complete dataset workbench

This release turns the old Captions page into **Caption Workbench**, a visual home for preparing, tagging, editing and inspecting the same pictures that train your LoRA. These are the feature highlights a first-time user should see, not a chronological stream of implementation commits. The [four screenshot captions above](#screenshot-gallery--caption-workbench-assets-pending-repository-upload) illustrate Tags, Waterfall and Constellation.

### Edit a training image — the cache follows your changes

**Crop, repair or remove unwanted objects without leaving Caption Workbench.** Select an image, use the streamlined **Crop** or **Remove** tools underneath the large preview, and Undo/Redo applied edits. The existing Fizgig **local LaMa** system does the object removal; while training occupies the GPU, the editor can use the CPU. The Image Prep watermark editor remains available separately in its floating glass overlay.

More importantly, **the training dataset is live**. For supported images and caption formats, committed edits are detected and the appropriate **VAE latent or text-encoder cache is refreshed at the next epoch boundary**. Edited masks and supported paired/control references also participate in the refresh. Re-encodes use atomic replacement and guard against the trainer consuming a stale cache after a failed, interrupted or superseded write. A removal takes effect through the live dataset update path instead of requiring the user to rebuild the whole cache manually.

**Add images during a run** where the existing training bucket supports their shape: new eligible stills can be encoded and admitted to a rotating pool while retaining the original number of optimizer steps. New aspect-ratio buckets, unsupported media/layout changes and unsafe updates are reported as needing attention or a restart rather than silently trained with old data. Live image/caption/mask/paired-reference handling has automated tests but **still requires end-to-end real-GPU acceptance before release sign-off**.

Implementation history includes [live still-image admission](https://github.com/Murf-Code/Fizgig-Advanced/commit/4885af56), [mask refresh](https://github.com/Murf-Code/Fizgig-Advanced/commit/7e8e3a3b), and [atomic cache protection](https://github.com/Murf-Code/Fizgig-Advanced/commit/b3cbf33d).

### Choose Validation Images and Sample Images right from your pictures

**Right-click a picture in Tags or Waterfall** to add it to, or remove it from, **Validation Images** or **Sample Images**. Select several pictures to manage them together; the menu reflects whether the current selection is already included. No trip back to Training or Samples settings is necessary, and the same existing training settings are updated rather than creating another list.

For Sample Images, Workbench **uses that picture's caption as the initial preview prompt** and pairs it with the selected training image. Captions that are empty are skipped with feedback. Removing a Workbench-created sample pairing does **not** erase unrelated manual prompts or a prompt the user has since edited. Detail crops are intentionally excluded from these image-pick actions.

See [training-image context actions](https://github.com/Murf-Code/Fizgig-Advanced/commit/7c4ed76b) and `caption_studio_training_picks.js`.

### Training intelligence, visible on every image

Turn on **Loss Status** to see a **second, color-coded outline** around a dataset thumbnail in Tags and Waterfall. The original selection ring stays visible, so selecting an image and seeing its learning state are independent.

Statuses come from Fizgig's **actual per-image loss watcher**, not from guessing that a low raw loss equals an easy image. The UI can distinguish **Stuck**, **Suspect**, **Watching**, **Learning**, **Easy**, **Fully mined**, **Warming up** and **Excluded**. Filter images by these verdicts, or sort by higher/lower relative loss, to review the pictures most in need of attention. An actively tracked training run automatically enables its loss colors unless the user opted out for that run; an old finished run does not silently force the option on.

This turns training-loss analysis into a **visual dataset-review workflow** instead of making users hunt through separate graphs. See [learning-status rings](https://github.com/Murf-Code/Fizgig-Advanced/commit/22402621) and [image-status filter](https://github.com/Murf-Code/Fizgig-Advanced/commit/bc34024a).

### Tag colors that mean something — and stay consistent

Tags are not assigned random unrelated colors in each view. **Constellation groups and tag relationships inform a shared automatic palette**, reused in Image Tags, All Tags and Constellation so the same tag keeps the same color across the workspace.

**Right-click a tag** to override its color: pick from eight prepared swatches, choose a custom color, or **reset to Auto color**. Saved manual choices take priority over the learned palette; existing dataset category assignments continue to be read. The color preferences are **remembered locally for that dataset in the browser**. The current UI no longer needs the old bulky color-settings panel.

See [shared colors and context palette](https://github.com/Murf-Code/Fizgig-Advanced/commit/f6737e21), [shared Constellation palette](https://github.com/Murf-Code/Fizgig-Advanced/commit/909bb941).

### Tags and natural-language captions, without mangling either

Workbench understands the difference between **comma-separated tag captions** and **natural-language prose**. Use Image Tags for quick select/edit of individual tags, with autocomplete; choose the **Natural language** caption mode to work in an expanding text field instead of the tag list. Switching back to **All Tags** brings back the searchable tag index.

With a mixed caption, the **first line may contain tags and subsequent lines may contain prose**. Tag sorting, deduplication and bulk actions preserve those prose lines rather than rewriting the whole caption as tags. The format detector favors prose when ambiguous, protecting ordinary sentences from accidental tag operations.

**Edits save automatically**; selected tag edits are sent promptly, while draft/history state and serialized saves protect against rapid repeated changes. Double-click an existing tag to rename it; leaving its replacement blank restores the original. Use Enter or a clickable autocomplete suggestion to add a tag, and select multiple tags for context actions, batch operations, drag/reorder or deletion. Session undo/redo covers supported committed edits; the local brush canvas has its own uncommitted stroke history.

See [format-safe tag operations](https://github.com/Murf-Code/Fizgig-Advanced/commit/2a828556), [natural-language editor](https://github.com/Murf-Code/Fizgig-Advanced/commit/f6737e21), and [autosave](https://github.com/Murf-Code/Fizgig-Advanced/commit/8c2e197c).

### See your dataset three different ways

- **Tags:** A focused full-height image, narrow scrollable thumbnail rail, two independently scrollable lists for Image Tags and All Tags, right-click actions, search-as-you-type, and in-place editing. Use Ctrl/Cmd and Shift to select multiple images, arrow keys to navigate, and shortcuts to focus searches or invoke common operations.
- **Waterfall:** A **full-width masonry gallery of the actual images**, not merely a tag-frequency bar chart. Thumbnail proportions are preserved, and selecting a picture carries the same active selection back into Tags; double-click to open the chosen image there. Context actions, including Validation and Sample Image picks, work here too.
- **Constellation:** A large interactive **tag co-occurrence map** with frequency-sized colored orbs, linked relationships, clusters, search, pan and zoom. Select a tag to spotlight related tags and their connection strengths while unrelated nodes dim. The focused view includes a ranked list of related tags. This is especially useful for finding patterns and inconsistent descriptions in a dataset.

The **Glass / Classic** appearance switch, responsive/resizable panels, compact editing toolbar and full-window layouts make these tools feel like one application rather than separate forms. The four supplied screenshots belong immediately above these highlights. See [full-size Waterfall and Constellation](https://github.com/Murf-Code/Fizgig-Advanced/commit/072920e2).

### Find near-duplicates without risking your originals

**Find Duplicate Images** scans for exact and visually similar images, using perceptual difference hashes and a color-grid comparison. Adjust the **0–16 similarity tolerance** (larger values are more lenient), inspect side-by-side thumbnails, and choose **one or multiple pictures to keep**.

Optionally **preview a tag merge** from unkept images into every chosen keeper; prose remains intact. Tag merging is separate from any deletion. Only the explicitly chosen **unkept images** are moved to a reversible `removed/` folder, with their related captions/masks and collision safeguards. Reviewed/locked pictures are protected. The scan is CPU-based and does not load a GPU training model. See [duplicate review guide](CAPTION_STUDIO_DUPLICATE_REVIEW.md).

### Faster captioning and safer bulk editing

Use Fizgig's existing local AI captioners on your **selected pictures**, rather than always captioning a whole folder. Choose to **caption missing images only**, **replace selected captions**, or **append an AI caption as another line**; reviewed items are protected and destructive choices require confirmation. Batch Find & Replace includes a preview and undoable commit. Tag queries support **AND / OR / NOT, exact quoted tags and exclusions**; live search and sorting make large tag lists approachable. Export a **caption wildcard** containing the complete caption for each image, instead of a de-duplicated vocabulary list.

Use **Mark Complete / Incomplete** and persistent review locks to protect pictures whose tags are finished. Delete image/tag actions require confirmation, and removed image groups can be restored without blindly overwriting existing files. The workbench avoids unwanted re-captioning of the entire dataset when an explicitly selected target is invalid or empty.

### Training workflow improvements worth showing users

- **Krea 2 Text-fusion calibration remains included.** Manually requested concepts get directly tested against controls, with an additional automatic target budget of up to 4% of training-image count; measured suppression and held-out verification determine whether an edit is retained. This is **not** the experimental Training Text Enhancer, which was deliberately reverted.
- **Text fusion only** is a separate Krea 2 training choice for the text-fusion blocks, alongside **Standard** and **Global replay** training with optional selective block checks. Global replay's compatibility work includes PoLoRA, training masks and Training Maps where supported.
- **Optional rolling best LoRA checkpoint** can capture a promising low-loss point **mid-epoch**, not just at the ordinary save interval. It supports training-loss or fixed validation-loss selection, smoothing/cooldown safeguards, and resume bookkeeping. The manual **Save Now** action and epoch saves still exist.
- **Clearer training feedback:** real epoch steps remaining for Global replay/resume, separate validation groups on the loss graph, and more useful pause/stop, Save Now and run-output states.
- **Smarter resume and caching:** skip demonstrably clean text/VAE cache stages; re-encode only changed images or captions; retain compatible caches and respect explicit cache policy where necessary. Cache validity includes fine-grained file timestamps, encoder provenance and safe fallbacks for complex datasets.
- **Studio upgrades:** upstream-adapted primary-LoRA quick switching with rollback; render-specific PNG setup metadata; per-browser-tab ownership protections; edit-reference aspect/Reference MP control; optional MiniMax H3 Image speed boost; and clearly flagged experimental Frankenstein/block-based model blending.

### Built for large datasets

Under the hood, Caption Workbench uses **recycled native-scrolling thumbnail rows**, deferred image decoding, bounded preview caches, progressively loaded Waterfall pictures and All Tags rows, **Constellation calculation only when needed**, compiled tag-search predicates, and reduced work from inactive tabs. The Remove brush uses a lightweight overlay, bounded/tiled history and an offscreen painting worker with a safe fallback. These choices aim to keep image navigation and editing responsive without wasting RAM and CPU.

**Release-note framing:** because this is the first public Workbench experience, describe these as *large-dataset design benefits*, not claims of a speedup over a Workbench version users never used. Benchmark numbers belong in technical notes after reproducible hardware measurements, not the opening feature story.

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
| [#398](https://github.com/shootthesound/Fizgig/pull/398) and [#399](https://github.com/shootthesound/Fizgig/pull/399) by [@spaghettified26](https://github.com/spaghettified26) | **Integrated.** Pillow 13.0.0 update for PNG EXIF safety (#398) and faster PNG dimension/header reading without full image decoding in dataset listings (#399), with regression tests and source credits. See [Pillow commit](https://github.com/Murf-Code/Fizgig-Advanced/commit/aaa1062363) and [PNG optimization](https://github.com/Murf-Code/Fizgig-Advanced/commit/72759c1aa2). |
| **TagBench / TagGUI** (user-supplied TagBench 1.2 reference) | Inspiration for caption editing, Waterfall, Constellation and duplicate review. The branch [explicitly credits TagGUI](https://github.com/Murf-Code/Fizgig-Advanced/commit/0c963857). Add the confirmed upstream URL/author before publication rather than inventing one. Fizgig adapts the experience to its own services and retains local LaMa. |

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
