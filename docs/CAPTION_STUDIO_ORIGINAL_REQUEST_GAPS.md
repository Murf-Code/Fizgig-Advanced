# Original request vs. Caption Studio branch — gap review

Source of truth: the user's first request ("add this UI and functionality to the caption tab", a *total overhaul*, reuse existing edit/crop tools, avoid unnecessary external APIs, build a new test branch from **wip-test**), followed by the requirement that **any dataset change while training refresh at the next epoch**.

TagBench reference inspected: `tagbench-build 1.2(1).zip` supplied in the conversation; relevant files are `tagbench/README.md`, `src/App.tsx`, `src/Explore.tsx`, `src/Duplicates.tsx`, `src/Paint.tsx` and `src/AutoCaption.tsx`.

This document is an **audit of gaps**, not a claim that they have been implemented.

## Explicit request reconciliation

| Original request | Result on `test/caption-workbench` | Remaining work |
|---|---|---|
| Branch from current `wip-test` on a separate test branch | Implemented, branched at `0e9fbf7` | Keep experimental branch isolated |
| Total overhaul of Captions tab visuals, adapted to Fizgig Advanced | **Partial**: new three-panel Studio view within old tab, but old setting/captioner sections remain stacked above it | Recompose tab so Studio is the primary workspace; move old tools into reachable panels/drawers |
| Add reference UI **and functionality** | **Partial**: browser, focused preview, caption editor, tag chips/index and basic bulk actions are present | See TagBench feature inventory below |
| Reuse existing cropping and image tools, avoid unnecessary APIs | **Partial**: edits hand off to Image Prep; existing AI captioners and local image edit APIs are reused | Native integrated edit/crop/remove/undo/redo experience within Captions tab; do not add Civitai Buzz dependency merely to replicate buttons |
| Re-encode captions **and images** after edits during active training, applying changes at next epoch | **Partial**: file/queue-based text refresh, still-image VAE refresh and removal exclusion | Mask/paired media, new images, video/audio, multi-dataset key mapping, and real active-training validation are not complete |

## Reference features not yet integrated

- **Color-coded tag categories:** create/assign/edit categories and colors, colorize chips and search/filter UI; TagBench includes context menus for tag colors.
- **Advanced tag search:** AND / OR / NOT expressions, exact tag clauses, combination/exclusion via tag actions. Fizgig currently has a substring search and missing-caption filter.
- **Tag exploration:** TagBench has three views, `tags`, `waterfall`, and interactive `constellation`, with tag co-occurrence links, selection, zoom/pan and focused browsing; only a simple frequency list is present in Studio.
- **Tag index controls:** sorting by frequency, name or tokenizer count, ascending/descending; tag autocomplete and range selection/multi-select are absent or not equivalent.
- **Duplicate image management:** perceptual/shape/color comparisons with strictness slider, choose one or multiple to keep, optionally merge tags into kept images before discarding duplicates. No comparable Caption Studio feature.
- **Review/done workflow:** marking items complete, restricting destructive edits on reviewed items, progress visible in browser. Not implemented.
- **Tag order and separators:** global reorder rules (e.g. front tags, reverse, alphabetical, frequency), configurable delimiters and spacing, plus multiline tag layouts. Studio assumes comma-separated tags on the **first line**.
- **TagBench-style undo/redo:** TagBench tracks caption, tag, and image edit history; Studio only has per-caption in-memory undo/redo, while Image Prep has separate history. Bulk edits cannot be undone together in Studio.
- **Wildcard export:** `__tagbench__` prompt wildcard text export, separate from Fizgig's dataset ZIP routes.
- **Integrated auto-caption workflow:** existing Fizgig models exist in the tab and Studio has Re-caption shortcut; TagBench's explicit target/selection batch workflow, append/replace choices, tags-vs-prose destination and tags-to-prose action are not reproduced in Studio.
- **Data arrival/workflow:** TagBench accepts ZIPs or sets of image/caption files within its workspace; Fizgig has data import in Start, but not equivalent drop-to-Studio interaction.
- **Picture tools in place:** TagBench's stage has crop, local remove/inpaint and prompt edit, with unified undo. Studio currently links to separate Image Prep (and refuses this shortcut for subfolder items).
- **Adjustable layout and keyboard ergonomics:** reference offers resizable panels, split tag panes, shortcut handling (image navigation, crop, remove, tag selection, search, undo). Studio's panels are fixed CSS columns with limited pointer actions.
- **Tagging a prose caption:** TagBench explicitly treats the first line as tags and subsequent lines as natural-language prose; current Studio assumes **every** first line is a comma-tag list. It must recognize or explicitly choose natural-language-only captions to avoid misinterpreting free-form captions as tags.

## Live-update gaps: high priority because of explicit user requirement

1. **New images:** the active trainer's cached `group` / bucket schedule is static. A newly added picture is not inserted into the current run at a boundary. The audit already documents this limitation; it contradicts the user's unqualified "dataset changed in any way" wording.
2. **Mask and sidecar edits:** the epoch scanner tracks main picture file metadata and associated caption metadata only. Changes to training masks, alpha/segmentation sidecars, control/reference images, detail-crop source relations and other relevant sidecars are not generally detected/rebuilt.
3. **Paired-reference images:** live still-image refresh refuses paired latent caches and excludes the image until a full recache/new run. This is a safety fallback, **not implementation of live re-encoding for this dataset type**.
4. **Audio/video:** the picture lookup scans image extensions only, so a modified clip or audio source is not sent through an equivalent media encoder path.
5. **Multi-folder/other subject dataset edits:** queue mapping is derived from the primary training folder and/or key prefixes; verify correct ownership of every secondary concept and detail-crop path before declaring universal synchronization.
6. **Synchronization races:** do an end-to-end test while a run is active: change caption and picture, edit twice before next epoch, remove/restore image, and test a failed or interrupted re-encoding. Ensure no stale latent/text cache is used after failure.
7. **Current shape rule:** keeping existing buckets protects schedule/step counts but means a changed aspect ratio will continue training at the previous bucket shape; this needs explicit UI disclosure, and mask alignment must be tested.
8. **Detection precision:** the source scanner uses file path, size and nanosecond mtime, not a content digest; changes that preserve metadata cannot be guaranteed to be noticed.

## Suggested acceptance checks

- Browser: Studio dominates the Captions tab; navigation does not lose drafts; complete tag management and duplicate/visualization flows; mobile and light/dark themes work; existing captioners and tools remain accessible.
- Trainer: captions, latents, masks and supported media update at next epoch without restart or stale cache; remove/restore is correct; all datasets and caption extensions are covered; unsupported changes are explicitly flagged rather than silently accepted.
- CI: targeted regression tests currently exist; a live GPU test has **not** been performed. The full existing project test suite is blocked by pre-existing syntax errors in `src/fizgig/krea2/direction_edit_v2.py` on `wip-test`.

## Priority

**P0:** Dataset-sync correctness for supported files, including masks; clear exclusions and feedback for unsupported changes; full active-training smoke test.

**P1:** True in-tab image editing, category-colored tags, advanced tag querying, duplicate manager, full undo/redo, review/done state, format-safe tag editing.

**P2:** Waterfall/constellation views, tag auto-complete/analytics, wildcard export, resizable panels and additional keyboard shortcuts.

External Civitai/Buzz calls and paid prompt editing from the TagBench demo are **not required** by the original request if equivalent local Fizgig functionality is available. Preserve local workflows where possible.
