# TagBench 1.2 parity and resource audit — 2026-11-22

**Target branch:** `test/caption-workbench`  
**Reference:** user-supplied `tagbench-build 1.2(5).zip`, particularly `src/App.tsx`, `src/Explore.tsx`, `src/Duplicates.tsx`, `src/AutoCaption.tsx`, `src/Paint.tsx` and helper modules.  
**Status:** source-level functional audit and targeted optimizations completed; **NOT production-certified**. No live desktop/GPU run was conducted.

## Shared selection — fixed

The main reported regression was genuine: the Waterfall click handler tracked image selection separately from the Tags page/anchor and did not trigger a synchronized view rebuild. The Tags renderer could choose the first thumbnail on its previously displayed page instead of the Waterfall-selected image.

The fix installs `caption_studio_selection.js` as shared, pure state transition logic:
- Tags and Waterfall use identical plain-click, Ctrl/Cmd-click and Shift-click range behavior.
- Active image, range anchor, selected path set and correct pagination are updated together.
- Constellation tag picks update the selected matching image and request correct page handoff.
- Returning to Tags scrolls the active thumbnail into view (as TagBench does).
- Context-menu fallbacks and arrow navigation use the live active path, not the former page's closure.
- Dataset refresh prunes paths that no longer exist. Switching training folders resets the selection.
- Waterfall updates `aria-pressed` along with visual marking.
- Also fixed a separate `loadCaptions` local variable shadowing its generation-guard function.

**Regression check:** `checks/test_caption_studio_selection.cjs`, run from `.github/workflows/checks.yml`. It covers selection across page boundaries, Ctrl/Cmd toggle, Shift ranges, active focus and stale-path pruning. It is a pure unit test; a real browser interaction remains to be checked.

## Resource optimization pass — committed

| Subsystem | Before | Now | Preserved behavior / caution |
|---|---|---|---|
| Constellation tag co-occurrence | Constructed every pair from every tag in every caption, then discarded pairs outside the top 120 | Count unique tags first; build pairs only among the top 120. Pair map is capped at **7,140** entries | Same top tags, counts and co-occurrence edge weights. Exact comparison verified on a synthetic dataset |
| Constellation color assignment | Repeated linear searches across up to 120 nodes for every unique tag | Pre-index node colors by family and normalized stem | Same manual category color priority; avoids repeated scans |
| Waterfall image gallery | Eagerly built the DOM and thumbnail elements for the entire filtered dataset | Build 120 tiles initially and append subsequent groups on intersection/scroll; lazy image decoding; dispose observers when leaving the view | All images remain accessible by scrolling; on browsers without IntersectionObserver a visible Load More button remains |
| All Tags list | Built thousands of tag row/button nodes on every caption selection or redraw | Mount the first 120, then append groups of 120 near the scroll end | All tags remain searchable and reachable; list is not truncated |
| Boolean tag filtering | Parsed an AND/OR/NOT expression once for **each image** | Compile the filter once per dataset view render, then evaluate it for each image | Exact matching and existing query grammar preserved; dedicated checks added |

**Illustrative synthetic benchmark (not an end-user hardware measurement):** a 1,200-caption / 27-tags-per-caption workload compared the former versus optimized Constellation graph builder. It returned identical top tags and 823 identical edges, taking **192 ms** before versus **15 ms** after in the same JS execution environment. This comparison isolates graph construction, not the overall visualization, image decoding, layout, browser memory, or training process. The exact speedup will vary by dataset and hardware.

**Test additions:** `checks/test_caption_studio_viz.cjs` now checks TagBench model edge weights, duplicate-tag handling and the top-120 bound; `checks/test_caption_studio_tags.cjs` checks compiled Boolean queries.

## Functional parity inventory

| TagBench 1.2 feature / interaction | Caption Workbench finding |
|---|---|
| Image selection shared between Tags, Waterfall and Constellation | **Repaired** in this audit; browser smoke test still required |
| Fully expanded Waterfall masonry gallery | Implemented; now progressive for lower resource use. Selection/paging and deep scroll need hands-on verification |
| Tag co-occurrence Constellation | Implemented using TagBench-inspired grouping, spring layout and palettes; behavior-focused unit checks added, but visual pan/zoom and performance still need interactive validation |
| All Tags list, search, sorts and category colors | Implemented; initial DOM now bounded. Compare nuanced category UI and bulk behaviors in browser |
| Boolean tag queries and tag context actions | Implemented for AND/OR/NOT and tag edits; compiled query tests added |
| Review/done and lock behavior | Implemented for covered mutation APIs, but not a universal filesystem lock; concurrent AI/image writes need race tests |
| Image/caption undo and redo | Committed actions use shared Studio history; uncommitted brush strokes use separate Image Prep history |
| Asynchronous AI captioning and duplicate keeper merges | **Partial parity:** these are not guaranteed as a single reversible Studio history transaction |
| Exact TagBench tags-to-prose transformations, presets and automated category logic | **Remaining parity work:** not all original semantics are present; no claim of full one-for-one coverage |
| Image crop and object removal | Integrated via Fizgig's own Image Prep editor and local LaMa instead of TagBench's CPU remover, by design |
| Auto-captioner, selected-image targets and modes | Native equivalents exist; nuanced TagBench placement/preset differences still need UI acceptance tests |
| Dataset import/ZIP, wildcard exports | Native operations exist; large-file/interrupted upload and collision workflows still require manual integration testing |
| Active training re-encoding after changes | Implemented for supported cases; **not GPU-certified**. See `CAPTION_WORKBENCH_FULL_BRANCH_AUDIT_2026-11-22.md` for unsupported and race-prone cases |

## Remaining risk and validation plan

1. **Interactive selection acceptance:** choose image #70+ in Waterfall, switch to Tags, confirm correct page and visible active tile; return to Waterfall, Ctrl/Cmd-select several items, Shift-range across images, right-click to act on precisely those images, switch between views again. Repeat with image-name and tag filters and a dataset refresh.
2. **Performance acceptance:** profile a representative 10k+ image/large-vocabulary dataset in Chrome or Edge DevTools (JS heap size, live DOM node count, image decode activity, long tasks, interaction latency) before and after the changes. The synthetic graph benchmark is not a RAM/VRAM measurement.
3. **Rendering acceptance:** test dark and light themes, narrow widths, Waterfall Load More fallback, resizing while images are loading, Constellation pointer pan/zoom, scrolling to selected images, and tag-list scrolling after search/sort.
4. **History acceptance:** finish AI caption / duplicate merge, then Undo/Redo; verify no partial reversals, and that review locks hold during concurrent operations.
5. **GPU/training acceptance:** test queued image/caption/mask edits, removals and restores at epoch boundaries including restart/resume, as required by the existing branch-wide audit.

Do **not** merge to `wip-test` solely because CI passes. The branch remains experimental pending GPU and interactive browser verification.
