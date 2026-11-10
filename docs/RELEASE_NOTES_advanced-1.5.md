# Fizgig Advanced 1.5

This update makes Fizgig Advanced faster and easier to keep an eye on. PoLoRA's optimizer step is now about 2× faster with the same maths, and up to about 3× with Compile, now on by default, and background loading reads the next pictures while your GPU trains, so a slow drive holds you up less. A new CPU/GPU/disk readout on the Run card shows where the time goes and tips you off when something's slowing training down. Edit masks has become a small picture editor, with Remove for watermarks and a proper crop tool. Validation now holds its pictures out of training, so it measures what the LoRA has never seen. Auto memory planning learns from your own runs, and every theme gets the shimmering bar effects.

## Heads up
- This update needs **PyTorch 2.15 (CUDA 13.4)**, a recent **NVIDIA driver (580 or newer)** and **Python 3.11 or newer**. Run the updater as usual.
- Updated to the latest Fizgig (v9.5.0). Batch size is now Fizgig's own setting for every model (Other Options → Batch size). If you had Batch Size set in the old box, set it again there.

## New
- **Edit masks is now a picture editor** (Image Prep, Masks card, 4. Edit masks), with two new tools next to Brush, Eraser and SAM click:
  - **Remove** (J): paint over a watermark or anything else and press Enter (or Remove it), and Fizgig fills it in from what's around it. Draw a loop and everything inside goes too. Right-click rubs the stroke out, the mouse wheel sizes the brush (Ctrl+scroll zooms), and "Hold for before" compares. It uses OpenCV's Telea fill, which comes with Fizgig, so there's nothing to download. Best on small things on plain or soft backgrounds.
  - **Crop** (C): a crop box with corner and edge handles that you can drag around, a dark surround, a rule-of-thirds grid while dragging and the size in pixels. Shape locks it to Free, Original, 1:1, 4:3, 3:4, 16:9, 9:16, 2:3 or 3:2. Enter crops, Esc cancels. The mask is cropped with the picture, and Crop waits while a run is training.
  - Ctrl+Z / Ctrl+Y undo and redo strokes, Removes and Crops in one history while Fizgig stays open (up to 30 picture changes). Ctrl+Z clears a Remove stroke you haven't applied first.
  - Pictures are saved in their own format (JPEG at quality 95 with full colour detail, lossless WebP stays lossless, PNG as PNG), keeping see-through parts and colour profiles. Phone photos are saved upright. Saves are written safely, so a failed save never breaks the file.
  - Mask Save works while a run is training: the run uses the edited mask the next time that picture comes up (a mask added to a picture that had none at the start waits for the next run).
  - **Watermark masks load into Remove.** If you used Mask watermarks too or Just hide the watermarks, each picture's watermark area is pre-loaded as a red Remove area (on by default, with a 2–5 px pad, 3 by default). Paint to add or right-click to erase, and your edits are saved to that picture's watermark mask, so they stay when you move between pictures. Red you paint on a picture the finder missed becomes its watermark mask too. Press Enter (or Remove it) for one picture, or use **Remove all watermarks**: click once and it turns into **Confirm Removal**, click again to start (the number of pictures is shown beside it). Originals are copied into the folder's removed_watermarks/ first. The training mask in that area is put back the way it was before the watermark was hidden, so your other mask edits stay. Ctrl+Z after Remove it brings back the picture, its red area and its training mask. Needs a new Make masks run so Fizgig can save the watermark area on its own. Blocked while a run is training, like Crop.
- **Hold the validation images out of training** (Image Validation, on by default): a few pictures are set aside before caching and never trained, so validation measures pictures the LoRA has never seen and overcooking shows up. Your picked validation images are the ones set aside; if you haven't picked any, Fizgig picks about a tenth (up to "Images to check") spread over your folders and picture shapes, and the log names them. Under 5 pictures, nothing is set aside. It works without Fixed validation loss too. Nothing is moved or copied (Fizgig just keeps a list), the pictures are back as soon as the run stops, pauses or ends (or at the next Start if the app was closed mid-run), and Resume sets aside the same ones. Not used for sliders, fine-tunes, edits, Fizgig LoRA Isolation or clip and voice models.
- **Background loading** (Preferences → Training memory, on by default): while the graphics card trains on one step, Fizgig reads the next steps' cached pictures and captions, so a slow drive holds training up less. Same pictures, same order, same results (a check compares runs with it on and off step by step). It uses a little more system RAM (two steps' worth). In a simulated slow-drive test, a short run took 1.6 s instead of 2.4–3.1 s. It stays off for runs with MiniMax reference pictures. FIZGIG_BACKGROUND_LOADING=0 or 1 overrides the setting.
- **CPU / GPU / disk readout** on the Run card: a small row under the tiles shows how long the GPU waits for data, GPU use and power, graphics memory, CPU and drive read speed, every few seconds. At the end it shows the run's averages and one plain tip (like "GPU waited on data 35% of the time - the cache drive may be slow"), also in the run log. It runs on its own thread, so training doesn't slow down, and its colours follow your theme, including the colour-blind one. FIZGIG_PERF_READOUT=0 turns it off.
- **Three new optimizers to try** (all experiments for now):
  - SOAP: Adam with a smarter view of each layer's shape, at normal AdamW learning rates. Slower and needs more memory, so best on smaller models like Anima and SDXL.
  - LoRA-TSD: like PoLoRA, trains each LoRA's down/up pair together. Picking it fills in LR 5e-3 (Adaptive LR: 2e-3 to 1e-2). Plain LoRA only.
  - CMuon (Fine-tune only, Fine-tune → Optimizer): Muon that steps each part of a combined weight (Q/K/V, gate+up, AdaLN) on its own, so no part drowns out the others. It holds a momentum copy of the weights being trained, so windows hold fewer parts. Adafactor is still the default.
- **PoLoRA boxes and preset:** Magnitude (magnitude_scale), Curl (curvature_beta) and Compile now have their own boxes under Optimizer, with hover text. Compile is on by default whenever PoLoRA is the optimizer (untick it, or put compile=False in the args, to turn it off); where it can't compile, PoLoRA runs uncompiled with a note in the log. They only show for PoLoRA and stay in step with Optimizer Args both ways (Optimizer Args is still the one record, so old args still work). Every model now has a built-in **🪄 PoLoRA** preset to get you going (Adaptive LR on from 1e-2 to 4e-2, magnitude_scale 3, curvature_beta 0.99, compile on, Learning Rate 2e-2, and the rest from the model's standard LoRA preset). New runs still start on the model's standard preset, so pick PoLoRA when you want it. Saved runs and Last Train keep their optimizer. Fine-tunes, sliders, edits and LoKR are unchanged.
- **"Rotated INT8 backward (experimental)"** in Memory & Precision, off by default. With an INT8 base, the backward pass normally runs in bf16 (exact); tick this and it runs in INT8 too, after a Hadamard rotation that spreads out gradient spikes so less is rounded away. It should be faster on cards with fast INT8 but keeps a second INT8 copy of the block weights (more VRAM), and warns that it may harm likeness. Only shown when Base precision is INT8 or Auto (with Auto, only used when Auto picks INT8).
- **Bar effects for every theme:** training, VRAM/RAM and download bars shimmer with a soft light sweep and tiny twinkling stars, and buttons glow and lift on hover, in each theme's own colours (blue for Fizgig, orange for Light n' Orange, teal for Graphite, blues only in Colour-blind safe, extra soft in Dim). Cosmic Pigeon and Deep Space's effects are polished too: they loop smoothly, no longer restart or reset when the status refreshes, the sweep crosses long bars in one pass, and stars only show on active bars. Off with "reduce motion".

- **JoyCaption, WD EVA02 and the Captioning models card** (from upstream Fizgig v9.5.0): the Captions tab can use JoyCaption Beta One for detailed training captions, and the WD EVA02 tagger for booru tags on anime and illustration. Preferences has a Captioning models card so you can download them ahead of time. Auto-recaption still uses only Qwen3-VL.
- JoyCaption's task menu has a new **e621 tags** type: a comma-separated e621-style tag list (species, anthro/feral, gender, body parts, fur colour, clothing, pose, background, artistic style), lowercase with spaces between words, cleaned and with duplicates and rating/score/meta junk dropped.
- **Fine-tune optimizers** (from upstream #312, with CMuon kept): picking Fine-tune switches to the Fine-tune optimizer (Adafactor by default; also AdamW8bit, AdamW, AdEMAMix8bit, Lion8bit, and CMuon). The LoRA optimizer row, including PoLoRA, is hidden and not sent; the Run card shows the optimizer that actually trains.

## Faster
- **PoLoRA and Muon steps do less busywork.** PoLoRA now keeps its optimizer state batched, updates weights a whole group at a time and runs its size checks (sigma_max) as fewer, bigger batches, also compiled when Compile is on: about 25–50% fewer operations per step. Muon does its look-ahead in one batched operation instead of one per LoRA matrix (a small gain, as the rest was already batched). The maths is the same (checks compare old and new steps), and resumes work both ways between old and new versions. PoLoRA's Compile is now on by default, and on Krea 2 it no longer falls back to the slower uncompiled step on torch 2.10. The first step takes about 30 seconds once while it compiles. A failed compile is logged as a warning so it's easier to spot.
- **Speed-ups in numbers.** The PoLoRA and Muon changes above don't change the maths, and they work on any NVIDIA card (and on CPU). The biggest gain is on a GPU. Here's PoLoRA's optimizer step on an RTX 4070 Ti, in milliseconds per step (median, lower is better):

  | Model | Rank | Uncompiled, before → after | Compiled, before → after |
  |---|---|---|---|
  | Krea 2 | 16 | 507 → 286 | 393 → 153 |
  | Krea 2 | 32 | 506 → 346 | 399 → 198 |
  | SDXL | 16 | 445 → 211 | 374 → 109 |
  | SDXL | 32 | 444 → 213 | 372 → 126 |
  | MOSS-Image | 16 | 152 → 78 | 136 → 39 |
  | MOSS-Image | 32 | 152 → 86 | 133 → 67 |
  | Anima | 16 | 202 → 103 | 167 → 56 |
  | Anima | 32 | 189 → 99 | 178 → 49 |

  The Krea 2 compiled numbers include the compile fix above (before, its compile fell back to the uncompiled step). The first step is slower while it compiles, about 30 seconds once.
- **Krea 2's text cache is about a sixth of the size on disk:** each caption keeps only its real words (about 80 tokens) instead of all 512 padded slots. Training is exactly the same. Existing caches still work; new or remade captions are written the small way.
- **Tag shuffle caches fewer, smarter variants:** identical variants are made and stored once, and one that matches the original caption uses the picture's own cache, so short captions no longer get copies of themselves. A new "Max caption variants per picture" box (default 8, counting the empty caption from caption dropout) keeps the text cache and first caching time in check. It only shows while tag shuffle, tag dropout or caption dropout is on.
- **Faster block swap switches itself off when it won't help** (32 GB of system RAM or less, only a few blocks swapped, or pictures over 1 MP), with a note in the log. Want it anyway? Tick "Even when it won't help" under Preferences → Training memory.

## Changed
- **Deep Space is now the default theme**, but only on a device (browser, phone or app window) where you've never picked one; if you have, you keep it. Picking the classic blue Fizgig theme is now remembered too. If you picked Fizgig before this change, you'll see Deep Space once: pick Fizgig again under Preferences → Theme and it stays.
- **Small signatures are found:** the watermark detectors now also look at four zoomed-in, overlapping quarters of each picture, so small or faint signatures that vanished when the whole picture was shrunk get caught (on one test set, 118 of 177 pictures instead of 65). It takes about 3–5× longer per picture. Works with or without SAM 3.1.
- **Watermark masks are cleaner** (when SAM 3.1 traces them):
  - Boxes are made bigger before tracing (extra room sideways for signature ends), and boxes that overlap or sit close together are merged first. Specks are removed, gaps between letters joined and the shape grown a little. A trace that looks like noise uses the whole bigger box instead.
  - When SAM finds several shapes, it keeps the biggest one mostly inside the detector's box plus nearby pieces, and drops art poking in from the edges. This works for symbols and crests, not just text.
  - When SAM misses a faint watermark or traces the picture's texture, Fizgig looks for strokes that stand out from the colours around them before falling back to the whole box.
  - New "Watermark detector sureness" slider (0.10 to 0.90, default 0.50, up from a fixed 0.30) leaves out boxes the detectors aren't sure of, which cuts out most misfires.
- **Detail crops:**
  - Crops you delete stay deleted: Start won't make them again at any size, and the full picture keeps training. Deleted at one size, that crop is set aside in the other sizes' "removed" folders too (nothing is thrown away). Put it back (or Undo in the Captions tab) and it trains again.
  - Image Prep makes them only at the Target megapixels you pick (Start makes any other size a run needs).
  - "Copy the picture's caption to its crops" works for detail crops too, including crops Start makes at other sizes.
  - Each size gets its own caption, written for what that crop shows, instead of a copy of the biggest one.
  - Start captions crops with the model picked on the Captions tab (JTP-3, WD, Florence-2 or Qwen3-VL), not always Qwen3-VL.
- **Auto memory planning** (precision and Blocks Swap) is more careful: it counts your LoRA rank and optimizer (a big rank or an optimizer that keeps more per weight, like AdamW, Prodigy or SOAP, gets room planned for it), and it learns from your runs by comparing each epoch's actual peak VRAM with what it expected (it only trusts a lower figure by up to 1 GB). New Preferences → Training memory → "Safety margin (GB)", 1.5 by default: raise it if other apps share your card, lower it to swap fewer blocks. Fine-tunes use it too.
- **Adaptive LR:**
  - PoLoRA's now works just like Muon's: Min and Max are real learning rates (picking PoLoRA fills in 1e-2 to 4e-2), the run starts in the middle, the Learning Rate box greys out and magnitude_scale stays as you set it. Per-image LR scales PoLoRA's rate the same way.
  - With validation on, it waits to raise the LR while training loss is spiking (3%+ above its recent average). If loss jumps 10%+ within 2 epochs of a raise, it goes straight back to the previous LR.
  - "Follow validation loss" is renamed **Adaptive LR: use validation loss**, with clearer help: on, it uses both losses (validation leads, training confirms); off, smoothed training loss only.
- **Loss graph:** it marks the epoch with the lowest average loss (green dot, plus "lowest epoch loss" under the graph) instead of the lowest smoothed loss, which could hide off the edge. Its "PoLoRA factor size" switch now controls the log too (every 25 steps when on).
- The Masks preview says how much of each picture is learned or left out, instead of "100% masked" on a picture with no watermark.
- Validation loss at Batch size above 1 checks just the pictures you picked, each on its own.
- The Style score in the Samples gallery is on by default, so every preview shows its style %. Switch it off for a run and it stays off.
- The "Learning rate is above/below the usual range" Start check now only shows for AdamW optimizers.
- Arrow keys make sense: in the picture viewer ← → step through pictures like the ‹ › buttons (now centred), and on training previews ↑ ↓ show the same prompt an epoch earlier or later.
- Small comforts: Preferences opens at the top of the page; the app window's Save window for epochs opens in the folder you used last, even after a restart; and opening Runs & outputs during training jumps straight to that run.
- Links and installs now point at Fizgig Advanced (Murf-Code/Fizgig-Advanced, branch fizgig-advanced): the README and docs/INSTALL.md clone commands (including the "installed from a ZIP" fix), the guides, About's release notes, the News card and the contributing, issues and discussions links. Pods built from this repo default to the fork (FIZGIG_REPO / FIZGIG_REF still override it). The RunPod and Vast.ai templates belong to upstream, so set `FIZGIG_REPO=https://github.com/Murf-Code/Fizgig-Advanced.git` and `FIZGIG_REF=fizgig-advanced` there. The update check looks for Fizgig Advanced releases (advanced-1.x), not upstream's v9.x. Credits, upstream release history and upstream-hosted model files still link to upstream.

## Fixed
- Image Prep failing to load on Python 3.11 and older (an f-string in the Face Crop Only summary). Thanks @tsk5cb-cmyk for the fix.

## From upstream Fizgig (v9.0.3 to v9.5.0)
Thanks to @shootthesound and everyone who contributed upstream. This release brings in everything from Fizgig v9.0.3 to v9.5.0, including:
- **Training maps** (v9.2.6 to v9.3.2): a Training maps button beside Per-image loss shows each picture with a heatmap of what's still wrong, what's been learned and where the LoRA acts, across the epochs. There's also a trigger-word attention map on every still-picture model except SDXL.
- **Studio maps** (v9.4.0): the same heatmaps over Studio's Tweaked picture for any LoRA, showing where it acts, what your tweaks change, the trigger word's attention and each block group.
- **Captioners** (v9.5.0): JoyCaption Beta One and the WD EVA02 tagger on the Captions tab, a Captioning models card in Preferences, your own extra instructions for a captioner, and detail crops that follow the picture's mask.
- **Muon everywhere** (v9.0.8 to v9.2.0): Muon is the default for every picture model's presets, with a faster batched Muon. Adaptive LR learns a ceiling.
- **Speed and setup** (v9.2.x): PyTorch 2.15 with CUDA 13.4, faster batched training (#221), LoKR 4–6× faster (#307), a health check in the updater, and no telemetry.
- **Fine-tune optimizers** (#312), MOSS-Image Turbo previews and the Turbo Train preset, and many smaller fixes. Thanks to @spinolover (#302) and @MrB4000 (#305).

The full upstream notes are in [Fizgig's releases](https://github.com/shootthesound/Fizgig/releases).

## For developers
- `scripts/bench_polora_step.py` times PoLoRA's optimizer step on real LoRA shapes (Krea 2, Anima, MOSS-Image, SDXL) at ranks 8 to 64, with and without compile: median time, spread, peak VRAM, where the time goes, saved as JSON. `--optimizer muon` times Fizgig's Muon the same way (BatchedMuon, or per-matrix Muon where that's missing), split into Newton-Schulz and the rest.
- Fixed a test that failed about half the time (the JTP-3 tagger's small network check): the copied-in JTP-3 code left part of the network unset before its weights loaded. Tagging itself was never affected.

## Update
Close Fizgig, then run `git pull` in your Fizgig folder on the `fizgig-advanced` branch.
