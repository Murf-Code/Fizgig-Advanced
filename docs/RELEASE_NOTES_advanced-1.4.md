# Fizgig Advanced 1.4

Fizgig Advanced 1.4 is up to date with **Fizgig v9.0.2**. It covers everything added since 1.3.1 (30 Oct 2026). Prodigy, Muon, Adaptive LR for every fixed-rate optimizer, the step-0 validation baseline and the Style score badge were already in 1.3.1.

## Added today (3 Nov 2026)

### Up to date with Fizgig v9.0.2
Everything from upstream Fizgig v8.9.96 to v9.0.2 is merged in, including:
- **Detail crops** for style LoRAs.
- **Transparent pictures** handled properly (laid on white where the model has no alpha VAE).
- **RunPod REST v2**: pod calls moved off GraphQL, which stops in January 2027.
- Caption dropout, Start at a smaller size, Start checks, the Run summary card, the Captions Table view, the sidebar meters, and many Helpers / Pods fixes.

See upstream's `docs/RELEASE_NOTES_v8.9.96.md` to `docs/RELEASE_NOTES_v9.0.2.md` for the full list.

### PoLoRA (new optimizer)
- Layers are paired by name. If none match, you get a plain error.
- `compile` falls back to plain mode if it can't compile.
- Effective magnitude = magnitude_scale × per-image factor × Adaptive LR factor, logged once per step.
- A "PoLoRA factor size" line on the loss graph.
- Picking it fills in LR 2e-2 and magnitude_scale 3 (2.5–3 works well at 2e-2). PoLoRA settings are cleared when you switch away.
- Its state is saved for resume, and there's no "LR very high" warning at 2e-2.

### Automagic V4 (new optimizer)
- Ostris's newer Automagic, vendored from ai-toolkit (MIT). It sets its own rate, so Adaptive LR stands down.

### Training
- **Two-click Stop:** Stop asks you to confirm with a second click. It resets after 30 seconds.
- **Caption dropout** now uses upstream's setting (Off / 0.05 / 0.10). At Batch Size 1 it works like upstream. Above 1, each picture still drops its own caption using a cached empty caption. Old percent values are moved over for you.
- **Style score:** upstream's 🎨 Style score is the same one we made, so the fork's duplicate is gone.

### Image Prep fixes
- **Detectors are trusted.** Each word goes through the detector chain with SAM last. The 60% "too big" rule now only applies to SAM.
- Detector-only crops drop the padding instead of being skipped.
- The log names the model that found each box.
- **Find this `OR` that:** type `ass OR pussy` (or `|`) to find either one.
- **Separate crop for each word** tick.
- Compact detectors table and a clearer converter message.

### Masks (with ADetailer-style YOLO detectors)
- **Mask watermarks too** tick on the Masks card ("Make masks"). Two YOLO watermark detectors (Watermark and NSFW Watermark) find watermarks, signatures and text, and black them out of each mask, so masked training ignores them.
- They're merged with any mask a picture already has. Image Prep's watermark masks go in a `masks` sub-folder, and training uses that one first.
- **Just hide the watermarks**, a new choice under "What the masks show". Each mask is all white, with only watermarks, signatures and text blacked out. It uses the same watermark detectors (your own Watermarks models too, switched-off ones skipped) and needs no character finder or SAM, in either mode. Mask watermarks too shows ticked while it's chosen. If a picture already has a mask (say from a character-mask run), the watermarks are blacked out on top of it, so the two merge; with no watermark found, that mask stays as it is. With Replace masks already there ticked, old masks are ignored: a fresh mask minus the watermarks, and none (the old one removed) where no watermark is found. A picture with no mask and no watermark trains as normal. Preview on 2 random pictures shows the merged result and changes nothing.
- **Trace watermark shape with SAM 3.1**, a new tick under Options. It shows when Mask watermarks too is on or Just hide the watermarks is chosen. SAM 3.1 outlines the watermark inside each box the detectors find, so only the letters or logo are blacked out, not the whole box. Where SAM finds nothing, the whole box is used. It's on by default once SAM 3.1 is downloaded. Fast on a GPU; a few seconds a picture on the CPU.
- Fixed: Image Prep's settings could forget some mask choices after changing another setting. They're kept now.
- Watermark boxes over 10% of the picture are dropped, because the detector sometimes fires on big blank areas.
- The detectors run with onnxruntime only, so nothing new needs installing. Missing models are skipped, and the log says so.
- See-through PNGs are laid on white for Make masks (automatic and SAM), SAM's preview and the Look filter, instead of on their hidden colour.

### YOLO detectors section
- The YOLO body-part and watermark models now sit in their own **YOLO detectors** section in Image Prep. It starts closed and remembers if you open it.
- **On/off tick for each model:** a model that's off is greyed out and skipped.
- **Add model:** add your own YOLO or ADetailer `.onnx` (a file or its folder), with an optional name, as Body parts or Watermarks. Added models run after the built-in ones and before SAM 3.1. `.pt` files point you to the converter.
- **No paths box:** Choose... and Upload... set each model's file, and the row just shows the model name. Paths you saved before still work.
- **Remove** needs two clicks ("Confirm remove", which goes back after 5 seconds). It only takes the model off the list and never deletes your file. Built-in models can only be switched off.

### Name and file picker
- The app is now called **Fizgig Advanced** in the browser tab, the header and the About window. About still credits Peter Neill.
- The repo is renamed to [Murf-Code/Fizgig-Advanced](https://github.com/Murf-Code/Fizgig-Advanced). Old links redirect.
- Every **Choose...** button opens the normal Windows file and folder window. If Fizgig is opened from another computer, or that window isn't available, you get Fizgig's own picker instead.

### Previews and tabs
- **Sample now** sends a prompt you just typed, and the override sees the sample picture.
- While training, the Training and Samples tabs open at the running run.

### Tidy-up
- Leftover merge markers fixed in train.py.
- Unused imports from merges removed.

## Also since 1.3.1
- **Adaptive LR** judges a plateau on the last 3 epochs' average loss, so one jumpy epoch can't drag the LR to the floor.
- **Image Prep never changes your training folder.** Outputs go to `resized/`, `crops/` and `skipped/` sub-folders, and training uses the resized copies (no duplicates).
- **Fit all words in one crop** option.
- The "Your originals" card says where outputs go (no more keep/replace choice).
- **Body-part detector chain:** Fdetailer, booru_yolo, All-in-One, assDetailer, then SAM last. Detector paths are shown on the page, and there's a converter script.
- YOLO's boxes beat SAM's for the same word.

## Sent upstream
These Fizgig Advanced features have been offered to upstream Fizgig as PRs:
- [#289](https://github.com/shootthesound/Fizgig/pull/289) Fast (send-only) block swap
- [#290](https://github.com/shootthesound/Fizgig/pull/290) Loss graph: Epoch average, Running avg, LR marker, validation line
- [#291](https://github.com/shootthesound/Fizgig/pull/291) Fixed validation loss: the Image Validation card
- [#292](https://github.com/shootthesound/Fizgig/pull/292) Prodigy, Muon and Automagic V4
- [#293](https://github.com/shootthesound/Fizgig/pull/293) Pause, then Save & stop
- [#294](https://github.com/shootthesound/Fizgig/pull/294) Two-click Stop

## Update
Close Fizgig, then run `git pull` in your Fizgig folder on the `fizgig-advanced` branch.
