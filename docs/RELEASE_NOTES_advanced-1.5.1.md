# Fizgig Advanced 1.5.1

A small patch on 1.5. Auto Blocks Swap now plans enough memory for 12 GB cards, so a Krea 2 run no longer spills into shared GPU memory. Remove Tool uses LaMa instead of Telea's smear, and the Image Editor has clearer names plus a button to put a picture back.

## Fixed
- **Auto Blocks Swap on 12 GB cards.** Auto was leaving out some of what a run actually holds, so it swapped too few blocks and training spilled into shared GPU memory. On a 12 GB 4070 Ti, Krea 2 at 0.25 MP went from Auto 17 (22–66 s/step) to Auto 20 at about 4 s/step. It now counts the EMA copy, the optimizer's own memory (sized to your rank), compile, context LoRAs (from the file's real rank and layer count) and a small reserve on cards under 20 GB. The `[vram] plan` line shows the breakdown, for example `Furry_Parts.safetensors (rank 16, 264 layers) +0.11`. If steps still run far slower than they should, the log warns once and tells you to raise Blocks Swap and check Shared GPU memory in Task Manager. The Auto line now says when Blocks Swap is your own setting, and the warm-up note no longer says "Still speeding up" when it isn't.

## New
- **Remove Tool now uses LaMa**, an AI inpainting model (about 206 MB, downloaded the first time you use it). Telea's smear is gone, so bigger watermarks and text come out clean. It runs on the GPU when there's room, and on the CPU while you're training. If the model can't download or load, Remove shows an error and leaves the picture alone.
- **Restore Original Image** in the Image Editor. When a picture was backed up before Remove, click once, then Confirm, and the original picture (and its training mask, if one was saved) is put back. The backup is kept, and Ctrl+Z undoes it.
- **Hover text on every Optimizer option**, including each optimizer and scheduler in the lists. The help under the boxes is shorter, so it all shows without a click. Compile's line is: "Speeds up PoLoRA after a 30-second warm-up. Training results are the same. Untick if it errors or runs out of VRAM."

## Changed
- The Masks card's fourth step is now **4. Image Editor** (was 4. Edit masks). Its picture tools are **Remove Tool** and **Crop Image**.
