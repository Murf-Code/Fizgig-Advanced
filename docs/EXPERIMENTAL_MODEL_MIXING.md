# Experimental model mixing in Fizgig Advanced

**Status:** Test branch `test/caption-workbench`; not a release promise. All operations are opt-in under the Studio tab.
Upstream MiniMax Frankenstein work is credited to [shootthesound/Fizgig](https://github.com/shootthesound/Fizgig/blob/master/docs/STUDIO.md#frankenstein-mode-minimax-h3-image).
This build does not merge upstream's UI and does not affect training, caption caching, validation or replay.

## 1. Mix compatible base checkpoints (MOSS / Anima / Krea 2 / SDXL)

Open **Studio > Experimental · Base checkpoint mixer**. Select a family, use **Choose…** for two checkpoint files
from that *same* family, then press **Check compatibility**. Matching blocks become sliders (0 keeps all of Base,
1 replaces that block with Donor; values in between are weighted averages). Choose a new output path and **Save merged
checkpoint**. You get an ordinary `.safetensors` model that compatible inference programs can load independently.
Unchanged tensors, text encoders, embeddings and VAE weights stay exactly as Base. Mix provenance is written in metadata.

For **SDXL/Illustrious**, both files must have the same LDM or Diffusers layout. An Illustrious checkpoint can be mixed
with a compatible Illustrious fine-tune, but unrelated SDXL-derived models may not share desirable semantic features.
For **MOSS**, try Base + SFT. For **Anima**, use compatible Anima checkpoints; the separate LLM adapter is not touched.
For **Krea 2**, use matching RAW-family checkpoints, not RAW vs distilled Turbo models.

**Compatibility is a gate, not a quality guarantee.** The preflight checks model blocks, tensor keys, shapes, dtypes,
and unsupported quantization. A file that happens to pass still may produce poor images, and there's no cross-family
MOSS↔Anima / Krea↔SDXL merging. Differences outside the selected diffusion blocks are intentionally left from Base.
This engine is for full checkpoints, **not LoRA mixing** (Studio's normal LoRA tools already handle that).

**Optimization:** The CPU export streams ~8 MiB tensor chunks and writes a new checkpoint atomically. It does not load
either model to GPU and needs no specific VRAM amount, even on a 12 GB card. It does require adequate **system RAM**,
disk free for at least one additional complete model, and time proportional to checkpoint size. Unchanged tensor bytes
are copied without PyTorch decoding, so CPU/RAM use is small; only the selected mixed chunks use FP32 interpolation.
Output never intentionally overwrites either input. Export progress and cancellation are available; when an output
already exists, confirmation is required. Don't compare CPU merge speed to a GPU live-render blend.

The CPU mixer **does not currently offer live before/after renders**: export the merged checkpoint and load it as a
normal base model to see the effect.

## 2. H3 Image Frankenstein live previews (upstream adapted)

When Studio's model is **MiniMax H3 Image**, choose experimental **Frankenstein** mode under the model selector.
The other side is **MiniMax H3 Video**, using its **Turbo LoRA at 0.75** for few-step compatibility. Both must be
configured in Preferences. Start loads H3 Image as the unchanged baseline and blends selected H3 Video blocks into
the Tweaked rendering. Block sliders are clamped to [0,1]. **✨Motion**, **✨Motion, gentle**, and **✨Reset All**
are separate mode presets; ordinary LoRA editing and save controls are hidden in this mode.

**Save blend as a new base** exports a stand-alone H3 Image checkpoint with the block-mix provenance. It is a
new full base model, not a LoRA. Budget ~21 GB of disk space for a typical pruned-int8 base. The two input models
are not modified. This path includes upstream's H3-specific int8 ConvRot and AdaLN blending, streamed swapped-block
support and chunks to avoid a second full model copy in memory.

**12 GB GPU warning:** this branch has an **experimental** auto block-swap calculation that uses int8 only, reserves
about 2.25 GB above H3 Image's loaded weights for small renders, and cuts int8 mixing rows to 512. Use **1024px**
previews initially; if planning cannot fit the streamed model it refuses instead of falling back to incompatible
4-bit blending. Upstream's documented range starts at **16 GB GPU**. This 12 GB change has **not been tested on an
actual 12 GB GPU**, and even a successful model load does not guarantee the full text encoder/preview fits.
Test NVIDIA GPU resource use and output quality before recommending it to others.

## Future architecture adapters

- **MOSS-Image**, **Anima**, **Krea 2**: presently CPU streamed export of same-family checkpoints, not H3 live mode.
- **SDXL/Illustrious**: presently same-family SDXL checkpoint export; U-Net block organization differs from the H3 DiT.
- Cross-model-family shape surgery, unmatched-block remapping, and quantized 4-bit merge are **not supported**.
  Only promote these with format-specific conversion rules, explicit tensor tests and output-quality checks.

## Verification still required

Automated regression suites cover byte-preserving copy, weighted averaging, compatibility/format refusal,
cancellation/atomic output and H3 input/slider/planner behavior. Test on real checkpoints, verify generated quality in
ComfyUI, measure RAM/disk use, test live H3 on NVIDIA and AMD, and verify Windows GPU-enabled WebView2 Studio layout.
