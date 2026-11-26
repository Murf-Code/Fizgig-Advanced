# Muon-ME closed-loop Krea 2 comparison — 2026-11-25

Source: user-supplied matrix-geometry-report(6).json, comparison(6).log and results(3).zip, run group 210823-af73aa; archived report equals standalone report. Queue commit a889f3c. All six runs finished (286 updates, sum run wall 4565.87 s excluding inter-run gaps). 13 training images, 2 fixed holdout images x 4 noise levels, seed 42, rank/alpha 64, 264 modules, 234.6M params, INT8 base, 0.25 MP target (actual buckets 448x544, 496x496, 528x464), batch/accum 1, constant LR 0.0002, EMA off. No optimizer state saves. Four quality arms 65 updates each, two stress arms 13 each.

| Arm | Swap | Final validation | Reported training loss | Training ms/step | Warm optimizer mean ms | Device peak GiB as logged | Max sampled post-step reserved GiB | Shared GPU peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Muon A |22|0.18244|0.1730|2556.4|68.8695|11.33|8.875|1.08|
| Muon repeat |22|0.18188|0.1672|2664.3|71.2338|11.30|8.85352|1.08|
| ME shape-local |22|0.18213|0.1652|2669.1|72.5781|11.64|8.86133|1.08|
| ME 128 |22|0.18167|0.1739|2649.7|58.2386|11.21|8.54492|1.08|
| ME 128 stress |16|0.18263|0.1685|19175.9|75.007125|11.98|10.90625|1.727|
| Muon stress |16|0.18248|0.1700|16391.9|983.42457|11.98|10.99023|1.518|

Validation trajectories (baseline and epochs):
- A: .18260 .18256 .18248 .18250 .18227 .18244
- A repeat: .18260 .18259 .18240 .18236 .18193 .18188
- shape-local: .18260 .18257 .18267 .18304 .18261 .18213
- ME128: .18260 .18283 .18291 .18236 .18220 .18167
- stress ME: .18260 .18263
- stress reference: .18260 .18248

ME128 warmed optimizer mean 16.86% lower than reference average; whole training step 1.51% slower than reference average (within observed A/A variation, not a demonstrated speed gain). Max sampled reserved 327 MiB less than reference mean, not a whole-run allocator peak. Device peak 0.105 GiB lower than reference mean. Quality gap to best control 0.00021, smaller than A/A endpoint gap 0.00056; no superiority or equivalence claim. 112 previews in archive; baseline image bytes identical across four quality arms; visually inspected baseline + 16 final images: coherent characters and landscape in all arms, pose/style variation even between original Muon controls. No blind scoring or checkpoint tensor comparison; archive contains no safetensors.

16-swap both complete but not acceptable clean-fit runs. Device peak 11.98/11.99 GiB, shared GPU memory increases above 1.08 GiB baseline, explicit spill warnings for both. ME128 optimizer still 75 ms vs reference 983 ms, but total step 19.18 vs 16.39 s; ME128 7.24x slower than own 22-swap run. Evidence points to pressure outside optimizer as major bottleneck, exact phase peak and residency not directly measured. Zero reported allocator retries do not exclude driver fallback. Do not recommend 16 swap rank64 on this evidence.

Next: no repeat of these six arms required. Retain ME128 as experimental candidate, 22 swap known working, short 20/21 swap probes only if investigating speed threshold. Add phase-resolved peak allocated/reserved and host/device counters; validate synchronous targeted activation offload and/or INT8 momentum separately with matched control. Consider true 1 MP rank16 with enough swap and source resolution; do not add independent savings. Keep existing adaptive TSD queue. No production default changes.
