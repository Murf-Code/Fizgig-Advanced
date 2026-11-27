# LB-H — maximize pure LittleBit

User scope decision: pure binary first. LB-G hybrid is preserved, parked, and not expanded. See LB_G_HYBRID.md and RESEARCH_STATUS.md.

## H1 convergence screen (a737af51, regression tests 27bcd0dd)

One-bit ceiling, same matched two-path E3 initialization, total rank ceiling 2048, original .2 weight/.4 forward/.4 input-gradient pooled objective, BF16 rounded scales and SmoothSign100. Use exact same old fit/dev/prior-confirmation capture partitions as F/G; all scoring panels are previously inspected. Fresh capture validation is a separate required gate, not implemented here. Suggested pilot key14 only (hardest), then key0; MLP is optional later because its total rank cap underfills one-bit budget. No hybrid, no new codec, no trainer changes.

| Arm | Sign learning rate | Sign warmup | Schedule |
|---|---|---|---|
| H0 | none | — | E3 initialization |
| H1 | 1e-4 | 0 | F2 constant |
| H2 | 1e-4 | 20 | F5 constant |
| H3 | 1e-4 | 20 | F5 through 100, cosine to 5e-6 at 500, floor to 1000 |
| H4 | 5e-5 | 20 | constant |
| H5 | 1e-4 | 20 | F5 through 100, freeze signs from 101, scales continue |
| H6 | 0 | — | scale-only |

All six optimization arms 1000 updates, fixed checkpoints 0/20/50/100/200/350/500/750/1000, scale LR .001 unchanged. No extra sweep of ranks or beta yet. Select checkpoint by prior development only, record prior confirmation for selected candidate. Save in-progress history at each checkpoint and completed candidate immediately; report is not a resumable checkpoint or deployable artifact. Record exact tiled weight error, sign gradient norm and current LR, cumulative flips, net flips, scale statistics, payload and decoded size, theoretical MAC count. Compare first 100 updates to LB-F/G; explicitly check long-horizon gains, oscillation, and weight/action tradeoff. Do not interpret reduction in local error as LoRA learning retention.

## Checks and limits

On PyTorch 2.10 CPU, reconstructed the exact rate/optimize excerpt (4062 characters, FNV-1a 1319198947) and imported F components excerpt (299 characters, FNV-1a 2913438331), matched against committed code; used copied qat_core mathematical subset. Synthetic 37x31 dual paths ranks 7+3, five token probes: constant/cosine prefix identical through 100, freeze preserved sign shadow tensors and cumulative flip counts while scales changed through 105, schedule midpoint/floor correct. These checks executed on isolated excerpts, not full repository pytest. Committed tests also compare directly to F5; that complete suite and CUDA runner remain unexecuted here. No new independent capture bank, actual VRAM or portable LoRA results.

## Next gate

Review H1 results, then narrow sign schedules and use fresh capture bank before promoting any winner. Investigate unequal binary ranks, activation-aware initialization, latent rotation, sign-specific optimizers only where evidence warrants. Preserve scope: maximize pure binary, validate complete block and actual LoRA transfer, then return to hybrid.
