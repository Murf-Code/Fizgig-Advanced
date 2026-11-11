# Fizgig Advanced 1.5.2

A small patch on 1.5.1. Auto Blocks Swap now learns from your own runs, so its plan gets closer to what your card actually uses. Thanks to Peter (shootthesound) for measuring the planner across 10 families and 4 card sizes in [issue #2](https://github.com/Murf-Code/Fizgig-Advanced/issues/2).

## Fixed
- **Auto Blocks Swap learns from your runs.** After a run, the next plan uses the peak VRAM that was really measured, including during previews. It applies that in full rather than capping it at 1 GB, and still uses it when the plan picks a different swap or rank. The same model at another precision counts at half. Once a run on your card has been measured, the 0.8 GB small-card reserve is dropped, because the measured peak already includes it. The `[vram] plan` line now shows the predicted peak. On a 12 GB 4070 Ti, Krea 2 went from swap 19 to swap 17 on the second run. It peaked at 9.3 GB against a prediction of 9.29 GB.
- **No more NF4 that can't fit.** When nothing fits, Auto now picks INT8 with more blocks swapped, unless NF4 really needs less memory. INT8 with a few blocks swapped also comes before dropping to NF4. This stops Krea 2 and Fibo 2 running out of memory on NF4 at 12 GB, rank 64.
- **Fibo 2 crash with block swap.** At Blocks Swap 12, 20, 28 or 36, Fibo 2 crashed with "mat2 is on cpu", because some of its blocks' cross-attention weights were left on the CPU. All weights now move properly.

## Changed
- In the Optimizer list, Automagic v4 now sits right after Automagic v3.
