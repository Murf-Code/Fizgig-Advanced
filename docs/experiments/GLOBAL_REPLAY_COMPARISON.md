# Global Replay comparison (test/global-replay-comparison)

Select a Krea 2 dataset on Start, leave Test Mode enabled (the default on this branch), optionally enter a fixed preview prompt, and click the existing Add to queue button. It queues all four runs for exactly three epochs each. Uncheck Test Mode to restore ordinary queue behavior. These are real fresh LoRA runs on the normal training queue, not the stream-sync microbenchmark. No run resumes another. Each case has a separate output folder under replay_comparisons.

| Case | Method | Adapter targets | Selective |
| --- | --- | --- | --- |
| standard | Standard | All normal Krea 2 adapters | No |
| standard-matched | Standard | Main transformer blocks 0–27 | No |
| replay | Global replay | Main transformer blocks 0–27 | No |
| selective | Global replay | Main transformer blocks 0–27 | Yes |

All cases use INT8, 22 swaps, rank/alpha 32, Muon, LR 0.00025, 0.25 MP, batch/accumulation 1, seed 42, compile off, fixed hold-out validation (four images when available, four timesteps, seed 42), and save each epoch. Optional previews share prompt and seed 1234. Automatic hold-out selection is deterministic for an unchanged dataset. Leave dataset, models, GPU settings and preferences unchanged throughout. Compatible production caches may be reused; exclude cache/model startup and preview time from steady-state step comparisons. Automatic OOM recovery is disabled for these comparison cases so a failed case cannot silently change precision or swap count. The ordinary queue may advance past a failed run; check all four completed before drawing conclusions.

Compare standard-matched vs replay for the replay algorithm, replay vs selective for selective checking, and standard vs replay for the user-facing tradeoff. Compare validation loss at equal updates and wall time, step time, peak VRAM and system RAM, fixed-seed previews and selective probe logs. A run with 28/28 active at every probe has not actually exercised block skipping. A short experiment is a smoke test, not proof of final image quality. Repeat with longer training and several seeds before generalizing.

The UI preflights all four cases before adding them; the existing queue starts them sequentially (or waits if held). Queue cases are not editable; remove and generate a new comparison instead. The old stream-sync diagnostic is no longer offered in this branch's UI; it remains available on test/stream-sync.
