# Experiment 5C — Learning-rate calibration on corrected captions

New benchmark condition: user adds anthro to applicable captions. Do not pool losses with 5B as if captions were unchanged. Keep image set, holdout, model, prompts and training seed fixed. Do not edit captions during the queue.

Eight independent fresh runs: unrestricted Replay at 0.00020, 0.00025, 0.000264, 0.00030, 0.00035, 0.00045; historical seeded Random (14 blocks between eight-step full probes) at 0.00045; Adaptive Gradient at 0.00045. Random is a fixed-count historical comparator, NOT the adaptive count-matched 5B control. All Muon, match_rms_adamw, momentum .95, Nesterov, NS5, decay .01. Same rank 32, alpha 32, INT8, 22 swaps, seed42, 29 training images, three held out, five epochs (145 updates), four previews per epoch and baseline. Strict deterministic execution, per-update adapter hashes and actual gradient masks, per-probe factor telemetry, raw and EMA validation, FP32 epoch snapshots. No optimizer implementation or selection rule changes.

Compare final and best raw/EMA validation, timestep losses, image fidelity, and unrelated-prompt drift. New baseline and all LR comparisons are within this queue. Six LR choices are a coarse sweep, not proof of optimum. Additional seeds/datasets and checkpoint geometry analysis remain necessary.

Start a fresh Test Mode queue only after caption edits; preserve 5B and 5C checkpoints locally. Export global-replay-results.zip. 80 FP32 research snapshots total (approximately 34.4 GB decimal), plus normal exports. Do not delete prior results.
