# 7C/7D — joint continuous attenuation and movement compensation

Four one-epoch runs, seed 42, same 13 training and two holdout images, rank 32, native Muon at LR .00045, fixed validation seed 42. No additional forward probes, samples or optimizer steps. All 28 blocks compute gradients and update momentum. Decoupled decay unchanged. After orthogonalization, use the actual LR-scaled native learning proposals; keep the same momentum state for the chosen endpoint.

* full: native unrestricted proposal.
* rotate: alternating complementary 14-block parity groups, multipliers .5 and 1.
* uniform: a shared multiplier chosen to match the actual effective Frobenius movement of rotate **on its own current parent and proposal**.
* compensated: same rotating relative allocation, common scale chosen to match the actual unrestricted movement **on its own current parent and proposal**, cap 1.5.

The existing compact FP64 QR curve gives a joint norm including decay and quadratic BA terms. Measure actual committed movement; require relative gap <= .001 (absolute floor 1e-12). Stop on unreachable target rather than silently change policy. The matching does not imply equal absolute movement across diverging training trajectories. Report all 13 local budgets, block shares, coefficients, timing and final raw/EMA validation. The full arm is instrumented equally where possible but not a production throughput benchmark. CPU copies and QR cost are research overhead; no claim of compute savings. Ordinary training and legacy research queue recipes unchanged. 7E momentum and 7G samples/preservation remain separate planned work, not extra exploratory arms.
