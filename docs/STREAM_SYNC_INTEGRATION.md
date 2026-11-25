# Stream Sync integration candidate

Selective port from Advanced test/stream-sync, based on WIP 471539d. Includes checkpoint recompute and sparse backward ordering, ring and host memory lifecycle, replacement hook retirement, and fail-closed VRAM reclaim with regressions. Existing Faster Block Swap preference/defaults are preserved; no automatic calibration, profiles or experimental strategy preference. Ring sweep environment override is retained for diagnostics; normal default remains two slots. CPU CI does not validate actual CUDA transfers. Upstream PR #403 overlaps with this port; upstream #405 parked preview adapter memory fix remains a separate compatibility task.

## Upstream credits

- [knoic, PR #403](https://github.com/shootthesound/Fizgig/pull/403): frozen-base send-only streaming and checkpoint recomputation.
- [shootthesound, #405 fix](https://github.com/shootthesound/Fizgig/commit/5b3b8e479183af6f2f24e6e4625336838d67ba5c): parked preview adapters; reported by [spinolover](https://github.com/shootthesound/Fizgig/issues/405). Adapted to Advanced's shared, classic and send-only paths.
- [shootthesound, pinned RAM benchmarks and policy](https://github.com/shootthesound/Fizgig/commit/2a76577b4): opt-in rather than unconditional auto pinning, informed by Murf-Code's benchmark. Advanced's own staged host-budget logic is retained.

- [shootthesound, preference and guard](https://github.com/shootthesound/Fizgig/commit/bcd6333df): user opt-in and environment precedence, with Advanced's conservative send-only allocation budget retained and a classic swap guard. [Persistence correction](https://github.com/shootthesound/Fizgig/commit/f246dbe80) and [distinct explanations](https://github.com/shootthesound/Fizgig/commit/da5b9e941) adapted to Advanced's existing Preferences layout. Faster Block Swap's existing Advanced default is unchanged.
