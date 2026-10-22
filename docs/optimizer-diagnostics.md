# Optimizer capture and replay

The application can capture one ordinary fused projection/Adam update. A private
runner replays the original kernel and an opt-in streamed-Adam candidate. The
candidate is not selectable in normal/Pure Training and has no accepted
performance benefit. Numeric behavior must be checked on the actual device;
cross-device bitwise equivalence is not assumed.

## Capture

Pause and join training, then select `Capture One Optimizer Step and Pause`.
This executes one eligible training step, not a read-only export. The initialized
model/configuration is used. ADC, reset, final and active benchmark steps are
rejected without advancing to a different window.

The worker copies pre-update state, runs the original fused dispatch, completes
normal advancement, exports under `benchmark_optimizer/` and stops. Resume
starts the next step. Output names are unique. Failure reports whether prepare,
Adam, GPU completion and iteration advancement occurred; completed Adam is not
rolled back. Uncertain submitted state requires reinitialization.

## Resources and format

Pre-state is 840 bytes/Gaussian, post-reference736 bytes/Gaussian. GPU pre-clones
and streamed post output are protected by resource checks: device memory-budget
support, clone headroom plus256 MiB reserve,128 MiB available RAM, file payload
plus64 MiB disk headroom and an8 GiB file limit. Unknown resources reject capture.

The little-endian `V3DGOPT1` format has a928-byte header, camera offset560
and actual push constants offset800. It stores exact device/compiler/module
identity, dimensions/counts/modes/steps, lengths/strides and FNV checksums.
The checksums detect corruption, not hostile modification.

| Block | Data | Stride |
| --- | --- | ---: |
| 0/1 | Pre parameters / Adam | 240 / 480 |
| 2/3 | Projected Gaussians / gradients | 48 / 48 |
| 4/5 | Pre densification / tile ranges | 16 / 8 |
| 6/7/8 | Post parameters / Adam / densification | 240 / 480 / 16 |

One reusable1 MiB coherent staging buffer operates on the compute producer
queue. Explicit barriers cover producer→copy, copy→update and update→export.
Unrequested steps do not add capture allocations or downloads.

## Replay and candidate

Linux commands from the repository root; Windows adds `.exe`:

```bash
build/test/Release/training_optimizer_gpu_tests --synthetic --require-validation --report optimizer_synthetic.json
build/test/Release/training_optimizer_gpu_tests --replay path/to/step.optimizer.bin --require-validation --report optimizer_replay.json
build/test/Release/training_optimizer_gpu_tests --synthetic --candidate-r1 --require-validation --report optimizer_candidate.json
```

Use fresh report names. `--require-validation` requires both core and
synchronization validation. Real replay requires matching UUID, driver/API,
compiler/debug configuration and shader fingerprint. Synthetic checks use
private inputs and do not establish target-scene performance.

All mutable arrays, including parameters, m/v and statistics, are restored
before each invocation. Post-reference and repeat comparisons are bitwise.
Non-finite data, invalid moments, identity mismatch or numeric failures stop
the runner. Fixed projected inputs in sequence tests are not recomputed training.

The streamed candidate retains typed AoS, FP32, group256 and dense Adam. It
finishes projection/SH derivatives before loading/updating each moment pair.
Source lifetime is not evidence of lower native registers or better speed.

Candidate replay uses the same `--candidate-r1` flag. It runs sequence checks
before timing and measures A/B/A,2 warmup and8 samples per phase, with complete
separately timed restore. Comparisons are outside timed phases. All samples
are retained. Outlier/drift and gain gates are diagnostic admission criteria,
not guarantees of full-training benefit.

A separate resource window accepts `--profile-r1-seconds 1..30` and
`--profile-variant baseline|stream-adam`, with candidate replay selected.
It remains correctness-gated and reports profiling-only status. Inspect native
register/shared/spill and dispatch-specific counters separately from timing;
unavailable evidence remains null.
