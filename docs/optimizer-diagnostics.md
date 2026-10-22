# Optimizer capture and replay

The application can capture one ordinary fused projection/Adam update. A private
runner replays the original kernel and one opt-in candidate: streamed Adam or
SH color-gradient reuse. Neither candidate is selectable in normal/Pure Training
or has an accepted performance benefit. Numeric behavior must be checked on the
actual device; cross-device bitwise equivalence is not assumed.

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
build/test/Release/training_optimizer_gpu_tests --synthetic --candidate-sh-color-reuse --require-validation --report optimizer_sh_color_synthetic.json
build/test/Release/training_optimizer_gpu_tests --replay path/to/step.optimizer.bin --candidate-sh-color-reuse --require-validation --report optimizer_sh_color_replay.json
```

Use fresh report names. `--require-validation` requires both core and
synchronization validation. Real replay requires matching UUID, driver/API,
compiler/debug configuration and shader fingerprint. Synthetic checks use
private inputs and do not establish target-scene performance.

Debug information is part of the module fingerprint. Rebuilds that change
source mappings can invalidate an older capture even when the original arithmetic
is unchanged. Use a capture from the matching application build; do not bypass
the identity check or edit the capture header.

All mutable arrays, including parameters, m/v and statistics, are restored
before each invocation. Post-reference and repeat comparisons are bitwise.
Non-finite data, invalid moments, identity mismatch or numeric failures stop
the runner. Fixed projected inputs in sequence tests are not recomputed training.

The streamed candidate retains typed AoS, FP32, group256 and dense Adam. It
finishes projection/SH derivatives before loading/updating each moment pair.
Source lifetime is not evidence of lower native registers or better speed.

The SH candidate evaluates the active color gradient, clamp mask and view
direction once for both position and SH-coefficient derivatives. SH formulas,
FP32 arithmetic, dense Adam, typed AoS and group256 remain unchanged. The
candidate uses a separate shader module; without its compile-time define the
original shader path remains selected. It requires no classification pass,
intermediate buffer, extra dispatch or storage-layout conversion. Compiler IR
changes do not establish fewer native instructions or a performance gain.

Select exactly one candidate. Duplicate or mixed candidate flags are rejected.
The SH synthetic suite covers 252 isolated cases and 12 optimizer-only sequence
steps, including SH degrees 0–3, clamp-boundary colors, degenerate view vectors,
mixed coefficients, no-support Gaussians and workgroup tails. Parameters, both
moments and all densification/padding words must match the original GPU
bitwise. Reports distinguish `sh_color_reuse_v1` under `sh_color_reuse` from
`stream_adam_v1` under the compatible `r1` key, with separate module fingerprints.

Candidate replay uses the corresponding candidate flag. It runs sequence checks
before timing and measures A/B/A,2 warmup and8 samples per phase, with complete
separately timed restore. Comparisons are outside timed phases. All samples
are retained. Outlier/drift and gain gates are diagnostic admission criteria,
not guarantees of full-training benefit.

A separate resource window accepts `--profile-r1-seconds 1..30` and
`--profile-variant baseline|stream-adam`, with streamed-Adam replay selected.
It remains correctness-gated and reports profiling-only status. Inspect native
register/shared/spill and dispatch-specific counters separately from timing;
unavailable evidence remains null.
