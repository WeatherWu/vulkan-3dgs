# Bounded real-training smoke runs

The application supports a non-persisted diagnostic step limit:

```text
--training-smoke-steps N --training-smoke-report NEW.json
```

Both options are required together. `N` is an integer from 1 to 29999. Smoke
runs require `--validation on`, available core and synchronization validation,
a fresh model at iteration zero, and the normal 30000-step Random schedule.
The application enables synchronization validation when smoke options are
present and counts error callbacks from the shared application instance,
including presentation. Missing validation support or preexisting validation
errors prevent starting a smoke run.

## Run a matched comparison on the RTX 3090

Build and run from the repository root on Linux:

```bash
cmake --build build --config Release

VULKAN_3DGS_FASTER_SORT=0 build/bin/Release/vulkan-3dgs \
  --gpu "RTX 3090" --validation on \
  --training-smoke-steps 3200 --training-smoke-report smoke_A_3200.json
```

In the UI, select the same dataset, downscale and seed used for both runs.
Use 3DGS Random, Fixed16, Tensor Subgroup/Global, RGB Baseline and mixed SSIM
for the sorting experiment. Preserve optimizer and densification settings.
Click **Start Training**. The CLI configures a boundary; it does not automatically
start training. Do not run Fixed Benchmark or optimizer captures beforehand.

At the limit, the worker pauses automatically after the complete normal step,
including any ADC/adoption, optimizer policy and scheduler advancement. It
inspects final loss/model finiteness and saves the report. The log contains
`Training smoke report saved:` and the UI reports a smoke pause at the reached
iteration. The model remains in memory and can be exported manually; Pure mode
does not treat this diagnostic stop as natural completion or automatically
export its final PLY.
The UI's normal total can still display 30000; the smoke status and CLI boundary
control the diagnostic pause independently.

Close the application after the first report is saved. Start a fresh process
for the candidate, preserving the same UI configuration:

```bash
VULKAN_3DGS_FASTER_SORT=1 build/bin/Release/vulkan-3dgs \
  --gpu "RTX 3090" --validation on \
  --training-smoke-steps 3200 --training-smoke-report smoke_B_3200.json
```

Click **Start Training** again. Report paths must be new; existing files are
never overwritten. A resumed nonzero-iteration model is rejected for smoke
runs. On Windows, use the corresponding `.exe` and set the same process
environment variable before launch.

## Training and report contract

The diagnostic limit does not change the scheduler's total iterations, learning
rate schedule, SH degree intervals, ADC/reset conditions or normal final-step
branch. It lives in the controller's worker loop and is checked only after
`trainStep()` returns. Normal application runs without these options retain
their existing policy and do not allocate a smoke recorder or scan model state.

The report is reserved before the first smoke step. Initial state fingerprinting
reads active parameters, both Adam moments and densification statistics before
the worker loop. Final finite inspection reads loss values and raw parameters,
including activated scales, after synchronizing diagnostic transfers; it does
not add a training step or validate final Adam-state equivalence. Initial
fingerprinting and final inspection are outside the reported worker-loop wall interval;
ordinary training validation remains part of normal step execution.

Schema `training_smoke_v1` retains device/build/shader identity, the immutable
run configuration, initial state fingerprint, every completed step's frame,
Gaussian/item counts, actual sorting path, optimizer/ADC/reset/resize events,
active SH degree, CPU step time and numeric-validation sample iteration. Loss
values from interval validation are explicitly associated with that validation
iteration; they are not described as a fresh per-step loss. Final loss/model
finite checks and final accumulated CPU/GPU profiling arrays are separate.
Profiling stage indices follow `TrainingCpuProfileStage` and
`TrainingGpuProfileStage` in `training_types.hpp`.

Numeric non-finite data, an unexpected Faster-sort fallback or a Vulkan error
stops the smoke and preserves a partial/error report when possible. A normal
user pause or window shutdown also produces a partial report. Unknown elapsed
time/error counts in an aborted recorder are null. No rollback is performed.

`SMOKE_LIMIT_REACHED_NOT_QUALITY_OR_SPEED_ADMISSION` means the requested boundary
was reached with no reported validation errors and a passing final finite scan.
`STOP_OR_PARTIAL_SMOKE` requires inspection of the reason and error fields.
Coverage of ADC, SH changes and opacity reset follows the recorded effective
configuration and events; reaching a step number alone does not establish that
those events occurred. A 3200-step run covers their default early boundaries.

These validation-enabled, instrumented runs establish bounded lifecycle evidence.
They do not establish training quality, full-training speed or an end-to-end
gain percentage. Keep both raw reports for review; uninstrumented timing and
full-training/held-out quality assessment are separate decisions.

## Shared depth-attachment synchronization

The presentation framebuffers share one depth image. The render-pass external
dependency covers early and late fragment tests and depth writes, as well as
color attachment access. This orders the previous pass's depth store operation
before the next clear/layout transition. `STORE_OP_DONT_CARE` and disabled splat
depth writes do not eliminate the attachment synchronization requirement. See
the [Vulkan synchronization examples](https://docs.vulkan.org/guide/latest/synchronization_examples.html).

An explicit headless regression exercises eight graphics submissions with shared
depth storage and the application's actual render-pass builder:

```text
build/bin/Release/graphics_depth_reuse_gpu_tests --report depth_reuse_check.json
```

The negative control intentionally uses the incomplete dependency and must
reproduce depth WAW errors; the fixed pass must have zero validation errors.
The report separates those expected control errors from fixed-path errors. The
test creates no window and advances no training. It is not an automatic
hardware-dependent CTest. Smoke startup still rejects accumulated application
validation errors; after applying synchronization fixes, restart the application
to begin a new validation interval rather than clearing the counter in place.
