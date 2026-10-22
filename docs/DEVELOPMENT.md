# Development guide

This guide is for contributors working on the Vulkan renderer, Gaussian
training path, viewer, or application architecture. For installation and basic
UI usage, start with the [README](../README.md).

Code, CMake targets, tests, and Slang declarations are the source of truth. The
guide documents current behavior only; planned optimizations belong under
[Known limitations](#known-limitations), not in the runtime description.

## Build and validate

### Prerequisites

- CMake 3.26 or newer
- A C++20 compiler
- Vulkan-capable GPU and driver
- vcpkg with the repository manifest installed
- Slang compiler supplied by the `shader-slang` vcpkg package

Configure the project as described in the README. Reuse the existing `build`
directory for normal development:

```powershell
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

Before handing off a performance-sensitive change, also run:

```powershell
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
cmake --build build --config Debug --target quality-new
git diff --check
```

The application is written to `build/bin/<Config>/vulkan-3dgs.exe` on Windows
and `build/bin/<Config>/vulkan-3dgs` elsewhere. Generated shaders are copied to
`build/bin/<Config>/shaders/`.

### Runtime validation control

Use `vulkan-3dgs --validation off` for performance runs and
`vulkan-3dgs --validation on` for correctness checks. The default is `on` in
Debug and Release; `--validation=off` / `--validation=on` are also accepted.
Invalid, missing or duplicate validation values fail before window creation.
Existing `--gpu` selection remains independent. This is a startup-only option,
not a persisted UI setting; it changes neither training numeric checks, logger
level nor shader compilation flags.

The option passes from the CLI through `Application` to `Context` before
instance creation. Enabled state after the layer-availability check still
flows into scheduling report metadata. An unavailable requested layer retains
the existing warning-and-disable behavior; `on` is not a strict availability
gate. Startup logs show requested/enabled state, describing application instance
configuration rather than externally forced Vulkan loader overrides. Independent
GPU diagnostic tools retain their own `--require-validation` gate.

### S4 optimizer diagnostics (experimental)

`Capture ONE S4 Optimizer Step and Pause` trains exactly one eligible ordinary
fused step on an already paused model, then exports pre-state and original
post-reference. It is not a read-only export or an optimization. See the
[S4 capture/replay guide](optimizer-diagnostics.md) for resource limits, failure
semantics, strict device/shader matching, validation commands and native evidence.
The explicit offline `--candidate-r1` streamed-Adam prototype preserves AoS and
original math. Target 3090 replay passes bitwise but fails performance admission;
local AMD multi-step bitwise differences remain unresolved. It is not selectable
in training. The guide records the target native evidence, strict
synthetic/replay gates and separate bounded profiling window. S4-1M layout
migration and S4-2 training integration remain unimplemented.

### S4-3 scheduling diagnostics (experimental)

The opt-in bounded scheduling sampler and independent direct/GPU-count sort
probe audit prepare-to-main gaps and capacity-dependent sort costs. They retain
the original training submissions, fences and CPU count/capacity handling;
GPU-driven training is not enabled. See the [scheduling diagnostic guide](scheduling-diagnostics.md)
for startup environment variables, validation/cost commands and interpretation.
Timing schema v2 additionally records GPU prepare duration, actual application
validation-layer configuration, C++ build/compiler/macros and logger levels.
Per-step training validation is distinct from Vulkan validation; unknown instance
configuration is null. No validation/logging settings or scheduling are changed.

## Architecture

```text
Application
├─ Window / Context / Renderer routing
├─ ViewerController + ViewerPanel
└─ TrainingController + TrainingPanel
   └─ GaussianTraining façade
      ├─ TrainingFrameScheduler
      ├─ TrainingImageRuntime
      ├─ TrainingStepExecutor
      ├─ TrainingValidationService
      ├─ TrainingProfilingService
      ├─ TrainingModelIO
      ├─ GaussianForwardRenderer
      ├─ GaussianBackwardRenderer façade
      │  ├─ BackwardLossPass
      │  ├─ PixelTo2DGSDispatcher
      │  ├─ ProjectionOptimizerPass
      │  └─ BackwardValidationPass
      └─ GaussianDensificationRenderer
```

The main dependency direction is:

```text
Application / UI
    -> controller or renderer façade
        -> focused training/graphics service
            -> Vulkan, image, and buffer wrappers
```

### Ownership boundaries

| Owner | Responsibility |
| --- | --- |
| `Application` | Window/context lifetime, input sampling, renderer selection, and frame routing |
| `ViewerController` | PLY model, camera policy, transforms, render settings, and viewer settings persistence |
| `TrainingController` | Training GPU, dataset lifecycle, worker, immutable snapshots, timer, benchmark, and export |
| `TrainingPanel` | Training ImGui and file-dialog state; dispatches typed controller actions |
| `GaussianTraining` | Dataset/model semantics and one training-step orchestration |
| `TrainingStepExecutor` | Prepare/main command buffers, submit/wait, tile-count handling, validation copy, and densification adoption |
| `TrainingBuffers` | Training GPU buffers and descriptor buffer information |
| `GaussianRenderer` | Normal-rendering façade over focused frame, resource, sorter, pipeline, and recorder modules |

UI code does not access `TrainingBuffers`, command buffers, descriptor sets, or
live training renderer state. The worker is the only normal mutator of
`GaussianTraining` while asynchronous training is active.

## Repository layout

```text
apps/                               Executable entry point
src/app/                            Application composition and GLFW window
src/viewer/                         Viewer controller, panel, settings, camera
src/graphics/                       PLY model and normal graphics rendering
src/render/                         Renderer interfaces shared by render paths
src/training/app/                   Training controller, settings, snapshots
src/training/ui/                    Training controls, diagnostics, dialogs
src/training/core/                  GaussianTraining and compute services
src/training/core/backward/         Backward loss/dispatch/optimizer/validation passes
src/training/cache/                 Device-local target-image cache
src/image/                          Decode, disk cache, host cache, streaming
src/context/                        Vulkan instance/device/surface context
src/vulkan/                         Vulkan resource wrappers
shaders/gaussian_compute_shader/    Normal rendering shaders
shaders/training_shader/            Training compute shaders and ABI declarations
tests/                              CPU/service/architecture regression tests
cmake/                              Opt-in code-quality targets
tools/                              Quality-evaluation utilities
```

## Application and viewer flow

Each frame performs the following work:

1. `Application::tick()` polls window events and calls
   `TrainingController::tick()`.
2. `Application` converts GLFW/ImGui input into `ViewerInputState`.
3. `ViewerController` updates Orbit/Fly camera policy once and prepares
   `ViewerRenderData`.
4. `ViewerPanel` and `TrainingPanel` draw through the renderer's ImGui callback.
5. `GaussianRenderer` records GPU sort, indirect splat draw, and ImGui, then
   presents the swapchain image.

`GaussianRenderer` is composed from:

- `GraphicsFrameRuntime`: acquire, fences, semaphores, submit, present, resize;
- `GraphicsPipelineSet`: render pass and Legacy/Compatible pipelines;
- `GraphicsSplatResources`: packed model SSBO, UBOs, and descriptors;
- `GraphicsSplatSorter`: key generation, radix sort, indirect draw arguments;
- `GraphicsCommandRecorder`: render-pass and draw command recording.

Legacy and SuperSplat Compatible profiles share the same camera. The profile
changes kernel support and blend semantics, not navigation behavior.

## Configuration persistence

`TrainingSettingsStore` and `ViewerSettingsStore` share the same
`training-settings.cfg` file but own disjoint keys. The default path is:

- Windows: `%APPDATA%/vulkan-3dgs/training-settings.cfg`
- Linux with `XDG_CONFIG_HOME`: `$XDG_CONFIG_HOME/vulkan-3dgs/training-settings.cfg`
- Linux fallback: `$HOME/.config/vulkan-3dgs/training-settings.cfg`
- final fallback: `.vulkan-3dgs-training-settings.cfg` in the current directory

Starting a run creates an immutable `TrainingRunConfig`. UI edits made while a
worker is active do not mutate the live run configuration.

## Dataset and model input

The training loader reads an undistorted MipNeRF360/COLMAP-style scene:

```text
scene/
├─ sparse/0/cameras.bin
├─ sparse/0/images.bin
├─ sparse/0/points3D.bin       optional
└─ images, images_2, images_4, or images_8/
```

The loader also accepts `sparse/` or root-level COLMAP binary files. It reads
actual image dimensions with STB, requires consistent dimensions, and rescales
PINHOLE/SIMPLE_PINHOLE intrinsics to the selected image directory. Distorted
camera models are not corrected by the training path.

After sorting loadable images by name, zero-based indices `0, 8, 16, ...` are
held out for validation. A single-view scene remains a training-only dataset.
Benchmark frame indices address the training subset, not the original full
COLMAP list. Match this split when evaluating held-out quality.

If `points3D.bin` is available, `TrainingModelIO` initializes positions and RGB
from sparse points and estimates scale with a CPU 3-nearest-neighbor search. If
no sparse points exist, it samples the configured random Gaussian count around
the camera extent.

The viewer and exporter use binary little-endian 3DGS PLY fields including
position, opacity, scale, quaternion rotation, SH DC, and SH rest coefficients.

## Image streaming and caching

Training images are stored as packed linear `RGBA8`, not permanent float
arrays.

1. `ImageStreamer` resolves source images, decodes them, and manages host LRU
   entries.
2. `ImageDiskCache` stores KTX2 array chunks with 16 layers per chunk and a
   default 20 GiB quota.
3. `DeviceImageCache` stores aligned device-local image slots and uses a
   three-slot upload ring with a timeline semaphore when supported.
4. `TrainingImageRuntime` selects the target descriptor and passes the pending
   upload semaphore/value to `TrainingStepExecutor`.

The automatic host budget uses up to 10% of currently available RAM while
reserving 2 GiB. The device cache keeps at least one image, reserves memory for
training/densification, and caps image storage at one quarter of the reported
device-local heap budget.

Set `VULKAN_3DGS_SYNC_IMAGE_UPLOAD=1` to force the synchronous upload fallback
for race diagnosis.

## Training lifecycle

`TrainingController` owns the training device and normal `std::jthread`.
Starting training performs validation/load, model initialization when needed,
renderer initialization, immutable run-config capture, and worker startup.

Normal mode publishes `TrainingUiSnapshot` every 100 iterations. Pure Training
runs the same GPU work but suppresses intermediate detailed snapshots, reduces
presentation frequency, reports wall time, and exports the configured PLY at
completion. Pause, GPU switch, dataset replacement, and shutdown stop and join
the worker before mutating resources.

Schedule modes:

- `3DGS Random`: random without replacement and a default 30,000-iteration stop;
- `Sequential`: ordered frames with no automatic stop, retained for debugging.

`Steps/Frame` is not a normal scheduler. The benchmark steps-per-frame setting
is used only by the fixed workload benchmark path.

## One training iteration

`GaussianTraining::trainStep()` prepares immutable step inputs and delegates GPU
execution to `TrainingStepExecutor`.

### Prepare submission

1. Upload or select the target image.
2. Project active Gaussians.
3. Count per-Gaussian tile coverage.
4. Run the hierarchical prefix passes.
5. Copy the 16-byte tile counter to persistent mapped memory.
6. Submit and wait for the prepare fence.
7. Grow tile buffers when the required item count exceeds capacity.

### Main submission

1. Emit contiguous tile items per Gaussian.
2. Stable-sort `(tile, depth)` as two 32-bit radix passes.
3. Rebuild per-tile `[start,count]` ranges.
4. For Adaptive Spatial, select parents and append stable child lists; then
   composite the selected tile layout into the rendered image.
5. Run loss, backward clear, loss-to-pixel, Pixel-to-2DGS, and
   projection/optimizer passes.
6. Optionally validate and densify/prune.
7. Submit, wait for the selected fence, collect compact stats, and adopt
   densified buffers when required.

The current implementation deliberately keeps the prepare readback because
main-command recording must know whether tile-item storage is large enough.

## Backward path

`GaussianBackwardRenderer` preserves the `BackwardRenderer` API and profiling
order while delegating command recording:

| Module | Pipelines and behavior |
| --- | --- |
| `BackwardLossPass` | Loss reduction, backward buffer clear, loss-to-pixel |
| `PixelTo2DGSDispatcher` | Direct, Workgroup Shared, Subgroup, Adaptive, Tile Gaussian Atomic, VkSplat Per-Splat, VkSplat Tensor |
| `ProjectionOptimizerPass` | 2DGS-to-3DGS, fused projection/Adam, separate Adam |
| `BackwardValidationPass` | Gaussian validation and final compact reduction |

Auto Pixel-to-2DGS selects only between Direct and the subgroup-adaptive path.
Unsupported explicit modes fall back to Direct. Per-Splat and Tensor modes are
never selected automatically.

### Training tiles

`Training Tile` supports Fixed8, Fixed16 (default), and experimental Adaptive
Spatial. Adaptive requires supported Tensor or Per-Splat backward; otherwise
training uses Fixed16. It sorts a 16x16 parent list once, selects profitable
parents under a child-item budget of one quarter of parent item count, appends
depth-stable 8x8 child lists, and dispatches refined work through GPU metadata.
There is no second global child sort or additional CPU count readback.

`Adaptive Split Threshold` defaults to 256. `Adaptive Tensor Child Batch`
supports 8/16/32, defaults to 8, and changes children only. Fixed16 Tensor
controls below do not alter Adaptive parent or child kernels. The reported
adaptive trials do not justify replacing Fixed16 as the default.

### Fixed16 Tensor controls

These stopped-only A/B controls keep the same descriptor ABI and use separate
compiled pipelines. Requests are persisted; `Active Tensor ...` and completed
benchmark labels report the actual gated path.

| Control / persisted key | Default | Effective condition |
| --- | --- | --- |
| `Tensor Subgroup Reduction (Experimental)` / `tensor_subgroup_reduction` | Off | Fixed16 + VkSplat Tensor; native subgroup32, compute BASIC/SHUFFLE, and base Tensor support |
| `Tensor Pixel Gradient Source` / `tensor_global_pixel_grad` | Shared (0) | Global (1) requires the above subgroup path |
| `Tensor RGB Latency Hiding (Experimental)` / `tensor_rgb_latency_hiding` | Off | Requires the above Global path |

Tensor capability still requires at least 45 KiB shared memory. Fixed16 Shared
reduction uses 45824 shared bytes/group, subgroup reduction 36612, and subgroup
with Global pixel gradients 33540. Global reads check both image coordinates.
The RGB scheduling variant preserves the same shared budget and changes RGB
consumption relative to geometry accumulation; it has no demonstrated stable
speedup. The Tensor name does not imply Tensor Core MMA.

Fixed16/Tensor/Subgroup/Global/Baseline RGB is the manually selected working
baseline for further experiments, not a change to application defaults.
The Global variant passed supplied single-step gradient comparisons and
improved a matched single-frame A/B/A; this is not cross-scene quality or
full-training performance validation. Register counts and `# Warp` are
compiler/driver observations, and `# Warp` is a theoretical residency bound.

### Loss and SSIM

The loss pass computes L1/DSSIM and stores `SsimBackwardState`; loss-to-pixel
consumes those coefficients. The default path traverses two-dimensional 11x11
neighborhoods with zero padding and fixed normalization.

`SSIM Mode` is a stopped-only A/B control, persisted as `ssim_mode`:

| Value | Mode | Loss | Loss to pixel | Scratch |
| ---: | --- | --- | --- | ---: |
| 0 (default) | 2D (Baseline) | Original 2D | Original 2D | None |
| 1 | Separable (Full, Experimental) | Horizontal moments + vertical loss | Horizontal coefficients + vertical gradient | 80 B/pixel |
| 2 | 2D Loss + Separable Backward (Experimental) | Original 2D | Horizontal coefficients + vertical gradient | 48 B/pixel |

Mode 2 reuses the original `SsimBackwardState` and executes three dispatches;
it never dispatches the separable forward-moment or vertical-loss variant.
SSIM constants, L1/DSSIM weighting, zero padding, normalization and validation
are unchanged; separable accumulation order differs. `BackwardLossPass` owns
pipeline selection and includes every dispatch/barrier in `Loss` and `Loss to
pixel` timings. `TrainingBuffers` allocates three or five float4 planes as
needed, replaces storage safely on mode changes, and releases it in baseline
mode or on resize. Allocation checks `maxStorageBufferRange` and shader indexing
limits. At 1559x1039 hybrid/full scratch costs about 74.1/123.6 MiB.

Legacy `separable_ssim=0/1` loads as modes 0/1; a present `ssim_mode` takes
precedence regardless of line order. Saving emits only `ssim_mode`; unknown
persisted values return to baseline and invalid API/config modes are rejected.
The full-separable prototype has not demonstrated an end-to-end gain. Hybrid
GPU correctness and speed remain unverified: compare current v9 pixel/projected
gradients with 0/1 first, then time Baseline/Hybrid/Baseline with 200/1000.
CPU tests do not establish a GPU speedup. Checkpointed warp backward is an
opt-in frozen-benchmark prototype.

### Checkpointed backward P0 gate

`training_checkpoint_layout.hpp` defines the experimental CPU layout/capacity
contract. `tests/training_checkpoint_reference.hpp` and
`tests/training_checkpoint_tests.cpp` implement the current scalar forward,
Tensor reverse reference, rejected prefix-state derivative, and reverse-boundary
state builder/replay. Layout 1 represents forward prefixes; layout 2 represents
reverse `(suffixColor, transmittance)` at the far edge of each 32-item bucket.

The original FP32 `C - prefixColor` candidate fails the existing projected
gradient tolerance in a nonnegative-HDR occlusion case: a dim contribution is
rounded out of final RGB and the reconstructed residual gives an incorrect
opacity gradient. Double agreement does not establish FP32 readiness. The
reverse-boundary CPU candidate passes the existing FP32 tolerance. Review also
added the production contract's `processedCount <= rangeCount` guard. Actual
mode 7 is currently available only through the P2 benchmark command.

```powershell
cmake --build build --config Debug --target training_checkpoint_tests
build/test/Debug/training_checkpoint_tests.exe --reverse-boundary-gate
```

The reverse-boundary gate has its own CTest entry. The old `--candidate-gate`
continues to report `EXPECTED REJECTION` and exit 1 for the unsafe prefix
candidate. CPU admission does not establish GPU correctness or performance.

### Checkpoint P1 state diagnostics

P1 adds opt-in GPU slot/task prefixes and reverse state construction while
the original Tensor dispatch still computes training gradients. Resources are
owned by `TrainingBuffers`; `CheckpointStatePass` belongs to
`PixelTo2DGSDispatcher`. Bindings 43/44 hold metadata/prefix scratch, 45–48
hold four 64-pixel state banks, 49 holds tasks, and 50 holds the 96-byte
control/diagnostic block. Capacity uses the existing prepare readback's true
tile-item count, is reused at its high-water mark, and is checked against
descriptor/index limits and available driver memory budget. Replacement is
transactional. Normal training releases the diagnostic workspace.

Select Fixed16 / Workgroup Shared / VkSplat Tensor / Subgroup / Global /
RGB Baseline, then click `Run P1 State Diagnostic`. The normal `Start Fixed
Benchmark` button provides the baseline on the same frozen model/frame.
The UI reports built/skipped measured steps. Other combinations are rejected;
resource failure logs a skipped diagnostic
and continues the original Tensor path. A skipped run cannot supply P1 cost
evidence. No additional count readback or queue wait occurs each step.

The profiler reports `Checkpoint layout`, `Checkpoint state build`, and
`Checkpoint tasks`; sum these for added GPU cost H and compare with the
ordinary benchmark's original Tensor backward B. `Checkpoint capacity` records CPU
preparation/allocation. The final measured step copies only 96 diagnostic
bytes through the existing submission/wait and logs slot/task counts, error
flags, and expected/written pixel-state counts. Nonzero flags or mismatched
counts fail the diagnostic. V8 exports identify whether checkpoint state was
built; P1 still records actual Tensor mode 6. Retain the run log.
Pipeline setup and workspace allocation also log their CPU durations separately
so that warmup resets do not hide startup costs.

Run the explicit headless GPU check separately from CPU CTest:

```powershell
cmake --build build --config Debug --target training_checkpoint_gpu_tests
build/test/Debug/training_checkpoint_gpu_tests.exe
```

This target checks unchanged forward outputs, reachable and poisoned unreachable
states, metadata/tasks, edge pixels, prefix blocks, HDR occlusion, repeated IDs,
and capacity-overflow suppression. It enables Vulkan validation when installed.
The GPU tool now also checks P2 gradients against an independent CPU reverse
reference and a supported GPU reference. It uses Tensor Subgroup/Global when
the device meets the original Tensor gate; otherwise it uses Direct. On a
non-native32 device it explicitly enables supported subgroup-size control to
test the same P2 shader at subgroup32. State construction now stops at B(32)
and skips tiles with maximum processed prefix <=32: no bucket reads B(0).
The independent CPU state oracle retains its complete reverse traversal.

### Checkpoint P2 warp benchmark

`Run P2 Warp Benchmark` uses the same frozen model/frame and P1 state builder,
then replaces the Tensor gradient dispatch with 128-thread groups containing
four independent subgroup32 tasks. Each lane owns one Gaussian and nine
gradient accumulators. Pixel suffix/T and loss gradients move between lanes;
32-pixel cooperative prefetch seeds the pipeline. Only subgroup registration
uses a workgroup barrier; the inner loop uses shuffles and no shared derivative
matrix. Production entry requires native subgroup32 and the P1 baseline settings.
This is a benchmark command, not a persisted training setting or Auto choice.

The control block appends Tensor fallback arguments and cumulative run counters.
Construction errors suppress the new dispatch and launch the original Tensor
over the full image. Counters reset at benchmark start and at the first measured
step, survive intermediate steps, and are read only after the existing final
submission wait. Consumer contract errors invalidate the frozen benchmark.
Runs with skipped steps, GPU fallbacks, or error flags cannot export a capture
labelled as a clean checkpoint run. `Checkpoint warp backward` includes the
conditional fallback dispatch and gradient barrier, and is nested in
`Pixel to 2DGS`; do not add both timings.

Use 0/1 ordinary/P2/ordinary captures before timing. Completed pure P2 captures
record actual pixel mode 7, bucket 32, state layout 2, task count, built state,
and zero fallbacks; Tensor-specific flags are false. Names contain
`_checkpoint_warp32_layout2`. Comparison requires the updated local script:

```powershell
python -B scripts/compare_projected_gradients.py baseline.bin p2.bin --allow-pixel-mode-change
```

Then use 200/1000 on the same frozen workload and compare
`Checkpoint layout + Checkpoint state build + Checkpoint tasks + Pixel to 2DGS`
with the ordinary Tensor `Pixel to 2DGS`. The reported P1 workload costs
1.37 ms versus a 2.99 ms Tensor baseline: P2 backward must be below 1.62 ms
to break even, or about 1.24 ms for the plan's 10% composite/backward target.
Re-measure H after the producer's redundant near-bucket work was removed.
These were pre-P2 acceptance budgets. Subsequent target screenshots passed the
frozen gradient check (relative L2 `3.71882e-7`, zero mismatched/non-finite
components), but measured H=1.18 ms and backward=3.02 ms, totaling 4.20 ms
against Tensor's 2.94 ms. Total step increased 8.83 to 10.16 ms. The direct
reverse-prepass route is stopped; retain the disabled experiment and tests,
without promoting it to normal training or running a full-training acceptance.

### Pixel-level EarlyCull experiment

`Run S3 EarlyCull Benchmark` is a one-shot frozen backward-only experiment.
Select Fixed16 / Workgroup Shared / Tensor / Subgroup / Global / RGB Baseline;
keep the same SSIM mode in both captures. It cannot combine with checkpoint
diagnostics. Auto, forward sampling, batch16, the pair matrix, reduction order
and defaults remain unchanged. The variant precomputes sixteen
FP32 thresholds per batch, adding 64 shared bytes (33604 total by declaration),
and no global workspace, prepass, new descriptor or queue wait.

Method 1 only culls finite opacity in `[1/255,1]` and original power in `[-80,0]`.
It uses `power < log(alphaMin/opacity)-2^-10`; the boundary band and unsupported
inputs execute the old sampling. Retained pairs use the original power/exp,
alpha and gradient expressions. Under the non-RelaxedPrecision FP32
[Vulkan instruction bounds](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#precision-and-operation-of-spirv-instructions),
division contributes under `3e-7` relative error, log/subtraction under `3e-6`
absolute error here, and exp plus alpha multiplication under `2.1e-5` relative
error. A conservative combined log-domain allowance of `2^-13` also covers
exp2 lowering; the exact `2^-10` guard is eight times this allowance. All
intermediates are normal in the optimization domain. This is a bounded method,
not an adjustable epsilon or permission to lower precision; target compiler/GPU
boundary tests and frozen gradients remain required.

```powershell
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
build/test/Debug/training_checkpoint_gpu_tests.exe --pixel-culling-only
python -B scripts/compare_projected_gradients.py baseline.bin early_cull.bin
```

The explicit GPU tool checks synthetic classifications around thresholds and
fallback inputs, reports eligible/old-exp/rejected pairs and fully rejected
native subgroups, then compares forward/BlendState and all twelve gradient
slots. It uses Tensor where native subgroup32 and its original resource gate
are available, otherwise a test-only Direct variant using the same sample
helper. Direct is not a Tensor runtime/performance validation. The synthetic
probe requires compute subgroup VOTE; the application variant does not add
that requirement. These synthetic counts are not target-scene savings.

Use ordinary/S3 0/1 captures first, then uninstrumented 200/1000 ordinary/S3/
ordinary timing only after correctness passes. `Pixel to 2DGS` already includes
threshold work; do not add it twice. Compare combined stages and Total step,
not exp counts alone. V9 records actual method 0/1 and names the candidate
`_pixel_cull_v1`. A reported target frozen pair passed (relative L2 `5.7464e-8`,
maximum absolute difference `4.76837e-7`, zero mismatched/non-finite/missing
components; forward/pixel differences zero). The two supplied timing images,
interpreted as Off/On, show Pixel-to-2DGS 3.14/3.12 ms and Total step 9.05/8.88 ms.
Unmodified sort/composite stages also vary; this is not a stable full-training
speedup claim. Scanline spans and forward culling remain unimplemented.

### EarlyCull full-training trial

`Tensor EarlyCull (Training, Experimental)` is a stopped-only checkbox,
persisted as `tensor_early_cull 0/1` and defaulting to **off**. Older files without
the key keep the default; malformed/unknown explicit values disable it. It
applies to normal and Pure training, including optimizer and densification
steps, only when the actual Fixed16/Workgroup Shared/Tensor/Subgroup/Global/
Baseline path and its EarlyCull pipeline are available. Otherwise keep the
existing selected path, without modifying other settings; `Training EarlyCull`
shows the reason. Math, Adam/ADC schedules, precision and shaders are unchanged
by this integration.

The immutable run config owns the request. The domain resolves a per-step CPU
policy before existing recording; no added buffer, pass or readback is needed.
`EarlyCull run steps` shows successful ordinary steps, active steps and requested
but inactive fallback steps. Counters reset on each Start/Resume Training, not
per frame or validation; benchmark steps never increment them. Start and final/
pause logs record selection and counters. Pure mode publishes final detailed
counts; partial runs therefore report only the latest Start/Resume segment.

Benchmark commands remain independent: `Start Fixed Benchmark` and P1/P2 force
culling off regardless of this checkbox; `Run S3 EarlyCull Benchmark` requests
method 1 explicitly and still rejects incompatible configurations. V9 and its
actual-path labels are unchanged. Starting ordinary training after a benchmark
restores the persisted normal-training selection, not the previous button's
override.

The reported full 30000-step trial now shows a regression, so keep this control
off except for explicitly justified reproduction. Do not repeat full runs as
part of S3-2. For interpreting past/future independently justified A/B runs,
preserve initialization, seed, split/resolution, SSIM, Adam/ADC and Tensor
settings and use different output paths. Resuming an already trained model is
not a matched fresh A/B. Counters from an uninterrupted supported On run should
be 30000 active and zero fallback; counts do not prove speed or held-out quality.
The feature remains unpromoted and outside the stable skill.

### S3-2 scanline feasibility diagnostics

This is an offline census and cost screen, not a scanline backward kernel.
The reported 30000-step EarlyCull trial regressed Pixel-to-2DGS from 2.77 to
2.91 ms and Total step from 9.92 to 10.03 ms (Off/On). Keep that experiment
off; do not repeat full training to validate this diagnostic.

1. On a frozen model, run **ordinary** `Start Fixed Benchmark`, warmup 0 /
   measured 1, Fixed16 / Workgroup Shared / Tensor / Subgroup / Global /
   RGB Baseline. Then click `Export S3-2 Diagnostic Inputs`. P1/P2 and S3
   EarlyCull captures are rejected. The normal-training checkbox does not
   affect this benchmark's path.
2. The explicit export writes `Output Folder/benchmark_scanline/*.scanline.bin`.
   It downloads projected Gaussians, **sorted** items/ranges, per-pixel
   BlendState and pixel gradients; a 1.2-million-Gaussian frame is roughly
   120 MB. It adds a stopped-only compute-to-download synchronization (with
   a semaphore bridge for a different transfer queue), not a hot-path pass or
   readback. Model, Adam, gradients and normal benchmark profiling are not
   modified. Existing captures are not overwritten.
3. Run the standalone GPU test runner on the captured input, with other GPU
   load stopped where possible. Linux uses the executable without `.exe`.

   ```powershell
   build/test/Release/training_checkpoint_gpu_tests.exe --scanline-capture path/to/frame.scanline.bin
   ```

   The runner prefers the capture's vendor/device IDs, writes a sibling
   `.json` report with numeric suffixes on collision, and never writes training
   gradients back. Start with frame 0; only add a few different views if the
   first result shows useful structure. No 200/1000 or 30000-step run is needed.

The separate `V3DSCN01` v1 input format has an 88-byte little-endian header
(magic plus 20 uint32), followed by 48-byte projected Gaussians, uint32 sorted
items, 16-byte tile ranges, 16-byte BlendStates and 16-byte pixel gradients.
Header fields are version, Gaussian count, iteration, frame, width, height,
item count, tile count, composite mode, pixel mode, subgroup, Global, RGB
schedule, SSIM, pixel-cull method, source vendor/device/driver, and two reserved
zeros. Only original Tensor Subgroup/Global/Baseline method 0 is accepted.
This file is not a model/optimizer checkpoint or a v9 gradient capture.
The reader checks exact length, metadata, dimensions, ranges and per-tile
counter bounds before GPU execution. Frame counters aggregate with uint64.

Read the report in this order:

- `eligible_pairs` excludes edge pixels, invalid IDs, `itemOffset >= processedCount`
  and the already trimmed maximum-prefix suffix. `already_guarded_pairs` and
  `already_trimmed_suffix_items` are existing savings, not new scanline gains.
- `oracle_empty_*` uses the original GPU power/exp/alpha predicate, not the
  proposed bounds. It counts rows, Gaussian **tile-items** and reverse batches
  that contain eligible pairs but no alpha contributors. Batch16 is aligned
  from each tile's maximum processed prefix exactly like Tensor, not from
  offset zero. Alpha contributions, not zero loss gradients, define the oracle.
- `span_*` measures what a guarded SPD row interval actually recognizes.
  Unsupported opacity/conics, ill-conditioning, non-finite arithmetic and
  near tangency retain the full row. The candidate expands the root interval
  outward, including at least one pixel of padding, and checks every proposed
  rejection against the original predicate. **Any false rejection rejects this
  candidate.** Zero observed errors is not production FP32/gradient admission.
- Row, Gaussian and batch pair-weighted opportunities are nested, not additive.
  A logical empty row is not automatically a native-subgroup-uniform skip.
  Only an integrated design could remove derivative-matrix work or barriers.
- `bounds_proxy_not_net_overhead_*` measures a separate batch16 bounds-only
  dispatch, including its staging/barriers/checksum but excluding mask use.
  The oracle census time is excluded. A supported native32 device also times
  the original Tensor before/after, with 2 warmup and 8 samples each. Missing
  timings are null. No-Tensor devices still validate census but cannot establish
  target Tensor profitability; cross-device IDs or >15% baseline drift are
  inconclusive.

`integration_resource_screen` also flags a separate risk: adding a naive 1024 B
span table plus 256 B of coefficients to the reported 33540 B Tensor baseline
would require 104460 B for three groups, exceeding the 100 KiB raw shared-memory
model for [Ampere 8.6](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#occupancy).
This is an arithmetic inference, not measured Vulkan residency; do not copy
CUDA-specific reservation/carveout rules into Vulkan. A future prototype must
budget native register/shared allocation and must not assume the standalone
bounds-only kernel preserves the original Tensor's residency.

`verdict` is a screen, not a speedup prediction: stop for invalid spans/input,
no eligible work, no recognized work, or weak recognition (both whole-row and
old-exp pair fractions <10%, and whole-batch fraction <5%). A bounds proxy
above 25% of the matched Tensor baseline also stops this candidate as an
engineering screen, not an impossibility proof. Other outcomes require review
before a limited integrated prototype. `estimated_speedup` deliberately stays
null and `production_kernel_implemented` false. Do not lower guards or add the
nested fractions to make the result look profitable.

```powershell
build/test/Debug/training_scanline_tests.exe
build/test/Debug/training_checkpoint_gpu_tests.exe --scanline-only
```

CPU checks cover capture I/O, truncation, invalid metadata/ranges, existing
guards and reverse batch alignment. Synthetic GPU checks compare all oracle
counters with an independent scalar reference and require zero false drops;
they include edge tiles, HDR, repeated/invalid IDs, alpha-min equality,
thin/indefinite conics, long lists and oracle-empty interior batches. These
tests do not substitute for a report from the target scene.

### S3-3 compact row-mask admission

R0 is a separate offline census. The target RTX 3090 frame-0 R0 report passed
with 13.16% geometric eligible-pair coverage, zero false drops, and matching
GPU/driver. This differs from S3-2's 45.89% row coverage, which also included
prefix-induced empty eligible sets. R1's isolated Tensor row-mask kernel and
R2's offline comparison runner are implemented, but target Tensor16 gradients,
native resources and performance still require validation.

```powershell
build/test/Release/training_checkpoint_gpu_tests.exe --row-empty-r0-capture path/to/frame.scanline.bin
```

Linux omits `.exe`. Reuse the binary capture, not the old JSON. The runner
writes a unique sibling `*.row_empty_r0.json`; it never writes training
gradients or times the census as a candidate kernel. The report separates
geometric empty rows, already-guarded rows, kept oracle-empty rows and actual
native-wave opportunities. Native waves use compute BASIC/VOTE and do not
describe Tensor's different Gaussian-reduction mapping.

The tangent lower bound retains uncertain inputs and uses the existing bounded
opacity/power exp/log contract. See the [numerical domain and error ledger](training-row-empty-contract.md).
Only RowEmpty census/candidate outputs use Slang's precise GLSL compatibility
backend. Normal shader flags are unchanged. R1 adds a 64-byte row mask and
64-byte threshold cache, refreshes both per batch, preserves zero pair writes
and existing barriers, and skips empty rows on both sampling and reduction.
Declared storage is 33668 bytes versus 33540, not measured native residency.
The FP32 audit rejects relaxed floats; its sole bounded-integer exception is
the subgroup lane builtin and its direct uint32 loads. New dependency
installation and silent compiler/capability fallback are not allowed.

Run synthetic validation first, then evaluate the unchanged binary capture
on its source device (Linux omits `.exe`):

```powershell
build/test/Release/training_checkpoint_gpu_tests.exe --scanline-only
build/test/Release/training_checkpoint_gpu_tests.exe --row-empty-capture path/to/frame.scanline.bin
```

The second command writes a unique `*.row_empty_r2.json`. It requires native32
and the original Tensor capability gate, matching source GPU/driver, a fresh
R0 census with zero false drops and at least 10% coverage, then compares all
12 gradient slots against original Tensor and a baseline repeat. Pixel
gradients/BlendStates are downloaded once outside timing to check they remain
unchanged; forward is not rerun. Errors abort timing. Only after correctness
passes does it collect short 2-warmup/8-measured A/B/A GPU timestamps. All
classification, mask use and trailing gradient barriers are inside the
candidate interval; allocation, clearing, census and downloads are outside
both intervals. Compiler backends differ, so this measures the net candidate,
not an isolated mask-only benefit. No time is inferred from the R0 fractions.

If the short test gains and Nsight native register/shared/residency review
does not show a disqualifying regression, stop profiling and run:

```powershell
build/test/Release/training_checkpoint_gpu_tests.exe --row-empty-capture path/to/frame.scanline.bin --timing --resources-reviewed
```

The resource flag is an operator attestation, not an automatic measurement.
Long 200/1000 A/B/A runs only if the repeated short screen still gains. More
than 2% baseline drift is inconclusive; less than 5% net gain, or gain no
greater than twice the A-to-A difference, fails this engineering gate. A pass
still needs resource/multiview review, not automatic normal-training enablement.
The runner never substitutes Direct or portable batch4 for target Tensor16.
On small-shared-memory devices, synthetic tests explicitly use batch4/Shared
wrappers to exercise the same mask/matrix/barrier code, comparing against an
unculled batch4 kernel and an independent scalar backward. These are not
Tensor16 performance or native-subgroup validation. Stable skill instructions,
training defaults and the v9 export format remain unchanged; no 30000-step run
or application benchmark integration is authorized by a short-test result.

### Projection and optimizer

Projection backward and Adam are fused by default. Set
`VULKAN_3DGS_SEPARATE_PROJECTION_OPTIMIZER=1` for A/B debugging. Set
`VULKAN_3DGS_COMPUTE_BACKWARD_CLEAR=1` to replace the normal `vkCmdFillBuffer`
clear with the compute fallback.

## Buffers and descriptor ABI

`TrainingBuffers` owns capacities independently for active Gaussians,
densification workspace/output, tile items, and image extent. The major groups
are:

| Group | Examples |
| --- | --- |
| Parameters | Gaussian parameters, gradients, Adam states |
| Projection | Projected Gaussians and projected gradients |
| Tiles | Coverage ranges, prefix scratch, sort keys/indices/storage, tile ranges |
| Pixels | Rendered/target color, loss, pixel gradients, blend/SSIM states |
| Densification | Densification state, output parameters/Adam, counters |
| Validation | Pixel partials, Gaussian partials, 112-byte final result |
| Auxiliary | Camera uniform, counters, preview instances |

All training shaders use descriptor set 0. Important bindings are:

| Binding | Resource |
| ---: | --- |
| 0/1/2 | Gaussian parameters, gradients, Adam state |
| 3/4/5 | Projected Gaussians, sorted tile items, tile ranges |
| 6/7/8 | Rendered color, packed target color, loss |
| 11 | Camera uniform |
| 12/13/14 | Pixel gradients, projected gradients, blend states |
| 16 | Densification state |
| 17/18/19 | Densified parameters/Adam and counters |
| 20–24 | Tile sort keys, indices, scratch, sorted output |
| 25–27 | Densification candidate buffers |
| 28 | SSIM backward state |
| 29/30/31 | Pixel partial, Gaussian partial, final validation result |
| 32/33 | Per-Gaussian tile range and prefix scratch |
| 34/35/36/37 | Adaptive split flags, child counts, item masks, child offsets |
| 38/39/40/41 | Adaptive parent prefixes, block scratch/offsets, active parents, child dispatch metadata |
| 42 | Optional SSIM scratch, three (hybrid) or five (full) float4 planes |
| 43/44 | Optional P1 checkpoint tile metadata and uint2 prefix scratch |
| 45–48 | Optional P1 checkpoint state banks, 64 pixel ranks per bank |
| 49/50 | Optional checkpoint bucket tasks and 96-byte control/fallback/run block |

Binding numbers and C++/Slang struct layouts are ABI. Update C++ descriptor
writes, Slang `[[vk::binding(N, 0)]]`, and static assertions together.
`TrainingBuffers` descriptor getters return values; store them in local
variables before assigning `pBufferInfo`.

## Synchronization

`TrainingStepExecutor` owns the training command pool, command buffers, and
fences. The worker executes synchronous GPU iterations:

```text
record prepare -> submit -> wait -> read tile count / resize
record main    -> submit -> wait -> collect validation / adopt densification
```

Passes record explicit Vulkan barriers for storage-buffer hazards. Transfer
clears use `TransferWrite -> ComputeShader`; shader outputs use destination
access appropriate for compute, transfer, or indirect commands. Backward pass
objects only record commands and never submit queues or own fences.

## Validation, profiling, and benchmark

Validation runs for the first five iterations and then at the configured
interval. The GPU reduces pixel and Gaussian diagnostics into one fixed
112-byte result. `TrainingValidationService` owns the three-slot host-coherent
readback ring, polls its fences, and produces current/history statistics. It
does not download full rendered, loss, or Gaussian buffers during normal
validation.

`TrainingProfilingService` owns the timestamp query pool and CPU/GPU sample
accumulators. Renderer/pass objects only observe the query pool. GPU stages
include projection, tile preparation, composite, loss, backward subpasses,
validation, optimizer, and densification.

Use `Start Fixed Benchmark` for kernel A/B. It pins one frame, freezes model and
iteration state, disables optimizer and densification, resets statistics after
warmup, and validates only the final measured step. It is not an end-to-end
training benchmark; use Pure Training for wall-clock comparisons.

### Gradient checks and timing

1. Pause on one model; keep the same benchmark frame, forward settings, and
   extent. Use warmup 0/measured 1 and export gradients after each completed run.
2. Compare the actual paths with the local helper:

   ```powershell
   python scripts/compare_projected_gradients.py baseline.bin candidate.bin
   python -m unittest discover -s tests -p "test_*.py"
   ```

3. For a new performance hypothesis, use warmup 200/measured 1000 and A/B/A,
   recording averages without resuming normal training. Verify measured
   progress reaches its target. `n=30000` accumulated training stats are not a
   completed 1000-step fixed benchmark.

`Export Benchmark Gradients` writes under the configured output directory's
`benchmark_gradients/`. Fixed16 Tensor names identify reduction, pixel-gradient
source, and Global RGB schedule; Adaptive Tensor names identify child batch.
All names also identify `_ssim_2d`, `_ssim_separable`, or
`_ssim_backward_separable`; existing names receive numeric suffixes.
The writer uses format v9: a 108-byte little-endian header with magic
`V3DGGRD1`, 23 uint32 fields and loss/alpha float32 values. Five fields follow
the v7 SSIM mode: checkpoint bucket size, state layout version, task count,
state-built flag, and fallback count. Baselines without checkpoint state use
zero checkpoint fields. V9 appends the actual backward pixel-culling method
(0 Off, 1 guarded FP32 EarlyCull) after the v8 checkpoint fields. Method 1
requires Tensor Subgroup/Global/Baseline without checkpoint state. V8 retains
its 104-byte layout. V6/v7 retain their 84-byte layouts; v6 SSIM accepts
only 0/1, and v7 adds 2=2D loss + separable backward. Mode 1 still means full
separable. Pixel mode 7 requires v8 or newer, Fixed16/Workgroup Shared, built
bucket32/layout2 state, and no fallback.
Payload is 12 float32 projected-gradient components per Gaussian, then one
membership byte per Gaussian, then `width*height` float4 pixel gradients
(RGB active, W padding). The comparator accepts v1–v9, reports actual paths, and
checks repeat captures retain the same path. Inspect mismatch counts, not
only exit status; do not loosen tolerance merely to pass. V6+ pairs additionally
report pixel-gradient errors using `--pixel-atol=1e-9` and the existing relative
tolerance; older pairs explicitly report pixel data unavailable. Non-finite
forward loss/alpha is rejected.

`Export Adaptive Tile Diagnostics` writes a CSV and companion mask file under
`benchmark_tiles/`. Adaptive construction timings are nested in sort/ranges;
parent/child backward timings and `Tile-local backward` are nested in
Pixel-to-2DGS and must not be added twice.

The current ignore rules exclude `scripts/`, capture outputs, `plan.md`, and
`.codex/`. Sync local helpers/plans/skills explicitly when transferring only
tracked source changes. The commands above require those local scripts.

### Shader source profiling

For source-correlated GPU Trace, enable the CMake option (default OFF):

```powershell
cmake -S . -B build -DVULKAN_3DGS_TRAINING_SHADER_DEBUG_INFO=ON
cmake --build build --config Release
```

Single-config Linux builds also need the intended `CMAKE_BUILD_TYPE`, such as
Release. The option applies Slang `-g2` to training shaders, with a dependency
stamp to rebuild after a toggle. It is not an environment variable.

Use the `ngfx` launcher and real-time shader profiling, select the active
compute device/time range, and capture while benchmark progress increases.
Green source correlation does not imply execution samples exist. Zero samples
and missing debug information are separate issues; Show Inactive cannot
recover an uncaptured workload. Keep ordinary timings separate from Nsight
captures, which can control GPU clocks. Expand hotspot callsites before
attributing LGSB/SHRTSB to a load, spill, or bank conflict; the sampled line can
be a consumer of data produced elsewhere.

## Densification and optimizer

Default densification starts at iteration 500, stops at 15,000, runs every 100
iterations, and resets opacity every 3,000 iterations. It supports clone,
split, opacity pruning, screen/world-size pruning, local split offsets, scale
shrink, and Adam-state handling.

`Densification Gradient` selects Visible Support (default,
`densification_gradient_mode=0`) or Compact (Projected Exposure, 1). These are
two implemented statistics modes, not a continuous support-ratio beta. Keep
the mode matched in performance and quality comparisons.

Active/output parameter and Adam buffers are ping-ponged on adoption. Workspace
capacity grows geometrically and is rounded to 16,384 Gaussians. Float atomic
add is used when `VK_EXT_shader_atomic_float` is available; uint-radius shader
variants provide the radius-max fallback when float min/max atomics are absent.

## Environment variables

| Variable | Effect |
| --- | --- |
| `VULKAN_3DGS_ALLOW_CPU_VULKAN=1` | Allow a CPU Vulkan device fallback |
| `VULKAN_3DGS_SYNC_IMAGE_UPLOAD=1` | Force synchronous target-image upload |
| `VULKAN_3DGS_COMPUTE_BACKWARD_CLEAR=1` | Use the compute backward-clear fallback |
| `VULKAN_3DGS_SEPARATE_PROJECTION_OPTIMIZER=1` | Disable fused projection/optimizer for A/B |

## Shader and quality workflow

Register every new shader explicitly with `compile_slang_shader(...)` or
`compile_training_shader(...)` in the top-level `CMakeLists.txt`.

Quality targets intentionally cover staged-refactor modules rather than legacy
or third-party sources:

```powershell
cmake --build build --config Debug --target format-new
cmake --build build --config Debug --target check-format-new
cmake --build build --config Debug --target clang-tidy-new
cmake --build build --config Debug --target clang-tidy-advisory-new
cmake --build build --config Debug --target quality-new
```

`clang-tidy-new` gates `bugprone-*`. The advisory target reports
`modernize-*`, `cppcoreguidelines-*`, and `bugprone-*` without failing. The
wrapper analyzes one source at a time, suppresses third-party summary noise,
and preserves actionable diagnostics and exit codes.

## Troubleshooting

- **Start Training is disabled:** read the status text and inspect dataset
  validation, selected GPU, output name, and settings validation.
- **A mode falls back to Direct:** inspect the active Pixel-to-2DGS mode and
  device subgroup/shared-memory support.
- **A training pause is long:** compare CPU `Main submit/wait`, `Validation`,
  `Densify adopt`, and GPU `Densify/prune` before changing synchronization.
- **Linux loader failure:** compare `ldd`, `vulkaninfo`, `VK_LAYER_PATH`, and an
  invocation without inherited `LD_PRELOAD`.
- **C1041/PDB contention on MSVC:** stop duplicate builds; `/FS` is appropriate
  only when multiple compiler processes legitimately share one PDB.

## Known limitations

- Training math is project-specific and is not claimed to match reference 3DGS
  exactly.
- Each training iteration still waits for prepare and main submissions.
- Tile sorting uses two stable 32-bit radix passes rather than one packed key.
- Loss and loss-to-pixel remain separate stages; full/hybrid SSIM use four/three
  dispatches and extra scratch bandwidth. Hybrid still needs target-GPU validation.
- Checkpointed warp backward remains a stopped performance experiment;
  backward EarlyCull has default-off reproduction controls and a passed
  target frozen pair, but its reported full-training trial regressed;
  quality equivalence remains unverified. S3-2 is offline diagnosis only.
  Scanline spans, GPU SoA/AoSoA storage, and GPU-driven count/sort
  are future work, not current behavior.
- Pixel backward kernel selection is manual; Auto does not choose VkSplat.
- RGB latency hiding has no demonstrated stable speedup; all Tensor A/B
  switches remain default-off and do not change Adaptive kernels.
- After the normal schedule completes, a synchronous training-complete check
  can stop a new multi-step fixed benchmark early; use an earlier frozen
  checkpoint and verify progress pending a separate lifecycle fix.
- The full Gaussian gradient buffer remains allocated for fallback and
  optimizer-disabled/densification paths.
- Densification can still trigger large allocations and bandwidth spikes.
- Training accepts undistorted PINHOLE/SIMPLE_PINHOLE cameras and ignores target
  alpha masks.
- Optimized paths have primarily been validated on RTX 3090 rather than a broad
  NVIDIA/AMD/Intel matrix.
