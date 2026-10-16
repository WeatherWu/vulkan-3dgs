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
consumes those coefficients. Both still traverse two-dimensional 11x11
neighborhoods with zero padding and fixed normalization. Separable SSIM and
checkpointed warp backward are not implemented.

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
Existing names receive numeric suffixes. The writer uses format v5: an 80-byte
little-endian header with magic `V3DGGRD1`, 16 uint32 fields and loss/alpha
float32 values, followed by 12 float32 gradient components and one membership
byte per Gaussian. The comparator accepts v1–v5, reports actual paths, and
checks repeat captures retain the same path. Inspect mismatch counts, not
only exit status; do not loosen tolerance merely to pass.

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
- Loss and loss-to-pixel remain separate passes.
- Separable SSIM, checkpointed warp backward, pixel-level EarlyCull/scanline
  prototypes, GPU SoA/AoSoA parameter storage, and GPU-driven count/sort are
  future optimization directions, not current behavior.
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
