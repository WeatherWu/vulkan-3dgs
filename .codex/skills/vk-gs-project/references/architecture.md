# Architecture Reference

## Repository Layout

- `CMakeLists.txt`: top-level build and Slang shader compilation.
- `apps/main.cpp`: normal app entry; output is `vulkan-3dgs.exe` on Windows and `vulkan-3dgs` elsewhere.
- `src/app/`: GLFW Window and Application composition, input sampling, and frame routing.
- `src/viewer/`: viewer controller/panel/settings and shared viewer camera.
- `src/graphics/`: PLY model and normal graphics Gaussian rendering.
- `src/training/app/`: training controller, immutable UI snapshots, and persisted settings.
- `src/training/ui/`: training controls, diagnostics, GPU selection, and file dialogs.
- `src/training/core/`: GaussianTraining façade, compute renderers, buffers, scheduler, executor, validation, profiling, model I/O, and image runtime.
- `src/training/core/backward/`: backward loss, pixel dispatch, projection/optimizer, validation passes, and shared command context utilities.
- `src/training/cache/`: device-local RGBA8 cache and timeline-semaphore upload ring.
- `src/render/`: base renderer interfaces: `Renderer`, `ForwardTrainingRenderer`, `BackwardRenderer`.
- `src/context/`: Vulkan instance/device/surface/queue context.
- `src/vulkan/`: wrappers for buffers, command pools, compute pipeline, swapchain, shader loading.
- `src/image/`: reusable image decode, KTX2 disk cache, host-memory cache, prefetch, and LRU streaming.
- `src/utils/`: shared file utilities, logger, system memory, and stb implementation.
- `shaders/gaussian_compute_shader/slang/`: normal graphics render shaders.
- `shaders/training_shader/slang/common/`: shared training shader code.
- `shaders/training_shader/slang/passes/`: training compute passes.
- `shaders/training_shader/slang/validation/`: compact GPU validation finalize pass.
- `shaders/training_shader/slang/uint_radius/`: float-atomic compatibility variants using uint radius bits.
- `tools/evaluate_psnr.py`: render a training PLY at held-out COLMAP views with gsplat, or compare existing image directories.
- `tests/`: image/cache, training-service, camera/render-profile, validation, and GPU-selection tests.
- `third_party/ImGuiFileDialog`: UI file/folder dialog.
- `third_party/vulkan_radix_sort`: 32-bit Vulkan radix sort shared by graphics and training sorting.

## Build

Use the existing configured build tree:

```powershell
cmake --build build --config Debug
```

The build generates SPIR-V into source shader `spv` folders and `build/bin/<Config>/shaders/`.

Top-level CMake requires `slangc` and compiles each Slang pass explicitly. When adding a shader, add it to `CMakeLists.txt`.

## Runtime Targets

- `build/bin/Debug/vulkan-3dgs.exe`: primary Windows app.
- `build/bin/Debug/vulkan-3dgs`: primary non-Windows app.

## Application/UI Flow

`Application` owns window/context/renderer setup, the main loop, platform-input sampling, and frame routing. It composes controllers and panels without owning their domain state:

- grouped `TrainingSettings` and `TrainingSettingsStore` preserve the existing configuration keys;
- `TrainingController` owns the training device, `GaussianTraining`, dataset lifecycle, timer, fixed benchmark, `std::jthread`, result handoff, and immutable `TrainingUiSnapshot`;
- `TrainingPanel` owns training ImGui and file-dialog state, while subpanels consume passive settings/snapshots and emit typed intents;
- `ViewerController` owns the PLY model, camera/input policy, matrices, render settings, and viewer-settings persistence; `ViewerPanel` modifies it only through its public interface.

`Application` calls `trainingController.tick()` and
`trainingPanel.draw(trainingController)`. It does not parse settings, dispatch
training commands, access the training device, or mutate lifecycle state.

Current `Start Training` behavior:

1. If dataset is not loaded, call validation and load.
2. If no trainable model exists, initialize from COLMAP sparse points or random fallback.
3. Initialize training renderers.
4. `TrainingController` starts its `std::jthread`, which calls `GaussianTraining::trainStep()` until completion or a stop request.
5. Normal mode publishes a mutex-protected `TrainingUiSnapshot` every 100 steps; Pure Training publishes only the final snapshot, throttles UI presentation, and shows elapsed time.
6. On Pure completion, export the configured PLY automatically. Pause requests stop after the current step and join the worker before mutating training resources.

`Benchmark Steps/Frame` is not the normal scheduler. Fixed workload benchmark still uses the main-thread fallback so its warmup/measured state and UI stop condition remain deterministic.

## Renderer Interfaces

`Renderer` assumes concrete renderers own their swapchain/images/buffers. Its methods have no image/buffer parameters:

- `renderToImage()`
- `presentImage()`
- `renderToBuffer()`

`ForwardTrainingRenderer` and `BackwardRenderer` are compute training base classes in `renderer.hpp`.

## Image Ownership

Image streaming is intentionally reusable and not owned by a renderer:

- `ImageDiskCache` owns KTX2 chunk persistence and quota enforcement.
- `ImageStreamer` owns source registration, decoded `RGBA8` host entries, worker-thread prefetch, request waiting, and LRU eviction.
- `TrainingFrameScheduler` chooses the dataset frame; `TrainingImageRuntime` owns the streamer instance and target upload.
- `DeviceImageCache` is Vulkan/training-specific because it owns device-local storage-buffer slots, staging memory, command buffers, and timeline synchronization.
- `TrainingBuffers` can use a device-cache descriptor override for the current target image, with the original target buffer retained as fallback.

This boundary allows future image-processing modules to instantiate `ImageStreamer` without depending on Gaussian training.

## Validation Ownership

Training validation remains internal:

- loss and optimizer passes produce workgroup partials;
- `BackwardValidationPass`, composed by `GaussianBackwardRenderer`, owns the finalize compute pipeline;
- `TrainingBuffers` owns partial and final device-local buffers;
- `TrainingValidationService` owns the three-slot persistent host readback ring and converts the 112-byte GPU result into the latest `TrainingValidationStats` plus cross-validation candidate-profile statistics;
- no separate public validation renderer or frontend configuration is exposed.

## Graphics Gaussian Renderer

`src/graphics/` handles normal model visualization:

- `GaussianModel` loads 3DGS-style binary little-endian PLY.
- `GaussianRenderer` is a facade that routes one graphics frame without directly owning low-level frame resources.
- `GraphicsFrameRuntime` owns swapchain acquire/recreate, command buffers, fences, semaphores, submit, and present.
- `GraphicsPipelineSet` owns the render pass and the Legacy/SuperSplat-compatible graphics pipelines.
- `GraphicsSplatResources` owns the packed model SSBO, UBOs, descriptor pool, and descriptor sets.
- `GraphicsSplatSorter` owns key generation, radix-sort storage, cached sort invalidation, barriers, and indirect draw arguments.
- `GraphicsCommandRecorder` records the sort, render pass, indirect draw, and ImGui commands.
- GPU key generation + `vulkan_radix_sort` sort visible splats.
- Graphics vertex/fragment shaders render instanced quads.
- `GaussianRenderProfile::Legacy` keeps the 3-sigma/straight-alpha path; `SuperSplatCompatible` uses finite normalized support and a premultiplied blend pipeline.
- Both profiles share `CameraController`: roll-free Orbit/Fly, damping, pan/dolly, larger-axis FOV, aspect-aware reset/focus distance, and scene-fitted near/far.
- `GaussianModel` retains center-only transform bounds, percentile-trimmed opacity-visible focus bounds, and conservative scale-expanded clipping bounds.
- SH bands 0 through 3 and the selected profile are passed in the graphics UBO. Two graphics pipelines provide the different static Vulkan blend states.

PLY fields expected include:

- `x`, `y`, `z`
- `f_dc_0..2`
- `f_rest_0..44`
- `opacity`
- `scale_0..2`
- `rot_0..3`

Normal graphics rendering should stay independent from training unless explicitly changing shared model/data contracts.
