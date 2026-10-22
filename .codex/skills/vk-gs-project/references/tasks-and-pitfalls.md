# Tasks And Pitfalls

## Common Commands

Build:

```powershell
cmake --build build --config Debug
```

Find files:

```powershell
rg --files
```

Find symbols:

```powershell
rg -n "symbolName" src shaders
```

Run tests for a multi-config build:

```powershell
ctest --test-dir build -C Debug --output-on-failure
ctest --test-dir build -C Release --output-on-failure
```

Run app after build:

```text
build/bin/Debug/vulkan-3dgs.exe
build/bin/Debug/vulkan-3dgs
```

## Adding Or Editing Training Shader Passes

1. Add or edit Slang under `shaders/training_shader/slang/passes/` or `common/`.
2. If a new pass is added, add `compile_training_shader(...)` in top-level `CMakeLists.txt`.
3. Add C++ pipeline/descriptors in the corresponding renderer.
4. Keep binding numbers consistent with shader declarations and `updateDescriptorSet`.
5. Build Debug to compile SPIR-V.

Validation shader passes live under `shaders/training_shader/slang/validation/`; compatibility variants live under `uint_radius/`.

## Vulkan Descriptor Pitfall

Do this:

```cpp
auto renderedColorInfo = trainingBuffers_->renderedColorInfo();
write.setPBufferInfo(&renderedColorInfo);
```

Do not take the address of a temporary returned by a getter.

## Training Dataset Errors

`Training target image size does not match training buffer extent`:

- Means target image decoded size differs from `TrainingBuffers::extent()`.
- Current fix is dataset loader using actual image dimensions via `stbi_info`.
- If it returns, check for stale build, mixed old executable, or per-frame image dimensions differing.

`MipNeRF360 scene is missing COLMAP sparse cameras.bin/images.bin`:

- Dataset root is wrong or sparse files are not under `sparse/0`, `sparse`, or root.

`MipNeRF360 scene is missing images/images_N directory`:

- No supported image folder exists. Supported candidates are requested `images_N`, `images_4`, `images_2`, `images_8`, `images`.

`Training images must have a consistent size`:

- Selected image directory contains mixed-resolution images. Current training buffers do not support dynamic per-frame extent.

`GaussianTraining::trainStep called without initialized training gaussians`:

- Dataset may be loaded, but `initializeModelFromDataset()` did not run or failed.

## UI Training Button Issues

`Start Training` depends on the dataset path and the const
`TrainingController::state()` view. Inspect `datasetValid`,
`datasetLoaded`, `initialized`, `running`, `status`, and `error`.
`Application` no longer owns parallel copies of these values.

Current expected behavior:

- Choosing a folder validates it.
- `Load Dataset` validates and loads.
- `Start Training` can auto validate/load if needed.
- Changing dataset path/downscale resets valid/loaded/initialized/running state.
- Normal training starts a `std::jthread`; pause requests stop after the current step. Join the worker before switching GPUs, reloading datasets, cleaning training resources, or exiting.
- UI code must read `TrainingUiSnapshot` while async training is active. Do not call non-thread-safe `GaussianTraining` getters directly from the UI thread.
- Pure Training publishes only the final detailed snapshot, throttles presentation, reports wall time, and automatically exports the configured PLY on normal completion.

## PLY Export

Training PLY export downloads `GaussianTrainParam` from GPU and writes 3DGS-style binary little-endian PLY:

- position
- SH DC
- SH rest
- raw opacity
- log scale
- normalized quaternion in `rot_0..3` order as w,x,y,z

Normal `GaussianModel` reads 3DGS-style PLY for visualization.

## Frontend/UI Style

This app uses ImGui. Prefer compact controls over explanatory in-app text.
Training controls live in `TrainingPanel` subpanels, not `Application`.

When adding parameters:

1. Add passive configuration in the appropriate `TrainingSettings` group and store/validation code.
2. Add a subpanel control and typed action; read runtime state through `TrainingUiSnapshot`.
3. Apply through `TrainingController` while stopped or from the immutable `TrainingRunConfig` at run start; do not access worker-owned live training from the UI thread.
4. Propagate through the owning training façade/service and add default, persistence, and policy tests.

Viewer controls are shared across both render profiles. `Legacy` and `SuperSplat Compatible` change graphics kernel/blend semantics, not camera behavior. Keep `CameraController` reset/focus aspect-aware and keep focus bounds separate from scale-expanded clipping bounds.

## Coding Practices In This Repo

- Prefer existing wrapper classes (`Buffer`, `CommandPool`, `ComputePipeline`) over raw repeated Vulkan boilerplate.
- Keep renderer ownership boundaries clear: normal graphics renderer should not own training resources.
- Use `apply_patch` for manual edits.
- Avoid unrelated refactors while debugging runtime issues.
- Add precise runtime errors with paths/sizes/counts when validating datasets.

## Linux Vulkan Loader And GLFW

- Do not mix a system-created `VkInstance`/`VkDevice` with entry points dispatched through a different Vulkan loader.
- Check the actual loader and GLFW paths with `ldd ./vulkan-3dgs | grep -E "vulkan|glfw"`.
- An inherited `LD_PRELOAD` from VirtualGL/VNC can inject another loader or dispatch layer. Reproduce with `env -u LD_PRELOAD ./vulkan-3dgs` before changing code.
- vcpkg's `vulkan-loader` must include the `xcb`, `xlib`, and `wayland` features for GLFW surface extensions on Linux.
- `glfwGetRequiredInstanceExtensions()` returning zero extensions usually indicates loader/platform discovery mismatch, not missing Vulkan headers.
- `VK_ICD_FILENAMES` must point to an ICD JSON that actually exists on that machine; common paths differ between `/usr/share/vulkan/icd.d` and `/etc/vulkan/icd.d`.
- Validation layers are runtime-discovered layers, not ordinary linked libraries. Installing `vulkan-validationlayers` does not guarantee discovery if layer manifests or `VK_LAYER_PATH` are wrong.

## Image Cache Diagnostics

- Host cache stores decoded linear `RGBA8`, so one image costs `width * height * 4` bytes.
- Default host budget is 10% of currently available RAM while reserving 2 GiB.
- Device cache stores the same `RGBA8` payload in aligned storage-buffer slots and caps allocation at one quarter of the device-local heap budget.
- Asynchronous upload staging normally costs three aligned images of host-visible memory.
- Use the UI `Image Cache` panel rather than estimating from process totals: inspect host cached/budget, KTX2 size/hits, GPU allocated/budget, slot/resident counts, staging bytes, heap usage, uploads, and evictions.
- Frequent GPU cache resizes around densification can evict resident images and cause subsequent upload stalls.

## Training Validation Diagnostics

- Periodic validation should only copy the 112-byte final result through the private staging ring. Pixel partials are 80 bytes and remain device-local.
- `downloadLoss()` and `downloadRenderedColor()` remain utility APIs but must not be called from the regular validation schedule.
- `downloadGaussianParams()` is expected for PLY export, not periodic validation.
- Keep threads beyond pixel/Gaussian count participating in workgroup barriers when editing reduction shaders.
- Keep bindings 29, 30, and 31 synchronized between Slang, `TrainingBuffers`, and `GaussianBackwardRenderer`.
- Keep tile item count on the dedicated 16-byte persistent readback path; generic `Buffer::download()` recreates staging resources and must not return to the per-step hot path.
- When changing Auto Pixel-to-2DGS selection, compute utilization from valid pixel lanes only; counting out-of-image edge lanes biases partial tiles toward Direct.
- Compare Direct/Auto/Subgroup with the fixed workload benchmark. Ordinary training averages mix frames, Gaussian states, and validation/densification events.
- Auto does not select VkSplat Per-Splat or VkSplat Tensor. When benchmarking those paths, select them explicitly and verify the UI `Active Pixel Backward` line did not fall back to Direct.
- Fixed Benchmark disables optimizer and densification. Use Pure Training, not benchmark total-step time, for end-to-end claims.
- Use warmup 0/measured 1 for gradient exports, then a separate 200/1000 A/B/A for a new timing hypothesis. Record averages and do not resume ordinary training between captures. Repeat captures must retain the same actual path; see `training.md` for v1–v7 compatibility.

## Backward And Optimizer A/B

- `VULKAN_3DGS_COMPUTE_BACKWARD_CLEAR=1` forces the compute clear fallback instead of the default `vkCmdFillBuffer` clear.
- `VULKAN_3DGS_SEPARATE_PROJECTION_OPTIMIZER=1` disables the default fused projection-backward/Adam pass.
- VkSplat Tensor needs roughly 45 KiB workgroup shared memory; unsupported devices fall back to Direct.
- The full Gaussian-gradient buffer is still allocated even with fusion enabled. Do not claim complete VkSplat VRAM parity from kernel fusion alone.

### Nsight shader profiling

- Configure `-DVULKAN_3DGS_TRAINING_SHADER_DEBUG_INFO=ON` in the existing build and rebuild the desired configuration. This CMake option defaults OFF and applies `-g2` to training shaders only; preserve the user's chosen value afterward.
- Use GPU Trace with real-time shader profiling while fixed-benchmark progress is increasing. Select the active compute device and time range. Use the `ngfx` launcher; direct `ngfx.bin` can miss its library setup.
- Green Correlation means debug/source data loaded, not that execution samples were collected. Show Inactive does not create missing samples. Source debug warnings and zero samples are separate issues.
- `# Warp` and its resource tooltip report theoretical limits. A reduction in `Smem` or `# Reg` helps residency only if it crosses the limiting allocation threshold; actual occupancy and time still need measurement.
- LGSB/SHRTSB waits can be attributed to the consumer rather than the load that produced its data. Expand callsites before assigning a cause; do not infer spilling, bank conflicts, or DRAM saturation from that column alone.

## PSNR Evaluation

`tools/evaluate_psnr.py` has two modes:

- `render MODEL_PLY SCENE_DIR`: load an Inria-style PLY, render filename-sorted every-eighth held-out COLMAP views through gsplat, save PNGs, and write per-image JSON;
- `compare REFERENCE_DIR RENDER_DIR`: pair existing images and compute PSNR.

Check `psnr.json` for single-view outliers before interpreting the mean. Keep factor/image directory, split, background, quantization, model iteration, and PLY checksum/path consistent across comparisons. The evaluator requires undistorted PINHOLE/SIMPLE_PINHOLE cameras.

## Densification Performance Diagnostics

- A high GPU `Densify/prune` value indicates shader bandwidth, global atomic contention, or capacity-sized dispatch overhead.
- A high CPU `Densify adopt` value indicates counter download and active-buffer destruction/recreation.
- A large total-step pause with modest values in both can come from `ensureDensificationCapacity()` or pre-densification image-cache resizing, which currently occur outside dedicated profiling stages.
- Avoid CPU-zeroing large device-local buffers through `Buffer::create`; prefer persistent workspace and GPU clears.
- Long-term compaction should use classify + prefix sum + direct scatter instead of global atomic append into candidate buffers.

## Current Training Strategy

- Normal Start Training runs continuously on the async worker. `Benchmark Steps/Frame` applies to the fixed-benchmark/main-thread fallback and is not a normal throughput control.
- `Sequential` mode keeps ordered frame selection and has no automatic total training iteration stop.
- `3DGS Random` mode uses a random-without-replacement viewpoint stack. With `N` frames, each stack refill covers all frames once in random order.
- `3DGS Random` mode sets `totalIterations = 30000`; the UI stops training automatically at completion.

When touching scheduling, keep these separate:

- frame selection mode
- total iteration limit
- deterministic random seed
- displayed training progress
