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

Run app after build:

```text
build/bin/Debug/vk_gs_windows.exe
build/bin/Debug/vk_gs
```

## Adding Or Editing Training Shader Passes

1. Add or edit Slang under `shaders/training_shader/slang/passes/` or `common/`.
2. If a new pass is added, add `compile_training_shader(...)` in top-level `CMakeLists.txt`.
3. Add C++ pipeline/descriptors in the corresponding renderer.
4. Keep binding numbers consistent with shader declarations and `updateDescriptorSet`.
5. Build Debug to compile SPIR-V.

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

`Start Training` depends on dataset path and state. Inspect:

- `training_dataset_valid_`
- `training_dataset_loaded_`
- `training_initialized_`
- `training_running_`
- `training_status_`
- popup `training_error_`

Current expected behavior:

- Choosing a folder validates it.
- `Load Dataset` validates and loads.
- `Start Training` can auto validate/load if needed.
- Changing dataset path/downscale resets valid/loaded/initialized/running state.

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

This app uses ImGui. Prefer compact controls over explanatory in-app text. Training controls live in `Application::drawTrainingControls()`.

When adding parameters:

1. Add field in `Application`.
2. Add ImGui control.
3. Apply value on `Start Training`, not while training is running, unless explicitly requested.
4. Propagate into `TrainingOptimizerConfig`, `TrainingDensificationConfig`, or `TrainingInitializationConfig`.

## Coding Practices In This Repo

- Prefer existing wrapper classes (`Buffer`, `CommandPool`, `ComputePipeline`) over raw repeated Vulkan boilerplate.
- Keep renderer ownership boundaries clear: normal graphics renderer should not own training resources.
- Use `apply_patch` for manual edits.
- Avoid unrelated refactors while debugging runtime issues.
- Add precise runtime errors with paths/sizes/counts when validating datasets.

## Current Training Strategy

- `Steps/Frame` controls how many `trainStep()` calls run per UI update.
- `Sequential` mode keeps ordered frame selection and has no automatic total training iteration stop.
- `3DGS Random` mode uses a random-without-replacement viewpoint stack. With `N` frames, each stack refill covers all frames once in random order.
- `3DGS Random` mode sets `totalIterations = 30000`; the UI stops training automatically at completion.

When touching scheduling, keep these separate:

- frame selection mode
- total iteration limit
- deterministic random seed
- displayed training progress
