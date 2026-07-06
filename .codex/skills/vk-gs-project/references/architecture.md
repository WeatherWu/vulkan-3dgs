# Architecture Reference

## Repository Layout

- `CMakeLists.txt`: top-level build and Slang shader compilation.
- `apps/main.cpp`: normal app entry, builds `vk_gs_windows`.
- `sandbox/`: debug utilities, including `dataset_probe.cpp`.
- `src/application.*`: GLFW app, ImGui UI, file dialogs, training controls, render loop.
- `src/renderer.*`: base renderer interfaces: `Renderer`, `ForwardTrainingRenderer`, `BackwardRenderer`.
- `src/context/`: Vulkan instance/device/surface/queue context.
- `src/vulkan/`: wrappers for buffers, command pools, compute pipeline, swapchain, shader loading.
- `src/gaussian_renderer/`: graphics PLY Gaussian renderer.
- `src/gaussian_training/`: compute training system.
- `src/utils/`: camera, file utilities, logger, stb implementation.
- `shaders/gaussian_compute_shader/slang/`: normal graphics render shaders.
- `shaders/training_shader/slang/common/`: shared training shader code.
- `shaders/training_shader/slang/passes/`: training compute passes.
- `third_party/ImGuiFileDialog`: UI file/folder dialog.
- `third_party/vulkan_radix_sort`: 32-bit Vulkan radix sort used by training tile sorting.

## Build

Use the existing configured build tree:

```powershell
cmake --build build --config Debug
```

The build generates SPIR-V into source shader `spv` folders and `build/bin/<Config>/shaders/`.

Top-level CMake requires `slangc` and compiles each Slang pass explicitly. When adding a shader, add it to `CMakeLists.txt`.

## Runtime Targets

- `build/bin/Debug/vk_gs_windows.exe`: primary app.
- `build/bin/Debug/vk_gs_sandbox.exe`: debug sandbox.

## Application/UI Flow

`Application` owns window setup, context initialization, render loop, ImGui controls, file dialogs, and training UI state.

Training UI state variables live in `src/application.hpp`:

- `training_dataset_valid_`
- `training_dataset_loaded_`
- `training_initialized_`
- `training_running_`
- `training_steps_done_`
- training optimizer/densification/init parameters

Current `Start Training` behavior:

1. If dataset is not loaded, call validation and load.
2. If no trainable model exists, initialize from COLMAP sparse points or random fallback.
3. Initialize training renderers.
4. Set `training_running_ = true`.
5. `Application::update()` calls `runTrainingStepFromUi()` `Steps/Frame` times per UI frame.

## Renderer Interfaces

`Renderer` assumes concrete renderers own their swapchain/images/buffers. Its methods have no image/buffer parameters:

- `renderToImage()`
- `presentImage()`
- `renderToBuffer()`

`ForwardTrainingRenderer` and `BackwardRenderer` are compute training base classes in `renderer.hpp`.

## Graphics Gaussian Renderer

`src/gaussian_renderer/` handles normal model visualization:

- `GaussianModel` loads 3DGS-style binary little-endian PLY.
- `GaussianRenderer` manages swapchain/render pass/pipeline/descriptors.
- GPU key generation + `vulkan_radix_sort` sort visible splats.
- Graphics vertex/fragment shaders render instanced quads.

PLY fields expected include:

- `x`, `y`, `z`
- `f_dc_0..2`
- `f_rest_0..44`
- `opacity`
- `scale_0..2`
- `rot_0..3`

Normal graphics rendering should stay independent from training unless explicitly changing shared model/data contracts.
