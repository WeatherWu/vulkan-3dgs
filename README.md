# vulkan-3dgs

[English](README.md) | [简体中文](README.zh-CN.md)

vulkan-3dgs is an experimental Vulkan + Slang 3D Gaussian Splatting project. The repository currently contains a standard PLY rendering path and a MipNeRF360/COLMAP dataset training path. The training side supports compute forward/backward passes, Adam optimization, densification/pruning, and PLY export.

## Current Targets

- `vulkan_3dgs_core`: core static library with Vulkan context setup, resource wrappers, camera utilities, PLY loading, and the Gaussian renderer.
- `vulkan_3dgs_app`: main application entry from `apps/main.cpp`. Windows and non-Windows builds output `vulkan-3dgs`.

Release builds include the main application and core library by default.

## Repository Layout

```text
vulkan-3dgs/
├── CMakeLists.txt
├── README.md
├── README.zh-CN.md
├── insect.ply
├── apps/
│   ├── CMakeLists.txt
│   └── main.cpp
├── shaders/
│   ├── gaussian_compute_shader/
│   │   └── slang/
│   │       ├── gaussian_common.slang
│   │       ├── gaussian_compute_shader.vert.slang
│   │       ├── gaussian_compute_shader.frag.slang
│   │       └── radix_keygen.comp.slang
│   └── training_shader/
│       └── slang/
│           ├── common/
│           └── passes/
└── src/
    ├── application.hpp/cpp
    ├── window.hpp/cpp
    ├── renderer.hpp/cpp
    ├── context/
    │   ├── context.hpp/cpp
    │   └── device.hpp/cpp
    ├── vulkan/
    │   ├── buffer.hpp/cpp
    │   ├── command_pool.hpp/cpp
    │   ├── compute_pipeline.hpp/cpp
    │   ├── shader.hpp/cpp
    │   └── swapchain.hpp/cpp
    ├── gaussian_renderer/
    │   ├── gaussian_model.hpp/cpp
    │   ├── gaussian_renderer.hpp/cpp
    │   ├── pipeline.hpp/cpp
    │   └── renderpass.hpp/cpp
    ├── gaussian_training/
    │   ├── gaussian_training.hpp/cpp
    │   ├── gaussian_forward_renderer.hpp/cpp
    │   ├── gaussian_backward_renderer.hpp/cpp
    │   ├── gaussian_densification_renderer.hpp/cpp
    │   ├── training_buffers.hpp/cpp
    │   ├── training_dataset.hpp/cpp
    │   └── training_types.hpp
    └── utils/
        ├── camera.hpp/cpp
        ├── file_utils.hpp
        └── logger.hpp
```

## Architecture Overview

`Application` owns the GLFW window, Vulkan `Context` initialization, and main loop. `Context` is the global entry point for the Vulkan instance, surface, device, and queues. `GaussianRenderer` derives from `Renderer` and owns the swapchain, render pass, graphics pipeline, compute pipeline, descriptor sets, GPU depth sorting, and per-frame submission.

The normal rendering path is:

```text
apps/main.cpp
  -> Application
  -> GaussianRenderer::setRenderData(...)
  -> GPU keygen + vulkan_radix_sort
  -> instanced quad graphics pipeline
  -> swapchain present
```

## Gaussian Data

`GaussianModel` currently supports 3DGS-style binary little endian `.ply` files:

- position: `x`, `y`, `z`
- color: `f_dc_0..2`, `f_rest_0..44`
- opacity: `opacity`
- scale: `scale_0..2`
- rotation: `rot_0..3`

Loading restores alpha and scale, then builds the covariance matrix from scale and quaternion rotation. `.splat`, `.gs`, and `.json` are reserved but not implemented yet.

## Training

`GaussianTraining` is a compute training path independent from the normal `GaussianRenderer`. It uses MipNeRF360/COLMAP-style datasets, reads actual image dimensions, scales COLMAP camera intrinsics to those dimensions, and trains Gaussian parameters.

The training path includes:

- compute forward tile renderer
- loss and backward passes
- Adam-style optimizer
- densification / pruning
- PLY export

The training UI supports two scheduling modes:

- `Sequential`: trains frames in dataset order and has no fixed total-iteration stop.
- `3DGS Random`: follows the original 3DGS viewpoint-stack strategy, samples frames randomly without replacement, refills the stack after a full pass, and stops automatically at 30000 iterations.

Densification statistics use shader float atomics, so the device must support and enable:

- `VK_EXT_shader_atomic_float`
- `VK_EXT_shader_atomic_float2`

If the GPU or driver does not support buffer float32 atomic add/min-max, device suitability checks fail and the training path will not continue in an unreliable state.

## Shader

Slang sources live in:

```text
shaders/gaussian_compute_shader/slang/
shaders/training_shader/slang/
```

CMake compiles them with `slangc` into SPIR-V and copies them into the runtime shader directory:

```text
build/bin/<Config>/shaders/
```

Current shaders include:

- `gaussian_common.slang`: shared Gaussian data structures, half-packed SH unpacking, and SH evaluation.
- `gaussian_compute_shader.vert.slang`: reads sorted Gaussian instances, projects covariance, computes screen ellipses, and evaluates SH color.
- `gaussian_compute_shader.frag.slang`: computes Gaussian alpha falloff and outputs color.
- `radix_keygen.comp.slang`: GPU visibility culling, sort-key generation, and indirect draw instance count writes.
- `training_shader/slang/passes/*`: training forward, loss, backward, optimizer, densify/prune, and preview packing passes.

## Build Dependencies

- CMake 3.26+
- C/C++ compiler with C++20 support
- Vulkan headers/library
- `slangc` (available through vcpkg's `shader-slang`)
- GLFW3
- GLM
- STB headers
- ImGui
- COLMAP
- ImGuiFileDialog (`third_party/ImGuiFileDialog` in this repository)

All CMake package dependencies are expected to be provided by vcpkg or an equivalent local package installation. The build no longer uses `FetchContent_Declare()` fallbacks, and missing packages fail during CMake configuration.

This repository includes a vcpkg manifest pinned to this registry baseline:

```bash
301856f5f2824f788a3ffa6332293861cccd23b4
```

Install the reproducible dependency set with:

```bash
vcpkg install --triplet x64-windows
```

Current vcpkg package versions from that baseline:

- `vulkan`: `2023-12-17`
- `glfw3`: `3.4#1`
- `glm`: `1.0.3`
- `imgui[glfw-binding,vulkan-binding]`: `1.92.8#1`
- `stb`: `2024-07-29#1`
- `shader-slang`: `2026.7.1`
- `colmap` without default features: `3.12.6#1`

Bundled third-party submodule versions:

- `third_party/ImGuiFileDialog`: `https://github.com/aiekick/ImGuiFileDialog.git` at `d0e97b2adc3d3452d72c750c7305dc0291acd052`
- `third_party/vulkan_radix_sort`: `https://github.com/jaesung-cs/vulkan_radix_sort.git` at `7b9912bb827fcc569854e45b43748fd27bc5dde3`

Training densification requires Vulkan float atomic extension support from the GPU and driver:

- `VK_EXT_shader_atomic_float`
- `VK_EXT_shader_atomic_float2`

On Linux, use `vulkaninfo` to check whether the driver exposes these extensions.

## Build

Visual Studio generator example:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
```

If Vulkan or vcpkg dependencies are not found automatically, pass paths explicitly:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" `
  -DCMAKE_TOOLCHAIN_FILE=C:/Users/weath/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DVulkan_INCLUDE_DIR=C:/Users/weath/vcpkg/installed/x64-windows/include `
  -DVulkan_LIBRARY=C:/Users/weath/vcpkg/installed/x64-windows/lib/vulkan-1.lib
```

Release build:

```powershell
cmake --build build --config Release
```

Example run paths:

```powershell
.\build\bin\Release\vulkan-3dgs.exe
```

Ubuntu + vcpkg example:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
  -DVCPKG_TARGET_TRIPLET=x64-linux

cmake --build build
```

On Ubuntu, install the Vulkan loader, driver, and baseline X11 development packages:

```bash
sudo apt install vulkan-tools libvulkan1 mesa-vulkan-drivers xorg-dev
```

Example vcpkg manifest install:

```bash
vcpkg install --triplet x64-linux
```

The non-Windows main application output is:

```bash
./build/bin/Debug/vulkan-3dgs
```

## Current Limitations

- `Application::initialize()` is currently called from the constructor and is not suitable for derived-class virtual dispatch.
- `.splat`, `.gs`, and `.json` loading are not implemented yet.
- GPU sorting and descriptor resource rebuilds remain active maintenance areas.
- The project does not use VMA yet; buffer/image memory is still allocated manually.
- `3DGS Random` matches the original 3DGS random viewpoint stack and 30000-iteration schedule, but the forward/backward math is still project-specific and should not be claimed as exact reference 3DGS parity.
- Training results currently enter the normal rendering path through PLY export. The main viewport does not automatically render training buffers in real time.

## Logging

Logging uses `utils/logger.hpp` and defaults to INFO and above. Debug builds set the log level to `DEBUG_VULKAN_3DGS`.

Per-frame INFO logs on hot paths have been trimmed. INFO is mainly kept for startup, device selection, model loading, swapchain initialization, and resize rebuild status.
