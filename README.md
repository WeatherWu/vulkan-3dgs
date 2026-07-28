# vulkan-3dgs

[English](README.md) | [简体中文](README.zh-CN.md) | [Development guide](docs/DEVELOPMENT.md) | [开发指南](docs/DEVELOPMENT.zh-CN.md)

`vulkan-3dgs` is an experimental Vulkan + Slang 3D Gaussian Splatting application. It can render standard 3DGS PLY models and train Gaussian scenes from MipNeRF360/COLMAP-style datasets.

## Features

- 3DGS PLY rendering with GPU sorting
- MipNeRF360/COLMAP dataset loading
- GPU forward and backward training passes
- Adam optimization, densification, and pruning
- Adaptive image streaming and caching
- Training validation and performance profiling
- PLY export

## Requirements

- CMake 3.26+
- C++20 compiler
- Vulkan-capable GPU and driver
- vcpkg

The repository contains a versioned `vcpkg.json` manifest for its dependencies.

## Build

Install dependencies:

```powershell
vcpkg install --triplet x64-windows
```

Configure and build on Windows:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" `
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

Run:

```powershell
.\build\bin\Release\vulkan-3dgs.exe
```

After startup, select the training device from the `Training GPU` combo in the `Training` panel. The presentation device continues to own the Surface and ImGui, while the training device requires compute, push descriptors, and float32 atomic add but does not require graphics, presentation, or a swapchain. Switching the training GPU releases existing training GPU resources and requires the dataset to be loaded again; the application does not restart.

A training GPU can also be selected at startup by Vulkan enumeration index, device name, or UUID:

```powershell
.\build\bin\Release\vulkan-3dgs.exe --gpu 1
.\build\bin\Release\vulkan-3dgs.exe --gpu "NVIDIA GeForce RTX 4090"
```

Without `--gpu`, automatic selection prefers a discrete GPU that passes the training requirements.

Ubuntu example:

```bash
sudo apt install vulkan-tools libvulkan1 mesa-vulkan-drivers xorg-dev

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
  -DVCPKG_TARGET_TRIPLET=x64-linux

cmake --build build
./build/bin/Release/vulkan-3dgs
```

## Training Dataset

Training expects a MipNeRF360/COLMAP-style scene containing camera metadata, source images, and optionally `points3D.bin` for Gaussian initialization.

In the application:

1. Select the dataset directory.
2. Choose the image downscale and training schedule.
3. Initialize or start training.
4. Monitor loss, Gaussian count, cache usage, and profiling data.
5. Export the trained scene as PLY.

## Tests

```powershell
ctest --test-dir build -C Debug --output-on-failure
ctest --test-dir build -C Release --output-on-failure
```

## Current Status

- The project is experimental and training behavior is still being aligned with reference 3DGS.
- High Gaussian counts can make densification noticeably slower.
- `.splat`, `.gs`, and `.json` loading are not implemented.
- Training results are currently transferred to the normal renderer through PLY export.

Implementation notes, architecture, cache policies, Vulkan requirements, and known performance bottlenecks are documented in the [Development Guide](docs/DEVELOPMENT.md) and [开发指南](docs/DEVELOPMENT.zh-CN.md).
