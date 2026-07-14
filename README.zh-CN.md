# vulkan-3dgs

[English](README.md) | [简体中文](README.zh-CN.md)

vulkan-3dgs 是一个 Vulkan + Slang 3D Gaussian Splatting 实验项目。当前仓库包含普通 PLY 渲染路径和 MipNeRF360/COLMAP 数据集训练路径，训练端支持 compute forward/backward、Adam 优化、稠密化/修剪和 PLY 导出。

## 当前目标

- `vulkan_3dgs_core`: 核心静态库，包含 Vulkan 上下文、资源封装、相机、PLY 读取和 Gaussian 渲染器。
- `vulkan_3dgs_app`: `apps/main.cpp` 的主应用入口。Windows 和非 Windows 平台输出名均为 `vulkan-3dgs`。

Release 默认只构建主应用和核心库。

## 目录结构

```text
vulkan-3dgs/
├── CMakeLists.txt
├── README.md
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

## 架构概览

`Application` 负责 GLFW 窗口、Vulkan `Context` 初始化和主循环。`Context` 是 Vulkan instance、surface、device 和队列的全局入口。`GaussianRenderer` 继承自 `Renderer`，负责 swapchain、render pass、graphics pipeline、compute pipeline、descriptor set、GPU 深度排序和每帧提交。

渲染路径大致如下：

```text
apps/main.cpp
  -> Application
  -> GaussianRenderer::setRenderData(...)
  -> GPU keygen + vulkan_radix_sort
  -> instanced quad graphics pipeline
  -> swapchain present
```

## Gaussian 数据

`GaussianModel` 当前支持 3DGS 风格的 binary little endian `.ply`：

- 位置：`x`, `y`, `z`
- 颜色：`f_dc_0..2`, `f_rest_0..44`
- 不透明度：`opacity`
- 尺度：`scale_0..2`
- 旋转：`rot_0..3`

读取时会恢复 alpha、scale，并根据 scale + quaternion 构建协方差矩阵。`.splat`、`.gs`、`.json` 目前只是预留，尚未实现。

## Training

`GaussianTraining` 是独立于普通 `GaussianRenderer` 的 compute 训练路径。它使用 MipNeRF360/COLMAP 风格数据集，加载真实图片尺寸，按 COLMAP 相机内参缩放后训练 Gaussian 参数。

训练路径包含：

- compute forward tile renderer
- loss 和 backward passes
- Adam-style optimizer
- densification / pruning
- PLY export

训练 UI 支持两种调度模式：

- `Sequential`: 按数据集图片顺序逐帧训练，没有固定总迭代停止。
- `3DGS Random`: 按原始 3DGS 的 viewpoint stack 策略随机抽图，抽完一轮重填，并在 30000 iterations 自动停止。

稠密化统计使用 shader float atomics，因此设备必须支持并启用：

- `VK_EXT_shader_atomic_float`
- `VK_EXT_shader_atomic_float2`

如果 GPU/驱动不支持 buffer float32 atomic add/min-max，设备适配性检查会失败，训练路径不会以不可靠状态继续运行。

## Shader

Slang 源码在：

```text
shaders/gaussian_compute_shader/slang/
shaders/training_shader/slang/
```

CMake 使用 `slangc` 编译为 SPIR-V，并复制到运行目录：

```text
build/bin/<Config>/shaders/
```

当前 shader 包含：

- `gaussian_common.slang`: 共享 Gaussian 数据结构、half-packed SH 解包和 SH 评估。
- `gaussian_compute_shader.vert.slang`: 按排序索引读取高斯实例，投影协方差，计算屏幕椭圆和 SH 颜色。
- `gaussian_compute_shader.frag.slang`: 计算高斯 alpha 衰减并输出颜色。
- `radix_keygen.comp.slang`: GPU 可见性裁剪、排序 key 生成和 indirect draw instance count 写入。
- `training_shader/slang/passes/*`: 训练 forward、loss、backward、optimizer、densify/prune 和 preview packing passes。

## 构建依赖

- CMake 3.26+
- 支持 C++20 的 C/C++ 编译器
- Vulkan headers/library
- `slangc`（可通过 vcpkg 的 `shader-slang` 获取）
- GLFW3
- GLM
- STB headers
- ImGui
- COLMAP
- ImGuiFileDialog（仓库内 `third_party/ImGuiFileDialog`）

所有 CMake package 依赖都应由 vcpkg 或等价的本地包安装提供。构建系统不再使用 `FetchContent_Declare()` 回退逻辑，缺少依赖时会在 CMake 配置阶段直接报错。

本仓库包含 vcpkg manifest，并固定到以下 registry baseline：

```bash
301856f5f2824f788a3ffa6332293861cccd23b4
```

使用下面命令安装可复现的依赖集合：

```bash
vcpkg install --triplet x64-windows
```

当前 baseline 解析到的 vcpkg 包版本：

- `vulkan`: `2023-12-17`
- `glfw3`: `3.4#1`
- `glm`: `1.0.3`
- `imgui[glfw-binding,vulkan-binding]`: `1.92.8#1`
- `stb`: `2024-07-29#1`
- `shader-slang`: `2026.7.1`
- 关闭 default features 的 `colmap`: `3.12.6#1`

仓库内 third-party submodule 版本：

- `third_party/ImGuiFileDialog`: `https://github.com/aiekick/ImGuiFileDialog.git` at `d0e97b2adc3d3452d72c750c7305dc0291acd052`
- `third_party/vulkan_radix_sort`: `https://github.com/jaesung-cs/vulkan_radix_sort.git` at `7b9912bb827fcc569854e45b43748fd27bc5dde3`

训练稠密化需要 GPU/驱动支持 Vulkan float atomic 扩展：

- `VK_EXT_shader_atomic_float`
- `VK_EXT_shader_atomic_float2`

可用 `vulkaninfo` 检查 Linux 驱动是否暴露这些扩展。

## 构建

Visual Studio 生成器示例：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
```

如果 Vulkan 或 vcpkg 依赖没有被自动找到，可以显式传入路径，例如：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" `
  -DCMAKE_TOOLCHAIN_FILE=C:/Users/weath/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DVulkan_INCLUDE_DIR=C:/Users/weath/vcpkg/installed/x64-windows/include `
  -DVulkan_LIBRARY=C:/Users/weath/vcpkg/installed/x64-windows/lib/vulkan-1.lib
```

Release 默认构建：

```powershell
cmake --build build --config Release
```

运行位置示例：

```powershell
.\build\bin\Release\vulkan-3dgs.exe
```

Ubuntu + vcpkg 示例：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
  -DVCPKG_TARGET_TRIPLET=x64-linux

cmake --build build
```

Ubuntu 上建议安装 Vulkan loader、driver 和基础 X11 开发包：

```bash
sudo apt install vulkan-tools libvulkan1 mesa-vulkan-drivers xorg-dev
```

vcpkg manifest 依赖安装示例：

```bash
vcpkg install --triplet x64-linux
```

非 Windows 平台主应用输出名为：

```bash
./build/bin/Debug/vulkan-3dgs
```

## 当前限制

- `Application::initialize()` 当前在构造函数中调用，不适合依赖派生类虚函数分发。
- `.splat`、`.gs`、`.json` 加载尚未实现。
- GPU 排序和 descriptor 资源重建仍是项目重点维护区域。
- 没有引入 VMA，buffer/image memory 仍为手写分配。
- `3DGS Random` 模式匹配原始 3DGS 的随机视角栈和 30000 iteration 调度，但 forward/backward 数学仍是项目实现，尚未声明与 reference 3DGS 完全一致。
- 训练结果当前通过 PLY 导出进入普通渲染路径，主视口不会自动实时渲染训练 buffer。

## 日志

日志使用 `utils/logger.hpp`，默认输出 INFO 及以上级别。Debug 构建会把日志级别调到 `DEBUG_VULKAN_3DGS`。

热路径上的每帧 INFO 已经尽量移除，INFO 主要保留启动、设备选择、模型加载、swapchain 初始化和 resize 重建等状态信息。
