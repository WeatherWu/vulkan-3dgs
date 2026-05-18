# vk-gs

Vulkan + 3D Gaussian Splatting 实验项目。当前仓库主要包含一个核心静态库、一个最小应用入口，以及一个用于加载 `insect.ply` 的 Debug 沙盒程序。

## 当前目标

- `vk_gs_core`: 核心静态库，包含 Vulkan 上下文、资源封装、相机、PLY 读取和 Gaussian 渲染器。
- `vk_gs_windows`: `apps/main.cpp` 的最小应用模板，只保留 `main()` 入口。
- `vk_gs_sandbox`: `sandbox/sandbox.cpp` 的调试入口，加载根目录下的 `insect.ply` 并启动 Gaussian 渲染窗口。

Release 默认只构建 `vk_gs_windows` 和核心库。`vk_gs_sandbox` 只用于 Debug 默认构建，也可以手动指定目标构建。

## 目录结构

```text
vk-gs/
├── CMakeLists.txt
├── README.md
├── insect.ply
├── apps/
│   ├── CMakeLists.txt
│   └── main.cpp
├── sandbox/
│   ├── CMakeLists.txt
│   └── sandbox.cpp
├── shaders/
│   └── gaussian_compute_shader/
│       └── src/
│           ├── gaussian_compute_shader.vert
│           ├── gaussian_compute_shader.frag
│           └── gaussian_compute_shader.comp
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
    ├── gaussian_renderer_compute/
    │   ├── gaussian_model.hpp/cpp
    │   ├── gaussian_renderer.hpp/cpp
    │   ├── pipeline.hpp/cpp
    │   └── renderpass.hpp/cpp
    └── utils/
        ├── camera.hpp/cpp
        ├── file_utils.hpp
        └── logger.hpp
```

## 架构概览

`Application` 负责 GLFW 窗口、Vulkan `Context` 初始化和主循环。`Context` 是 Vulkan instance、surface、device 和队列的全局入口。`GaussianRenderer` 继承自 `Renderer`，负责 swapchain、render pass、graphics pipeline、compute pipeline、descriptor set、GPU 深度排序和每帧提交。

渲染路径大致如下：

```text
sandbox.cpp
  -> GaussianModel::loadFromFile("insect.ply")
  -> Application
  -> GaussianRenderer::setRenderData(...)
  -> GPU depth sort compute shader
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

## Shader

GLSL 源码在：

```text
shaders/gaussian_compute_shader/src/
```

CMake 使用 `glslangValidator` 编译为 SPIR-V，并复制到运行目录：

```text
build/bin/<Config>/shaders/
```

当前 shader 包含：

- `gaussian_compute_shader.vert`: 按排序索引读取高斯实例，投影协方差，计算屏幕椭圆和 SH 颜色。
- `gaussian_compute_shader.frag`: 计算高斯 alpha 衰减并输出颜色。
- `gaussian_compute_shader.comp`: bitonic sort，用于按深度排序高斯索引。

## 构建依赖

- CMake 3.26+
- C++20 编译器
- Vulkan headers/library
- `glslangValidator`
- GLFW3
- GLM
- STB headers

项目会优先 `find_package()` 查找依赖；找不到 GLFW/GLM/STB 时，`src/CMakeLists.txt` 里有 FetchContent 回退逻辑。离线环境建议提前通过 vcpkg 或系统包安装依赖，避免配置阶段尝试访问 GitHub。

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

Debug 构建沙盒：

```powershell
cmake --build build --config Debug --target vk_gs_sandbox
```

运行位置示例：

```powershell
.\build\bin\Release\vk_gs_windows.exe
.\build\bin\Debug\vk_gs_sandbox.exe
```

## 当前限制

- `apps/main.cpp` 只是最小模板入口，还没有应用逻辑。
- `sandbox` 依赖根目录的 `insect.ply`，路径当前按构建输出目录相对路径解析。
- `Application::initialize()` 当前在构造函数中调用，不适合依赖派生类虚函数分发。
- `.splat`、`.gs`、`.json` 加载尚未实现。
- GPU 排序和 descriptor 资源重建仍是项目重点维护区域。
- 没有引入 VMA，buffer/image memory 仍为手写分配。

## 日志

日志使用 `utils/logger.hpp`，默认输出 INFO 及以上级别。Debug 构建会把日志级别调到 `DEBUG_VKGS`。

热路径上的每帧 INFO 已经尽量移除，INFO 主要保留启动、设备选择、模型加载、swapchain 初始化和 resize 重建等状态信息。

