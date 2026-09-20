# vulkan-3dgs

[English](README.md) | [简体中文](README.zh-CN.md) | [Development guide](docs/DEVELOPMENT.md) | [开发指南](docs/DEVELOPMENT.zh-CN.md)

`vulkan-3dgs` 是一个实验性的 Vulkan + Slang 3D Gaussian Splatting 应用，可用于渲染标准 3DGS PLY 模型，也可以使用 MipNeRF360/COLMAP 风格数据集训练 Gaussian 场景。

## 功能

- 使用 GPU 排序的 3DGS PLY 渲染
- MipNeRF360/COLMAP 数据集加载
- GPU forward/backward 训练
- Adam 优化、稠密化和修剪
- 自适应图片流与缓存
- 训练验证和性能统计
- PLY 导出

## 环境要求

- CMake 3.26+
- 支持 C++20 的编译器
- 支持 Vulkan 的 GPU 和驱动
- vcpkg

仓库中的 `vcpkg.json` 已对依赖进行版本控制。

## 构建

安装依赖：

```powershell
vcpkg install --triplet x64-windows
```

Windows 配置和构建：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" `
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

运行：

```powershell
.\build\bin\Release\vulkan-3dgs.exe
```

应用启动后可在 `Training` 面板的 `Training GPU` 下拉框中选择训练显卡。显示与 ImGui 继续使用支持 Surface present 的显示设备，训练设备只要求 compute、push descriptor 和 float32 atomic add，不要求 graphics、present 或 swapchain。切换训练显卡会清理旧训练 GPU 资源，数据集需要重新加载，但应用不会重启。

也可以在启动时按 Vulkan 枚举索引、设备名称或 UUID 指定训练显卡：

```powershell
.\build\bin\Release\vulkan-3dgs.exe --gpu 1
.\build\bin\Release\vulkan-3dgs.exe --gpu "NVIDIA GeForce RTX 4090"
```

未指定 `--gpu` 时，程序自动选择通过训练适用性检查的独立显卡。

Ubuntu 示例：

```bash
sudo apt install vulkan-tools libvulkan1 mesa-vulkan-drivers xorg-dev

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
  -DVCPKG_TARGET_TRIPLET=x64-linux

cmake --build build
./build/bin/Release/vulkan-3dgs
```

## 查看器相机

`Camera` 面板提供 `Legacy` 和 `SuperSplat Compatible` 两种渲染配置，两者使用同一个相机控制器。Legacy 保留原来的 3σ Gaussian 核和非预乘 Alpha 混合；兼容配置使用有限归一化核和预乘 source-over 混合。SH 阶数可在 0～3 之间选择。

- 左键拖动：Orbit；Fly 模式下用于转向
- 右键拖动：平移
- 滚轮：Dolly 缩放
- `V`：切换 Orbit/Fly
- `F`：聚焦已加载模型；`Shift+F`：重置视角
- Fly 模式：`W/A/S/D`、`Q/E`，`Shift` 为 10 倍速度，`Alt` 为 0.1 倍速度

共用相机采用无滚转焦点模型、阻尼、长边 FOV 和按场景包围范围拟合的 near/far。相机设置和所选配置会随现有应用设置一起保存。

## 训练数据集

训练使用 MipNeRF360/COLMAP 风格场景，需要相机信息和源图片，并可选使用 `points3D.bin` 初始化 Gaussian。

在应用中：

1. 选择数据集目录。
2. 设置图片缩放和训练调度方式。
3. 初始化或开始训练。
4. 查看 loss、Gaussian 数量、缓存占用和性能数据。
5. 将训练结果导出为 PLY。

## 测试

```powershell
ctest --test-dir build -C Debug --output-on-failure
ctest --test-dir build -C Release --output-on-failure
```

## 当前状态

- 项目仍处于实验阶段，训练行为正在继续向 reference 3DGS 对齐。
- Gaussian 数量较高时，稠密化可能明显变慢。
- `.splat`、`.gs`、`.json` 加载尚未实现。
- 训练结果当前通过 PLY 导出进入普通渲染路径。

内部架构、缓存策略、Vulkan 要求和已知性能瓶颈见 [开发指南](docs/DEVELOPMENT.zh-CN.md)，英文版本见 [Development Guide](docs/DEVELOPMENT.md)。
