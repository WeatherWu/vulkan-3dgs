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
