# 开发指南

本文面向修改 Vulkan 渲染器、Gaussian 训练路径、查看器或应用架构的贡献者。安装和基础 UI 用法请先阅读[项目说明](../README.zh-CN.md)。

代码、CMake target、测试和 Slang 声明是事实来源。本文只描述当前实现；尚未实现的优化只放在[已知限制](#已知限制)，不混入运行流程。

## 构建与验证

### 前置条件

- CMake 3.26 或更高版本
- 支持 C++20 的编译器
- 支持 Vulkan 的 GPU 与驱动
- 已安装仓库 manifest 的 vcpkg
- 由 `shader-slang` vcpkg 包提供的 Slang 编译器

按照 README 完成配置。日常开发复用现有 `build` 目录：

```powershell
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

交付性能敏感修改前，还应运行：

```powershell
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
cmake --build build --config Debug --target quality-new
git diff --check
```

Windows 可执行文件位于 `build/bin/<Config>/vulkan-3dgs.exe`，其他平台位于 `build/bin/<Config>/vulkan-3dgs`。生成的 shader 会复制到 `build/bin/<Config>/shaders/`。

## 架构

```text
Application
├─ Window / Context / Renderer 路由
├─ ViewerController + ViewerPanel
└─ TrainingController + TrainingPanel
   └─ GaussianTraining façade
      ├─ TrainingFrameScheduler
      ├─ TrainingImageRuntime
      ├─ TrainingStepExecutor
      ├─ TrainingValidationService
      ├─ TrainingProfilingService
      ├─ TrainingModelIO
      ├─ GaussianForwardRenderer
      ├─ GaussianBackwardRenderer façade
      │  ├─ BackwardLossPass
      │  ├─ PixelTo2DGSDispatcher
      │  ├─ ProjectionOptimizerPass
      │  └─ BackwardValidationPass
      └─ GaussianDensificationRenderer
```

主要依赖方向为：

```text
Application / UI
    -> controller 或 renderer façade
        -> 专用 training/graphics service
            -> Vulkan、image、buffer wrapper
```

### 所有权边界

| 所有者 | 职责 |
| --- | --- |
| `Application` | Window/context 生命周期、输入采样、renderer 选择和帧路由 |
| `ViewerController` | PLY 模型、相机策略、变换、渲染设置和查看器配置持久化 |
| `TrainingController` | 训练 GPU、数据集生命周期、worker、不可变 snapshot、计时、benchmark 和导出 |
| `TrainingPanel` | 训练 ImGui 和文件对话框状态；向 controller 分发类型化操作 |
| `GaussianTraining` | 数据集/模型语义和单次训练 step 编排 |
| `TrainingStepExecutor` | prepare/main command buffer、submit/wait、tile count、validation copy 和 densification adopt |
| `TrainingBuffers` | 训练 GPU buffer 和 descriptor buffer info |
| `GaussianRenderer` | 普通渲染 façade，组合 frame、resource、sorter、pipeline 和 recorder 模块 |

UI 不访问 `TrainingBuffers`、command buffer、descriptor set 或 live training renderer 状态。异步训练期间，worker 是 `GaussianTraining` 的唯一常规修改者。

## 目录结构

```text
apps/                               可执行程序入口
src/app/                            Application 组合与 GLFW Window
src/viewer/                         查看器 controller、panel、settings、camera
src/graphics/                       PLY 模型与普通 graphics 渲染
src/render/                         两条渲染路径共享的 renderer 接口
src/training/app/                   训练 controller、settings、snapshot
src/training/ui/                    训练控件、诊断和文件对话框
src/training/core/                  GaussianTraining 与 compute service
src/training/core/backward/         反向 loss、dispatch、optimizer、validation pass
src/training/cache/                 device-local target image cache
src/image/                          解码、磁盘缓存、主存缓存和流式加载
src/context/                        Vulkan instance/device/surface context
src/vulkan/                         Vulkan 资源 wrapper
shaders/gaussian_compute_shader/    普通渲染 shader
shaders/training_shader/            训练 compute shader 与 ABI 声明
tests/                              CPU/service/架构回归测试
cmake/                              可选代码质量 target
tools/                              质量评测工具
```

## Application 与查看器流程

每帧执行：

1. `Application::tick()` 轮询窗口事件并调用 `TrainingController::tick()`。
2. `Application` 将 GLFW/ImGui 输入采样为 `ViewerInputState`。
3. `ViewerController` 只更新一次 Orbit/Fly 相机策略并准备 `ViewerRenderData`。
4. `ViewerPanel` 与 `TrainingPanel` 通过 renderer 的 ImGui callback 绘制。
5. `GaussianRenderer` 录制 GPU sort、indirect splat draw 和 ImGui，然后呈现 swapchain image。

`GaussianRenderer` 由以下模块组成：

- `GraphicsFrameRuntime`：acquire、fence、semaphore、submit、present、resize；
- `GraphicsPipelineSet`：render pass 和 Legacy/Compatible pipeline；
- `GraphicsSplatResources`：打包模型 SSBO、UBO 和 descriptor；
- `GraphicsSplatSorter`：key generation、radix sort、indirect draw 参数；
- `GraphicsCommandRecorder`：render pass 与 draw command 录制。

Legacy 和 SuperSplat Compatible 共用同一相机。Profile 只改变 kernel 支撑与混合语义，不改变导航行为。

## 配置持久化

`TrainingSettingsStore` 和 `ViewerSettingsStore` 共用一个 `training-settings.cfg`，但各自拥有互不重叠的 key。默认路径为：

- Windows：`%APPDATA%/vulkan-3dgs/training-settings.cfg`
- 设置了 `XDG_CONFIG_HOME` 的 Linux：`$XDG_CONFIG_HOME/vulkan-3dgs/training-settings.cfg`
- Linux fallback：`$HOME/.config/vulkan-3dgs/training-settings.cfg`
- 最终 fallback：当前目录中的 `.vulkan-3dgs-training-settings.cfg`

训练开始时会生成不可变 `TrainingRunConfig`。Worker 运行期间的 UI 修改不会改变当前 live run。

## 数据集与模型输入

训练加载器读取未畸变的 MipNeRF360/COLMAP 风格场景：

```text
scene/
├─ sparse/0/cameras.bin
├─ sparse/0/images.bin
├─ sparse/0/points3D.bin       可选
└─ images、images_2、images_4 或 images_8/
```

也支持 `sparse/` 或根目录 COLMAP 二进制文件。加载器通过 STB 读取真实图片尺寸，要求图片尺寸一致，并将 PINHOLE/SIMPLE_PINHOLE 内参缩放到所选图片目录。训练路径不会校正畸变相机模型。

可加载图片按名称排序后，零起始索引 `0、8、16、...` 留作验证；只有一张图片的场景仍全部用于训练。Benchmark frame索引指向训练子集，不是原始完整COLMAP列表。评测held-out质量时必须匹配这一拆分。

存在 `points3D.bin` 时，`TrainingModelIO` 使用稀疏点的位置和 RGB，并通过 CPU 3 近邻估计 scale。没有稀疏点时，会在相机范围附近采样配置数量的随机 Gaussian。

查看器和导出器使用 binary little-endian 3DGS PLY，包含 position、opacity、scale、quaternion rotation、SH DC 和 SH rest 字段。

## 图片流式加载与缓存

训练图片保存为打包的线性 `RGBA8`，不会永久保存为 float 数组。

1. `ImageStreamer` 解析源图片、解码并管理主存 LRU entry。
2. `ImageDiskCache` 使用每 chunk 16 层的 KTX2 array，默认磁盘配额为 20 GiB。
3. `DeviceImageCache` 保存对齐的 device-local image slot；支持时使用带 timeline semaphore 的三槽上传 ring。
4. `TrainingImageRuntime` 选择 target descriptor，并把待等待的上传 semaphore/value 交给 `TrainingStepExecutor`。

自动主存预算最多使用当前可用内存的 10%，同时保留 2 GiB。设备缓存至少保留一张图片，为训练/稠密化预留显存，并将图片缓存限制在报告的 device-local heap budget 的四分之一以内。

设置 `VULKAN_3DGS_SYNC_IMAGE_UPLOAD=1` 可强制使用同步上传 fallback，以排查竞争问题。

## 训练生命周期

`TrainingController` 拥有训练设备和普通 `std::jthread`。开始训练会依次执行数据集验证/加载、必要时初始化模型、初始化 renderer、复制不可变 run config 并启动 worker。

普通模式每 100 次迭代发布一次 `TrainingUiSnapshot`。Pure Training 执行相同 GPU 工作，但不发布中间详细 snapshot、降低 presentation 频率、报告 wall time，并在结束后导出配置的 PLY。暂停、切换 GPU、更换数据集和关闭程序都会先停止并 join worker。

调度模式：

- `3DGS Random`：无放回随机采样，默认在 30,000 次迭代停止；
- `Sequential`：按顺序选择帧且不会自动停止，只用于调试。

`Steps/Frame` 不是普通训练调度器。Benchmark steps-per-frame 只用于 fixed workload benchmark。

## 单次训练迭代

`GaussianTraining::trainStep()` 准备不可变 step 输入，并把 GPU 执行交给 `TrainingStepExecutor`。

### Prepare 提交

1. 上传或选择 target image。
2. 投影 active Gaussian。
3. 计算每 Gaussian 的 tile 覆盖数量。
4. 执行分层 prefix pass。
5. 把 16 字节 tile counter 复制到持久映射内存。
6. 提交并等待 prepare fence。
7. 当所需 item 超过 capacity 时扩展 tile buffer。

### Main 提交

1. 按 Gaussian 连续写入 tile item。
2. 通过两次稳定的 32-bit radix pass 排序 `(tile, depth)`。
3. 重建每 tile 的 `[start,count]`。
4. Adaptive Spatial先选择parent并追加稳定child列表，再按选定tile布局合成渲染图片。
5. 执行 loss、backward clear、loss-to-pixel、Pixel-to-2DGS 和 projection/optimizer。
6. 按需执行 validation 与 densify/prune。
7. 提交并等待选定 fence，收集紧凑统计，并在需要时 adopt densified buffer。

当前实现保留 prepare readback，因为录制 main command 前必须知道 tile-item storage 是否足够。

## 反向路径

`GaussianBackwardRenderer` 保留 `BackwardRenderer` API 和 profiling 顺序，把 command recording 分发给：

| 模块 | Pipeline 与行为 |
| --- | --- |
| `BackwardLossPass` | Loss reduction、backward buffer clear、loss-to-pixel |
| `PixelTo2DGSDispatcher` | Direct、Workgroup Shared、Subgroup、Adaptive、Tile Gaussian Atomic、VkSplat Per-Splat、VkSplat Tensor |
| `ProjectionOptimizerPass` | 2DGS-to-3DGS、融合 projection/Adam、独立 Adam |
| `BackwardValidationPass` | Gaussian validation 与最终紧凑 reduction |

Auto Pixel-to-2DGS 只在 Direct 与 subgroup-adaptive 路径之间选择。不支持的显式模式会回退 Direct；Per-Splat 和 Tensor 不会被 Auto 选择。

### 训练tile

`Training Tile`支持Fixed8、Fixed16（默认）和实验性Adaptive Spatial。Adaptive要求支持的Tensor或Per-Splat backward，否则训练使用Fixed16。它只排序一次16×16 parent列表，在parent item数量四分之一的child预算内选择有收益的parent，追加保持深度顺序的8×8 child列表，并通过GPU元数据调度细分工作；不增加第二次全局child排序或CPU计数回读。

`Adaptive Split Threshold`默认256。`Adaptive Tensor Child Batch`支持8/16/32，默认8，只改变child。下面的Fixed16 Tensor控件不改变Adaptive parent或child内核。已有Adaptive试验不能支持替换Fixed16默认值。

### Fixed16 Tensor控件

这些A/B控件只在停止时修改，使用相同descriptor ABI和独立编译pipeline。请求配置会持久化；`Active Tensor ...`和已完成benchmark标签报告实际经过能力保护的路径。

| 控件 / 持久化key | 默认值 | 生效条件 |
| --- | --- | --- |
| `Tensor Subgroup Reduction (Experimental)` / `tensor_subgroup_reduction` | Off | Fixed16＋VkSplat Tensor；native subgroup32、compute BASIC/SHUFFLE及基础Tensor支持 |
| `Tensor Pixel Gradient Source` / `tensor_global_pixel_grad` | Shared（0） | Global（1）要求上面的subgroup路径生效 |
| `Tensor RGB Latency Hiding (Experimental)` / `tensor_rgb_latency_hiding` | Off | 要求上面的Global路径生效 |

Tensor能力检查仍要求至少45 KiB共享内存。Fixed16 Shared归约每工作组使用45824 B，Subgroup归约36612 B，Subgroup＋Global像素梯度33540 B。Global读取检查像素的两个坐标。RGB调度变体保持相同共享存储预算，改变RGB消费与几何累加的相对位置，目前没有明确的稳定提速证据。Tensor名称不表示使用Tensor Core MMA。

Fixed16/Tensor/Subgroup/Global/Baseline RGB是手动选择的后续实验工作基线，不是程序默认值变更。Global变体通过了提供的单步梯度比较，并在匹配单帧ABA中改善了耗时；这不等于跨场景质量或整轮训练性能验证。寄存器数量与`# Warp`属于编译器/驱动观测，后者是理论驻留上限。

### Loss与SSIM

Loss pass计算L1/DSSIM并保存`SsimBackwardState`，loss-to-pixel消费这些系数。两者仍遍历二维11×11邻域，使用零填充和固定归一化。可分离SSIM和Checkpointed Warp Backward尚未实现。

### Projection与优化器

Projection backward 与 Adam 默认融合。设置 `VULKAN_3DGS_SEPARATE_PROJECTION_OPTIMIZER=1` 可用于 A/B 调试；设置 `VULKAN_3DGS_COMPUTE_BACKWARD_CLEAR=1` 可用 compute fallback 替代常规 `vkCmdFillBuffer` 清理。

## Buffer 与 descriptor ABI

`TrainingBuffers` 分别管理 active Gaussian、densification workspace/output、tile item 和图片 extent 的 capacity。主要分组为：

| 分组 | 示例 |
| --- | --- |
| 参数 | Gaussian 参数、梯度和 Adam state |
| 投影 | Projected Gaussian 和 projected gradient |
| Tile | 覆盖范围、prefix scratch、sort key/index/storage、tile range |
| 像素 | Rendered/target color、loss、pixel gradient、blend/SSIM state |
| 稠密化 | Densification state、输出参数/Adam、counter |
| Validation | Pixel partial、Gaussian partial、112 字节最终结果 |
| 辅助 | Camera uniform、counter、preview instance |

所有训练 shader 使用 descriptor set 0。重要 binding 为：

| Binding | 资源 |
| ---: | --- |
| 0/1/2 | Gaussian 参数、梯度、Adam state |
| 3/4/5 | Projected Gaussian、排序后的 tile item、tile range |
| 6/7/8 | Rendered color、打包 target color、loss |
| 11 | Camera uniform |
| 12/13/14 | Pixel gradient、projected gradient、blend state |
| 16 | Densification state |
| 17/18/19 | Densified 参数/Adam 与 counter |
| 20–24 | Tile sort key、index、scratch、排序输出 |
| 25–27 | Densification candidate buffer |
| 28 | SSIM backward state |
| 29/30/31 | Pixel partial、Gaussian partial、validation 最终结果 |
| 32/33 | 每 Gaussian tile range 与 prefix scratch |
| 34/35/36/37 | Adaptive split flag、child count、item mask、child offset |
| 38/39/40/41 | Adaptive parent prefix、block scratch/offset、active parent、child dispatch元数据 |

Binding 数字与 C++/Slang struct layout 都是 ABI。修改时必须同步更新 C++ descriptor write、Slang `[[vk::binding(N, 0)]]` 和 static assertion。`TrainingBuffers` descriptor getter 返回值；赋给 `pBufferInfo` 前必须保存在生命周期稳定的局部变量中。

## 同步

`TrainingStepExecutor` 拥有训练 command pool、command buffer 和 fence。Worker 执行同步 GPU iteration：

```text
record prepare -> submit -> wait -> read tile count / resize
record main    -> submit -> wait -> collect validation / adopt densification
```

各 pass 为 storage-buffer hazard 显式录制 Vulkan barrier。Transfer clear 使用 `TransferWrite -> ComputeShader`；shader 输出按下一个操作设置 compute、transfer 或 indirect access。Backward pass 对象只录制命令，不提交 queue，也不拥有 fence。

## Validation、profiling 与 benchmark

Validation 在前五次迭代执行，之后按配置间隔执行。GPU 把 pixel 和 Gaussian 诊断归约成一个固定 112 字节结果。`TrainingValidationService` 拥有三槽 host-coherent readback ring、轮询 fence，并生成当前/历史统计。常规 validation 不下载完整 rendered、loss 或 Gaussian buffer。

`TrainingProfilingService` 拥有 timestamp query pool 和 CPU/GPU sample 累计；renderer/pass 只观察 query pool。GPU 阶段包括 projection、tile preparation、composite、loss、backward 子 pass、validation、optimizer 和 densification。

使用 `Start Fixed Benchmark` 进行 kernel A/B。它固定一帧，冻结模型与迭代状态，关闭 optimizer 和 densification，在 warmup 后清空统计，并只在最后一个 measured step 执行 validation。它不是端到端训练 benchmark；wall-clock 对比应使用 Pure Training。

### 梯度检查与计时

1. 暂停在同一模型上，保持benchmark frame、forward设置和extent不变。用warmup 0/measured 1，每次完成后导出梯度。
2. 用本地辅助脚本比较实际路径：

   ```powershell
   python scripts/compare_projected_gradients.py baseline.bin candidate.bin
   python -m unittest discover -s tests -p "test_*.py"
   ```

3. 对新性能假设使用warmup 200/measured 1000做ABA，记录average，不在中间恢复普通训练。确认measured进度达到目标；`n=30000`整轮训练累计统计不是完成的1000步Fixed Benchmark。

`Export Benchmark Gradients`写入配置输出目录下的`benchmark_gradients/`。Fixed16 Tensor文件名区分归约、像素梯度来源和Global RGB schedule；Adaptive Tensor文件名区分child batch。重名文件使用数字后缀。当前写v5：80字节little-endian header，包含magic `V3DGGRD1`、16个uint32和loss/alpha两个float32，之后是每Gaussian 12个float32梯度分量和一个membership字节。比较器读v1–v5，报告实际路径，并检查repeat捕获的路径一致。除退出状态外还要看mismatch计数，不能只为通过而放宽容差。

`Export Adaptive Tile Diagnostics`在`benchmark_tiles/`写CSV和配套mask文件。Adaptive构建计时嵌套于sort/ranges；parent/child backward与`Tile-local backward`嵌套于Pixel-to-2DGS，不能重复相加。

当前ignore规则排除`scripts/`、捕获输出、`plan.md`和`.codex/`。只传输受Git跟踪的源码时，需显式同步本地辅助脚本、计划和技能；上述命令要求这些本地脚本存在。

### Shader源码profiling

需要源码关联的GPU Trace时，开启默认OFF的CMake选项：

```powershell
cmake -S . -B build -DVULKAN_3DGS_TRAINING_SHADER_DEBUG_INFO=ON
cmake --build build --config Release
```

Linux单配置构建还需设置预期的`CMAKE_BUILD_TYPE`，例如Release。该选项为训练shader添加Slang `-g2`，通过依赖stamp在切换后重编译；它不是环境变量。

使用`ngfx`启动脚本和real-time shader profiling，选择实际compute设备/时间范围，在benchmark进度递增时捕获。绿色源码关联不表示采到了执行样本。零样本和缺少debug信息是两个问题，Show Inactive不能补回未捕获的工作负载。普通计时与可能控制GPU时钟的Nsight采集分开。归因LGSB/SHRTSB前展开调用点；采样行可能是等待其他位置所产生数据的消费指令，不能直接认定load、spill或bank conflict。

## 稠密化与优化器

默认稠密化从第 500 次迭代开始，在 15,000 次停止，每 100 次执行一次，并每 3,000 次重置 opacity。支持 clone、split、opacity pruning、screen/world-size pruning、局部 split offset、scale shrink 和 Adam state 处理。

`Densification Gradient`选择Visible Support（默认，`densification_gradient_mode=0`）或Compact（Projected Exposure，1）。这是两种已实现的统计模式，不是连续support-ratio beta；性能和质量对比必须保持该模式一致。

Adopt 时 active/output 参数和 Adam buffer 进行 ping-pong。Workspace capacity 按几何比例增长并向 16,384 Gaussian 对齐。支持 `VK_EXT_shader_atomic_float` 时使用 float atomic add；不支持 float min/max atomic 时使用 uint-radius shader 变体完成 radius-max fallback。

## 环境变量

| 变量 | 作用 |
| --- | --- |
| `VULKAN_3DGS_ALLOW_CPU_VULKAN=1` | 允许 CPU Vulkan device fallback |
| `VULKAN_3DGS_SYNC_IMAGE_UPLOAD=1` | 强制同步 target image upload |
| `VULKAN_3DGS_COMPUTE_BACKWARD_CLEAR=1` | 使用 compute backward-clear fallback |
| `VULKAN_3DGS_SEPARATE_PROJECTION_OPTIMIZER=1` | 为 A/B 关闭融合 projection/optimizer |

## Shader 与质量检查流程

每个新 shader 都必须在顶层 `CMakeLists.txt` 中通过 `compile_slang_shader(...)` 或 `compile_training_shader(...)` 显式注册。

质量 target 只覆盖阶段化重构模块，不默认处理 legacy 或第三方源码：

```powershell
cmake --build build --config Debug --target format-new
cmake --build build --config Debug --target check-format-new
cmake --build build --config Debug --target clang-tidy-new
cmake --build build --config Debug --target clang-tidy-advisory-new
cmake --build build --config Debug --target quality-new
```

`clang-tidy-new` 强制检查 `bugprone-*`。Advisory target 报告 `modernize-*`、`cppcoreguidelines-*` 和 `bugprone-*`，但不会使构建失败。Wrapper 逐个分析源文件、过滤第三方摘要噪声，并保留真实诊断和退出码。

## 排障

- **Start Training 不可用：**阅读状态文本，并检查数据集验证、所选 GPU、输出名称和 settings validation。
- **模式回退 Direct：**检查 active Pixel-to-2DGS mode、设备 subgroup 和 shared-memory 能力。
- **训练暂停时间长：**先比较 CPU `Main submit/wait`、`Validation`、`Densify adopt` 与 GPU `Densify/prune`，再修改同步逻辑。
- **Linux loader 失败：**比较 `ldd`、`vulkaninfo`、`VK_LAYER_PATH`，并尝试去除继承的 `LD_PRELOAD`。
- **MSVC C1041/PDB 冲突：**先停止重复 build；只有多个编译进程确实共享同一个 PDB 时才使用 `/FS`。

## 已知限制

- 训练数学是项目实现，不能宣称与 reference 3DGS 完全一致。
- 每次训练迭代仍会等待 prepare 和 main submission。
- Tile sort 使用两次稳定的 32-bit radix pass，而非单一 packed key。
- Loss 与 loss-to-pixel 仍是两个 pass。
- 可分离SSIM、Checkpointed Warp Backward、像素级EarlyCull/scanline原型、GPU SoA/AoSoA参数存储和GPU-driven count/sort是后续方向，不是当前功能。
- Pixel backward kernel 由用户选择，Auto 不选择 VkSplat。
- RGB latency hiding尚无明确稳定收益；三个Tensor A/B开关仍默认关闭，也不改变Adaptive内核。
- 普通训练计划完成后，同步路径的training-complete检查可能提前终止新启动的多步Fixed Benchmark；独立生命周期修复前，应使用更早的冻结检查点并确认进度。
- 完整 Gaussian gradient buffer 仍为 fallback、optimizer-disabled 和 densification 路径保留。
- Densification 仍可能产生较大的分配和带宽峰值。
- 训练只接受未畸变 PINHOLE/SIMPLE_PINHOLE 相机，不处理 target alpha mask。
- 优化路径主要在 RTX 3090 上验证，尚无完整 NVIDIA/AMD/Intel 性能矩阵。
