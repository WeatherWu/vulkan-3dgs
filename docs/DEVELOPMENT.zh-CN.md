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
4. 构建选定的adaptive列表，并按实际tile布局合成渲染图片。
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

实验分支提供Fixed8、Fixed16（默认）及Adaptive Spatial。Adaptive要求支持的Tensor或Per-Splat，否则选择Fixed16。实现排序一次16×16 parent，在parent item四分之一预算内追加深度稳定的8×8 child列表，通过间接dispatch处理active child。Split threshold默认256，child Tensor batch可选8/16/32、默认8。可选性不代表质量或性能收益已经验收。

### Fixed16 Tensor控件

控件只在训练停止时编辑，请求配置持久化；active标签和benchmark导出记录能力检查后实际生效的路径。

| 控件 / key | 默认值 | 生效条件 |
| --- | --- | --- |
| Tensor Subgroup Reduction / `tensor_subgroup_reduction` | Off | Fixed16＋Tensor、native subgroup32、compute BASIC/SHUFFLE及基础Tensor支持 |
| Tensor Pixel Gradient Source / `tensor_global_pixel_grad` | Shared | Global要求上面的subgroup路径生效 |

Tensor能力检查要求至少45 KiB共享内存。Fixed16 Shared、Subgroup、Subgroup＋Global声明共享存储分别为45824、36612、33540字节。Global读取检查像素的两个坐标。Tensor名称表示工作组织方式，不表示Tensor Core MMA。
RGB Latency Hiding是额外默认关闭的实验开关，要求Subgroup/Global，以独立pipeline改变RGB消费顺序；可用性不代表收益已经验收。

### Loss与SSIM

SSIM使用11×11 Gaussian窗口、零填充和固定归一化。

| ID | Loss | Pixel backward | Scratch |
| ---: | --- | --- | ---: |
| 0（默认） | 二维 | 二维系数邻域 | 无 |
| 1 | 横向矩、纵向loss | 横向系数、纵向梯度 | 80字节/像素 |
| 2 | 二维 | 横向系数、纵向梯度 | 48字节/像素 |

模式1为全可分离实验，模式2保留二维forward loss/state。BackwardLossPass拥有dispatch/barrier，TrainingBuffers拥有按extent/mode分配的存储；所有工作都计入对应profiling区间。控件仅停止时编辑并持久化，非法API值报错，未知配置值回到二维。benchmark/export记录实际路径，不使用待应用的UI选择。

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
| 34–41 | 实验性adaptive构建与dispatch buffer |
| 42 | 可选SSIM scratch |
| 43/44 | Checkpoint元数据与prefix scratch |
| 45–48 | Checkpoint状态bank |
| 49/50 | Checkpoint control与task |

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

冻结同一模型/帧，warmup0、measured1后导出：

```powershell
python -B scripts/compare_projected_gradients.py baseline.bin candidate.bin
python -B -m unittest discover -s tests -p "test_*.py"
```

writer输出V3DGGRD1 v9，header108字节，后接每Gaussian projected梯度/membership及每像素float4梯度；reader支持v1–v9。header记录实际tile/pixel/composite/SSIM路径、Tensor flags、checkpoint layout/task/fallback及pixel-culling方法，不使用待应用UI选择。发生skip/fallback/error的checkpoint运行不能导出为纯checkpoint捕获。

先比较数值再计时，repeat保持实际路径一致，跨pixel内核比较须显式允许。profiling捕获、有界计时及普通/Pure Training属于不同范围；嵌套stage不能重复相加。

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

## 实验方法

本分支包含可选择或离线实验；实现存在不代表数值等价或性能已经验收。

- Checkpoint状态诊断：`Run Checkpoint State Diagnostic`保留原Tensor backward并构建反向bucket边界状态/task；`Run Checkpoint Warp Benchmark`执行checkpoint warp内核。两者要求已有Fixed16/workgroup/Tensor Subgroup/Global基线。mode7仅用于benchmark，fallback/error必须可见。
- Pixel EarlyCull：默认关闭的训练开关和独立冻结benchmark请求。guarded FP32谓词须满足能力/路径检查，不支持请求使用可观测回退，不改变forward或SSIM；不假定训练质量/性能等价。
- Scanline导出：`Export Scanline Inputs`在停止状态保存冻结projected数组、排序items/ranges、BlendState及pixel梯度到`benchmark_scanline/`。`training_checkpoint_gpu_tests`接受`--scanline-capture`、`--row-empty-r0-capture`、`--row-empty-capture`，不回写训练梯度。算术假设和限制见[分类器参考](training-row-empty-contract.md)。
- Optimizer捕获、回放及流式Adam见[优化器指南](optimizer-diagnostics.md)，捕获会实际训练一步。
- 三时间戳调度及私有排序探针见[调度指南](scheduling-diagnostics.md)，未实现GPU-driven训练。

应用在 **Application Settings** 保存下次启动的验证层偏好（两种构建的初始默认值均为on）；`--validation on|off`只覆盖当前进程，不写回保存值。面板显示requested／actual及重启提示，修改偏好不重建Vulkan实例。偏好在创建实例前应用，可用性检查后记录requested/enabled；层不可用时保留告警并禁用行为。不改变训练数值检查、logger或shader flags。正确性检查开启，性能运行与profiling分开；独立GPU runner保留自己的严格验证选项。

非持久化选项`--training-smoke-steps N --training-smoke-report NEW.json`让新初始化的Random训练在诊断边界暂停，保持30000步日程不变。该模式要求core／sync验证并保存生命周期报告，参见[有界真实训练烟测](training-smoke.md)。

持久化的 **Faster-GS Sorting (Experimental)** 复选框默认关闭，经不可变训练配置传递。UI单独显示最近一步的实际排序，参见[UI训练与完整对照](training-ui.md)。

## 已知限制

- 训练数学是项目实现，不能宣称与 reference 3DGS 完全一致。
- 每次训练迭代仍会等待 prepare 和 main submission。
- Tile sort 使用两次稳定的 32-bit radix pass，而非单一 packed key。
- Loss 与 loss-to-pixel 仍是两个 pass。
- Pixel backward kernel 由用户选择，Auto 不选择 VkSplat。
- 完整 Gaussian gradient buffer 仍为 fallback、optimizer-disabled 和 densification 路径保留。
- Densification 仍可能产生较大的分配和带宽峰值。
- 训练只接受未畸变 PINHOLE/SIMPLE_PINHOLE 相机，不处理 target alpha mask。
- GPU能力按设备检查；跨设备性能和最终训练质量需要分别评估。
