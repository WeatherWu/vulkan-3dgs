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

### 运行时验证层控制

性能测试使用`vulkan-3dgs --validation off`，正确性检查使用`vulkan-3dgs --validation on`。Debug和Release都默认`on`，也接受`--validation=off`／`--validation=on`。验证层参数值缺失、非法或重复时，在创建窗口前报错。既有`--gpu`选择独立保留。这是仅启动时生效的选项，不写入UI配置，也不改变训练数值检查、日志级别或shader编译flags。

选项从CLI经`Application`传至`Context`，在创建实例前生效。可用性检查后的启用状态仍传入调度报告。请求的层不存在时保留原有告警并禁用行为，`on`不代表强制可用性关口。启动日志显示请求值及启用状态，描述应用实例配置，不能内省外部Vulkan loader强制覆盖。独立GPU诊断工具保留自身的`--require-validation`关口。

### S4优化器诊断（实验）

`Capture ONE S4 Optimizer Step and Pause`会对已暂停、已初始化的模型实际执行一个符合条件的普通融合步，再导出pre状态和原kernel的post参考；它不是只读导出或提速实现。资源限制、失败状态、严格device/shader匹配、验证命令及native证据要求见[S4捕获与回放指南](optimizer-diagnostics.md)。

显式离线`--candidate-r1`流式Adam原型保留AoS和原数学公式；目标3090回放逐位通过但性能未准入，本地AMD多步逐位差异仍未定因，不能在训练中选择。指南记录目标native证据、严格合成/回放关口及独立有界profiling窗口。S4-1M布局迁移和S4-2训练接线仍未实施。

### S4-3调度诊断（实验）

显式启用的有界时序采样和独立的直接／GPU-count排序工具，用于审计prepare到main的间隔及容量相关的排序成本。原训练提交、fence和CPU数量／容量处理全部保留，尚未开启GPU-driven训练。启动环境变量、正确性／成本命令和数据解释见[调度诊断指南](scheduling-diagnostics.md)。

时序schema v2补充GPU prepare时长、应用实际验证层配置、C++构建／编译器／宏和日志级别。逐步训练数值检查与Vulkan验证层分开记录；未提供的实例配置记为null。本轮不改变验证／日志设置或调度。

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

Loss pass计算L1/DSSIM并保存`SsimBackwardState`，loss-to-pixel消费这些系数。默认路径遍历二维11×11邻域，使用零填充和固定归一化。

`SSIM Mode`是停止状态可修改的A/B选项，配置键为`ssim_mode`：

| 值 | 模式 | Loss | Loss to pixel | Scratch |
| ---: | --- | --- | --- | ---: |
| 0（默认） | 2D (Baseline) | 原二维 | 原二维 | 无 |
| 1 | Separable (Full, Experimental) | 横向矩 + 纵向Loss | 横向系数 + 纵向梯度 | 80 B/pixel |
| 2 | 2D Loss + Separable Backward (Experimental) | 原二维 | 横向系数 + 纵向梯度 | 48 B/pixel |

模式2复用原`SsimBackwardState`，共三次dispatch，不派发可分离前向矩或纵向Loss变体。保持SSIM常量、L1/DSSIM权重、零填充、归一化和validation，只改变可分离部分的累加顺序。`BackwardLossPass`选择pipeline，所有dispatch/barrier计入`Loss`与`Loss to pixel`。`TrainingBuffers`按需分配三个或五个float4平面，切换时安全替换资源，基线模式或resize时释放，分配前检查`maxStorageBufferRange`和shader索引限制。1559×1039下混合/全可分离约需74.1/123.6 MiB。

旧`separable_ssim=0/1`加载为模式0/1；有`ssim_mode`时不受行序影响，优先使用新键。保存仅写`ssim_mode`；未知持久化值回退基线，无效API/config模式会被拒绝。全可分离原型尚未显示端到端收益。混合模式仍待GPU正确性和性能验证：先0/1比较当前v9的pixel/projected梯度，再200/1000做基线→混合→基线计时。CPU回归不代表GPU提速；Checkpointed Warp Backward目前是显式选择的冻结benchmark原型。

### Checkpointed反向P0准入

`training_checkpoint_layout.hpp`定义实验性CPU布局/容量合同；`tests/training_checkpoint_reference.hpp`和`tests/training_checkpoint_tests.cpp`实现标量forward、Tensor反向参考、已拒绝的前缀导数及反向边界状态生成/重放。布局1表示forward前缀，布局2表示每32项bucket远端的`(suffixColor, transmittance)`。

原FP32 `C - prefixColor`候选在非负HDR遮挡样例中未通过现有projected梯度容差：微弱贡献在最终RGB中被舍入丢失，重建残余色产生错误的opacity梯度。双精度一致不能代替FP32准入。反向边界CPU候选已通过现有FP32容差，已补齐`processedCount <= rangeCount`守卫；实际模式7目前仅由P2 benchmark命令选择。

```powershell
cmake --build build --config Debug --target training_checkpoint_tests
build/test/Debug/training_checkpoint_tests.exe --reverse-boundary-gate
```

反向边界准入有独立CTest项；旧`--candidate-gate`继续对不安全前缀候选输出`EXPECTED REJECTION`并退出1。CPU准入不代表GPU正确性或性能。

### Checkpoint P1状态诊断

P1增加可选GPU slot/task prefix和反向状态构建，训练梯度仍由原Tensor dispatch计算。资源属于`TrainingBuffers`，`CheckpointStatePass`属于`PixelTo2DGSDispatcher`。Binding 43/44保存元数据/prefix scratch，45–48保存四个64像素state bank，49保存task，50保存96 B控制/诊断块。容量根据已有prepare回读的真实tile-item数量配置，按高水位复用，检查descriptor/索引范围及驱动可用显存预算；资源替换具备事务性。普通训练会释放诊断workspace。

选择Fixed16 / Workgroup Shared / VkSplat Tensor / Subgroup / Global / RGB Baseline后点击`Run P1 State Diagnostic`。原`Start Fixed Benchmark`按钮提供同一冻结模型/帧的基线，UI显示实际执行/跳过的measured步数。其他组合被拒绝；资源失败会记录诊断跳过并继续原Tensor，跳过的运行不能作为P1成本证据。不增加逐step数量回读或queue等待。

Profiler显示`Checkpoint layout`、`Checkpoint state build`、`Checkpoint tasks`，三者之和为新增GPU成本H，与普通benchmark的原Tensor backward时间B比较。`Checkpoint capacity`记录CPU准备/分配时间。最后测量步只经已有提交/等待拷贝96 B诊断，日志给出slot/task数量、错误标志及预期/实际写入的pixel-state数量；非零标志或数量不符会判定失败。V8导出记录是否生成checkpoint状态；P1实际梯度模式仍为Tensor（6），同时保留运行日志。Pipeline准备与workspace分配另行记录CPU耗时，避免warmup重置隐藏启动成本。

无窗口GPU检查与CPU CTest分开执行：

```powershell
cmake --build build --config Debug --target training_checkpoint_gpu_tests
build/test/Debug/training_checkpoint_gpu_tests.exe
```

该target检查forward输出不变、可达状态及未写槽位投毒、元数据/task、边缘像素、prefix跨block、HDR遮挡、重复ID及容量溢出抑制；安装了Vulkan validation时会启用。当前也检查P2梯度与独立CPU反向、受支持的GPU参考路径一致：满足原Tensor能力门槛时使用Tensor Subgroup/Global，否则使用Direct；非native32设备显式启用所支持的subgroup-size control，以subgroup32运行同一P2 shader。状态生成现止于B(32)，最大processed前缀不超过32的tile直接跳过，因为没有bucket读取B(0)；独立CPU状态oracle仍执行完整反向遍历。

### Checkpoint P2 warp benchmark

`Run P2 Warp Benchmark`沿用同一冻结模型/帧和P1状态生成，用128线程/组的四个独立subgroup32任务替换Tensor梯度dispatch。每lane持有一个Gaussian和九个梯度累加器；像素suffix/T及loss梯度沿lane传递，以32像素协作预取作为流水线输入。只有subgroup登记使用一次workgroup barrier，内层使用shuffle且无共享导数矩阵。程序入口要求native subgroup32及P1基线配置。这是一次性benchmark命令，不是持久化训练设置，也不由Auto选择。

Control新增Tensor回退参数及跨step累计计数。构建错误将新dispatch设为0，原Tensor间接覆盖整图；计数在benchmark开始与第一个measured步重置，在中间step保留，仅借最后已有提交等待回读。消费者合同错误会判定冻结benchmark失败。含CPU跳过、GPU回退或错误标志的运行不能导出为纯checkpoint捕获。`Checkpoint warp backward`包含条件回退dispatch及梯度barrier，嵌套于`Pixel to 2DGS`，不能重复相加。

计时前先做0/1普通/P2/普通梯度捕获。完成的纯P2导出记录实际pixel mode 7、bucket32、state layout2、任务数量、状态已生成及零回退；Tensor专用标志为false，文件名含`_checkpoint_warp32_layout2`。需使用更新后的本地脚本比较：

```powershell
python -B scripts/compare_projected_gradients.py baseline.bin p2.bin --allow-pixel-mode-change
```

数值通过后在同一冻结负载用200/1000，比较`Checkpoint layout + Checkpoint state build + Checkpoint tasks + Pixel to 2DGS`与普通Tensor的`Pixel to 2DGS`。历史P1负载新增1.37 ms、Tensor基线2.99 ms，因此当时预算为P2反向低于1.62 ms才能不变慢，约低于1.24 ms才达到计划中Composite/Backward组合阶段10%目标。后续目标截图梯度通过（relative L2 `3.71882e-7`、零超容差/非有限分量），但H=1.18 ms、反向3.02 ms，合计4.20 ms，高于Tensor 2.94 ms；Total step 8.83→10.16 ms。直接reverse-prepass路线已停止，保留关闭的实验与测试，不推广至普通训练，不再进行本路线的整轮验收。

### 像素级EarlyCull实验

`Run S3 EarlyCull Benchmark`是一次性的冻结反向实验。选择Fixed16 / Workgroup Shared / Tensor / Subgroup / Global / RGB Baseline，两组保持相同SSIM；不能与checkpoint诊断组合。Auto、forward采样、batch16、pair矩阵、归约顺序与默认值不变。每batch预计算16个FP32阈值，共享内存增加64 B（按声明合计33604 B），不增加全图workspace、预pass、descriptor或queue等待。

方法1仅裁剪有限opacity `[1/255,1]`、原power `[-80,0]`的输入，条件为`power < log(alphaMin/opacity)-2^-10`；临界带及不支持输入走旧采样。保留pair仍用原power/exp、alpha及梯度表达式。按非RelaxedPrecision FP32的[Vulkan指令误差界](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#precision-and-operation-of-spirv-instructions)，此范围division相对误差低于`3e-7`、log/减法绝对误差低于`3e-6`，exp加alpha乘法相对误差低于`2.1e-5`。合并采用`2^-13`的保守log域预算，并覆盖exp2 lowering；精确`2^-10` guard为该预算的八倍。优化域中间量均为normal。这不是可调epsilon或降精度许可，仍须目标编译器/GPU边界测试和冻结梯度检查。

```powershell
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
build/test/Debug/training_checkpoint_gpu_tests.exe --pixel-culling-only
python -B scripts/compare_projected_gradients.py baseline.bin early_cull.bin
```

显式GPU工具检查阈值附近和fallback输入的合成分类，报告eligible/旧exp/剔除pair及完全剔除native subgroup数，再比较forward/BlendState和全部12个梯度槽位。native subgroup32且满足原资源门槛时用Tensor，否则用同一采样helper的测试专用Direct变体；Direct检查不是Tensor runtime或性能验收。合成probe要求compute subgroup VOTE，应用变体没有增加该能力要求。合成计数不代表目标场景节省。

先做普通/S3的0/1捕获；数值通过才做不带统计的200/1000普通/S3/普通计时。阈值工作已包含在`Pixel to 2DGS`，不能重复相加；比较组合阶段及Total step，不只看exp数量。v9记录实际方法0/1，候选文件名含`_pixel_cull_v1`。用户报告的目标冻结pair已通过：relative L2 `5.7464e-8`、最大绝对差`4.76837e-7`、零超容差/非有限/缺失分量，forward/pixel差为0。按两张计时截图依次解释为Off/On，Pixel-to-2DGS为3.14/3.12 ms、Total step为9.05/8.88 ms；未修改的sort/composite也有变化，不能宣称已有稳定全训练提速。Scanline跨度与前向裁剪尚未实现。

### EarlyCull全训练试验

`Tensor EarlyCull (Training, Experimental)`仅停止时可编辑，配置键`tensor_early_cull 0/1`，**默认关闭**。旧文件没有该键时使用默认值，显式非法/未知值关闭。它用于普通/Pure训练，包括optimizer和densification步，仅实际Fixed16/Workgroup Shared/Tensor/Subgroup/Global/Baseline及EarlyCull pipeline可用时启用；否则保留既有选定路径，不修改其他设置，`Training EarlyCull`显示原因。本次接线不修改数学、Adam/ADC调度、精度或shader。

不可变run config保存请求，domain在原录制前解析每步CPU策略，不增加buffer、pass或readback。`EarlyCull run steps`显示成功普通step总数、实际启用步和请求开启但未启用的回退步。计数在每次Start/Resume Training时重置，不随frame/validation重置，benchmark步不计入。开始和结束/暂停日志记录选择与计数；Pure模式最终发布详细计数，分段训练只报告最近一次Start/Resume区间。

Benchmark命令独立：`Start Fixed Benchmark`及P1/P2无视普通训练复选框，始终关闭裁剪；`Run S3 EarlyCull Benchmark`显式请求方法1，仍拒绝不兼容配置。v9及实际路径标签不变。从benchmark切回普通训练时恢复持久化的普通训练选择，不沿用上一个按钮的覆盖值。

用户报告的完整30000步试验现已显示回退，除有独立理由复现外保持此开关关闭；S3-2不要求重跑全训练。解读过去或另有明确理由的未来A/B时，保持初始化、种子、数据拆分/分辨率、SSIM、Adam/ADC、Tensor设置一致，输出到不同路径。暂停后继续同一已训练模型不是匹配的重新初始化A/B。受支持配置下一次不中断的开启组应显示30000 active、0 fallback，但计数不证明提速或held-out质量。该功能不推广，也不写入稳定skill。

### S3-2 scanline可行性诊断

这是离线工作量统计与成本筛查，不是scanline反向内核。用户报告的30000步EarlyCull试验中，Pixel-to-2DGS关闭/开启为2.77/2.91 ms，Total step为9.92/10.03 ms；保持该实验关闭，不再为本次诊断重复全训练。

1. 在冻结模型上运行**普通**`Start Fixed Benchmark`，warmup 0 / measured 1，选择Fixed16 / Workgroup Shared / Tensor / Subgroup / Global / RGB Baseline。完成后点击`Export S3-2 Diagnostic Inputs`；拒绝P1/P2及S3 EarlyCull捕获。普通训练复选框不会改变该benchmark路径。
2. 显式导出写入`Output Folder/benchmark_scanline/*.scanline.bin`，下载projected Gaussian、**排序后**item/range、逐pixel BlendState和pixel gradient；约120万Gaussian的帧约120 MB。只在停止状态增加compute到下载的同步，不同transfer queue使用semaphore桥接，不增加热路径pass/readback，不修改model、Adam、梯度或普通benchmark profiling，不覆盖旧捕获。
3. 尽量停止其他GPU负载，再用独立GPU工具读取捕获。Linux去掉可执行文件的`.exe`。

   ```powershell
   build/test/Release/training_checkpoint_gpu_tests.exe --scanline-capture path/to/frame.scanline.bin
   ```

   工具优先匹配捕获的vendor/device ID，写入同目录`.json`报告，重名使用数字后缀，不将梯度写回训练。先测frame0；首份报告确有结构性机会时才补少量其他视角。不需要200/1000或30000步。

独立输入格式`V3DSCN01` v1采用88 B little-endian header（magic加20个uint32），之后依次为48 B projected Gaussian、uint32排序item、16 B tile range、16 B BlendState和16 B pixel gradient。Header字段为version、Gaussian count、iteration、frame、width、height、item count、tile count、composite mode、pixel mode、subgroup、Global、RGB schedule、SSIM、pixel-cull method、来源vendor/device/driver及两个保留0。只接受原Tensor Subgroup/Global/Baseline方法0；这不是model/optimizer checkpoint，也不是v9梯度捕获。读取器在GPU执行前检查精确长度、元数据、维度、range和逐tile计数上界，整帧用uint64汇总。

按以下顺序读报告：

- `eligible_pairs`排除边缘无效像素、非法ID、`itemOffset >= processedCount`以及已经截断的最大processed前缀尾部；`already_guarded_pairs`和`already_trimmed_suffix_items`是原有节省，不能再算作scanline收益。
- `oracle_empty_*`使用原GPU power/exp/alpha判定，而非拟议bounds，统计含eligible pair但无alpha贡献的整行、Gaussian **tile-item**和反向batch。Batch16严格从各tile最大processed前缀的远端分组，不从offset0对齐。Oracle判断alpha贡献，不把loss gradient为0算作空间裁剪。
- `span_*`统计带保护的SPD行区间实际能识别多少机会。不支持的opacity/conic、病态条件、非有限计算和近切线保留整行；根区间向外取整并至少扩张1 pixel，每个拟剔除点都与原判定核对。**出现误剔除即拒绝该候选**；观察到0错误不等于生产FP32/梯度准入。
- 行/Gaussian/batch的pair加权机会互相包含，不能相加。逻辑空行不自动等于native subgroup一致跳过，必须另有集成设计才能移除导数矩阵工作或barrier。
- `bounds_proxy_not_net_overhead_*`单独测batch16 bounds-only dispatch，包含自身staging/barrier/checksum，不含使用mask的成本；不使用昂贵oracle统计的时间。支持native32的设备另做原Tensor前/后计时，各2次warmup、8次sample；缺失时间为null。无Tensor设备可验证统计，但不能证明目标Tensor划算；设备ID不匹配或前后基线漂移>15%时不下成本结论。

`integration_resource_screen`另标记驻留风险：已报告的33540 B Tensor若直接增加1024 B span表及256 B系数，三个组需要104460 B，超过[Ampere 8.6](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#occupancy)的100 KiB原始共享内存容量模型。这是算术推断，不是Vulkan实际驻留测量，也不能照搬CUDA专属的预留/carveout规则。未来原型必须核算native寄存器/共享分配，不能假设独立bounds-only核仍保持原Tensor驻留。

`verdict`只是筛查，不预测加速比：候选/输入错误、无eligible工作、没有识别到工作或识别很弱（整行及旧exp pair比例都<10%，且整batch比例<5%）时停止。Bounds proxy高于匹配Tensor基线25%也作为工程筛查停止该候选，不是“不可能加速”的数学证明。其他结果仍需人工复核，再决定是否做小型集成原型。`estimated_speedup`刻意为null，`production_kernel_implemented`为false；不缩guard或叠加互相包含的比例制造收益。

```powershell
build/test/Debug/training_scanline_tests.exe
build/test/Debug/training_checkpoint_gpu_tests.exe --scanline-only
```

CPU检查覆盖捕获I/O、截断、非法元数据/range、已有guard和远端batch对齐。合成GPU检查把所有oracle计数与独立标量参考逐项比较，并要求零误剔除；覆盖边缘tile、HDR、重复/非法ID、alpha-min相等、细长/非正定conic、长列表和内部oracle空batch。这些测试不能替代目标真实场景报告。

### S3-3紧凑行掩码准入

R0是独立离线统计。目标RTX3090的frame0 R0报告以13.16%的几何eligible pair覆盖、零误剔除及GPU/driver匹配通过；这不同于S3-2包含prefix空集合的45.89%。R1独立Tensor行掩码内核与R2离线比较器已实现，但目标Tensor16梯度、native资源及性能仍待验收。

```powershell
build/test/Release/training_checkpoint_gpu_tests.exe --row-empty-r0-capture path/to/frame.scanline.bin
```

Linux去掉`.exe`。复用二进制捕获，而非旧JSON；工具写同目录唯一后缀的`*.row_empty_r0.json`，不修改训练梯度，也不把统计时间作为候选内核时间。报告区分几何空行、已有guard空行、被保留的oracle空行和实际native wave机会。Native wave统计使用compute BASIC/VOTE，不代表Tensor中不同的Gaussian归约映射。

切线下界保留不确定输入，沿用已有有界opacity/power的exp/log合同，详见[数值域与误差账本](training-row-empty-contract.md)。仅RowEmpty census/候选使用Slang的precise GLSL兼容后端，普通shader编译选项不变。R1新增64 B行mask与64 B阈值缓存，每batch重建，保留零pair写入及原barrier，在采样和归约两端跳过空行。声明共享量为33668 B，对照33540 B，不是native驻留实测。FP32审计拒绝宽松浮点精度，唯一有界整数例外为subgroup lane builtin及其直接uint32读取；不安装新依赖，不静默回退编译后端或设备路径。

先运行合成验证，再在来源设备评估原二进制捕获（Linux去掉`.exe`）：

```powershell
build/test/Release/training_checkpoint_gpu_tests.exe --scanline-only
build/test/Release/training_checkpoint_gpu_tests.exe --row-empty-capture path/to/frame.scanline.bin
```

第二条命令写唯一后缀的`*.row_empty_r2.json`，要求native32和原Tensor能力gate、来源GPU/driver匹配、重新R0统计零误剔除且覆盖>=10%，再将完整12个梯度槽位与原Tensor及基线repeat比较。Pixel gradient/BlendState在计时外回读一次检查只读，forward不重算；错误立即中止计时。正确性通过后才做短测2 warmup/8 measured A/B/A。候选时间包含分类、mask使用和末尾梯度barrier；分配、清零、census及下载在两组区间外。编译后端不同，因此测量净候选而非只隔离mask收益，也不根据R0比例推算时间。

若短测有收益，且Nsight的native寄存器/共享分配/驻留复核未发现不可接受的回退，关闭profiler后运行：

```powershell
build/test/Release/training_checkpoint_gpu_tests.exe --row-empty-capture path/to/frame.scanline.bin --timing --resources-reviewed
```

资源flag是操作者确认，不是自动实测。再次短测仍有收益才执行长200/1000 A/B/A；基线漂移>2%为不确定，净收益<5%或不大于两次A差值的两倍则未通过工程门槛。通过仍需资源/多视角复核，不自动开启普通训练。运行器绝不以Direct或portable batch4替代目标Tensor16。小共享内存设备的合成检查显式使用batch4/Shared wrapper，验证同一mask/矩阵/barrier代码，对照不裁剪batch4与独立标量反向；不冒称Tensor16性能或native subgroup验收。稳定skill、训练默认值与v9导出格式不变；短测结果不授权30000步训练或应用benchmark接线。

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
| 42 | 可选的SSIM scratch，混合三个或全可分离五个float4平面 |
| 43/44 | 可选P1 checkpoint tile元数据与uint2 prefix scratch |
| 45–48 | 可选P1 checkpoint state bank，每bank覆盖64个pixel rank |
| 49/50 | 可选checkpoint bucket task与96 B控制/回退/跨step计数块 |

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

`Export Benchmark Gradients`写入配置输出目录下的`benchmark_gradients/`。Fixed16 Tensor文件名区分归约、像素梯度来源和Global RGB schedule；Adaptive Tensor文件名区分child batch。所有文件名增加`_ssim_2d`、`_ssim_separable`或`_ssim_backward_separable`，重名使用数字后缀。当前写v9：108字节little-endian header，包含magic `V3DGGRD1`、23个uint32和loss/alpha两个float32；在v7 SSIM模式后追加checkpoint bucket大小、state layout版本、任务数、state-built标志及回退次数，最后追加实际反向pixel-culling方法（0 Off、1 guarded FP32 EarlyCull）。方法1要求Tensor Subgroup/Global/Baseline且未构建checkpoint。未构建checkpoint的基线相关字段为0。v8保留104 B布局；v6/v7仍沿用84 B布局，v6 SSIM仅接受0/1，v7新增2表示二维Loss加可分离反向；模式1仍表示全可分离。Pixel mode7要求v8或更新、Fixed16/Workgroup Shared、已生成bucket32/layout2状态及无回退。Payload依次为每Gaussian 12个float32 projected梯度、每Gaussian一个membership字节、`width*height`个float4像素梯度（RGB有效、W填充）。比较器读v1–v9，报告实际路径，并检查repeat捕获的准确模式一致。除退出状态外还要看mismatch计数，不能只为通过而放宽容差。两个v6或更新文件还会报告pixel梯度误差，默认`--pixel-atol=1e-9`并沿用既有相对容差；旧文件明确报告无pixel数据。非有限forward loss/alpha会被拒绝。

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
- Loss与loss-to-pixel仍是两个阶段；全可分离/混合SSIM使用四次/三次dispatch并增加scratch流量，混合模式仍待目标GPU验收。
- Checkpointed Warp Backward是已停止扩展的性能实验；反向EarlyCull保留默认关闭的复现控制及通过的目标冻结pair，但报告的全训练性能回退，质量等价仍未验证。S3-2仅做离线诊断；scanline生产跨度、GPU SoA/AoSoA参数存储和GPU-driven count/sort仍未实现。
- Pixel backward kernel 由用户选择，Auto 不选择 VkSplat。
- RGB latency hiding尚无明确稳定收益；三个Tensor A/B开关仍默认关闭，也不改变Adaptive内核。
- 普通训练计划完成后，同步路径的training-complete检查可能提前终止新启动的多步Fixed Benchmark；独立生命周期修复前，应使用更早的冻结检查点并确认进度。
- 完整 Gaussian gradient buffer 仍为 fallback、optimizer-disabled 和 densification 路径保留。
- Densification 仍可能产生较大的分配和带宽峰值。
- 训练只接受未畸变 PINHOLE/SIMPLE_PINHOLE 相机，不处理 target alpha mask。
- 优化路径主要在 RTX 3090 上验证，尚无完整 NVIDIA/AMD/Intel 性能矩阵。
