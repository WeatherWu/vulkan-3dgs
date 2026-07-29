# 开发指南

[English](DEVELOPMENT.md) | [简体中文](DEVELOPMENT.zh-CN.md)

本文档面向项目贡献者和维护者，集中记录实现细节。主 README 保持面向项目使用者的简洁内容。

## 构建目标

- `vulkan_3dgs_core`：核心静态库。
- `vulkan_3dgs_app`：来自 `apps/main.cpp` 的应用入口。
- `image_cache_tests`：图片解码和缓存测试。
- `training_ssim_tests`：DSSIM backward 系数测试。
- `training_validation_tests`：两级 validation reduction 测试。

Windows 应用输出为 `vulkan-3dgs.exe`，非 Windows 平台输出为 `vulkan-3dgs`。

## 目录结构

```text
apps/                                  应用入口
src/application.*                     窗口、UI 和应用流程
src/context/                           Vulkan instance、device 和 queue 初始化
src/vulkan/                            Vulkan 资源封装和显存图片缓存
src/image/                             可复用图片解码、磁盘缓存和主存流式加载
src/gaussian_renderer/                 普通 PLY 渲染路径
src/gaussian_training/                 compute 训练路径
shaders/gaussian_compute_shader/slang/ 图形渲染和排序 shader
shaders/training_shader/slang/common/  训练共享 shader 代码
shaders/training_shader/slang/passes/  训练 compute passes
shaders/training_shader/slang/validation/ 紧凑 validation finalize pass
shaders/training_shader/slang/uint_radius/ radius 原子操作 fallback
tests/                                 CPU 和缓存测试
```

## 主要架构

`Application` 负责 GLFW 窗口、Vulkan context、UI 状态、文件对话框和主循环。

项目包含两条独立的 Gaussian 路径：

- `GaussianRenderer` 使用 graphics shader、GPU key generation 和 `vulkan_radix_sort` 加载并渲染 PLY 模型。
- `GaussianTraining` 负责数据集选择、训练 buffer、compute renderer、优化、稠密化、validation、图片缓存和 PLY 导出。

训练路径不是 `GaussianRenderer` 的封装，它拥有独立的 GPU 资源和 compute pipeline。

应用维护两个逻辑设备角色：

- presentation device 要求 graphics、当前 GLFW Surface 的 present、swapchain、push descriptor、anisotropy 和 shader draw parameters，负责普通渲染与 ImGui；
- training device 不查询 Surface，只要求 compute queue、push descriptor 和 `VK_EXT_shader_atomic_float` 的 buffer float32 atomic add，transfer 可使用专用队列或回退到 compute queue。

`Training` 面板的 GPU 选择和 `--gpu` 只作用于 training device。切换训练设备会清理已有训练 GPU 资源并要求重新加载数据集，不重建 GLFW、ImGui 或 swapchain。训练 `Buffer` 必须显式使用 training device 的 transfer/compute queue-family 索引，不能从全局 presentation `Context` 获取队列族。

全局 Vulkan-Hpp dispatcher 只由 presentation device 初始化。Training device 不重新初始化 dispatcher，否则其未启用的 swapchain 函数可能覆盖显示路径的 device-level 函数指针。

## Gaussian PLY 数据

普通渲染器支持 binary little-endian 3DGS 风格 PLY 属性：

- 位置：`x`、`y`、`z`
- SH：`f_dc_0..2`、`f_rest_0..44`
- 不透明度：`opacity`
- 尺度：`scale_0..2`
- 旋转：`rot_0..3`

加载时会恢复 scale 和 opacity 的参数化值，并根据 scale 和 quaternion rotation 生成 covariance。

## 数据集加载

训练 loader 支持 MipNeRF360/COLMAP 风格结构：

- `sparse/0/cameras.bin` 和 `sparse/0/images.bin`，并支持其他兼容位置；
- `images_4`、`images_2`、`images_8` 或 `images` 等图片目录；
- 可选的 `points3D.bin` 稀疏点。

程序从图片元数据读取真实尺寸，并将 COLMAP 内参缩放到所选图片尺寸。所有训练帧必须使用一致的尺寸。

稀疏点用于初始化位置和颜色。初始 opacity、scale、rotation 和 optimizer state 由训练实现生成。没有稀疏点时，可以使用随机 fallback 初始化场景。

## 数据集图片流

图片子系统与训练逻辑分离，后续其他图片处理模块也可以复用。

- `ImageDiskCache` 负责 KTX2 chunk 持久化、验证、恢复和配额控制。
- `ImageStreamer` 负责 source 注册、主存 `RGBA8` entry、请求等待、worker 线程预取和 LRU 淘汰。
- `GaussianTraining` 选择数据集帧并持有 streamer 实例。
- `DeviceImageCache` 负责训练专用的 device-local 图片槽和上传同步。

图片以线性 `RGBA8` 保存，每像素 4 字节，target 值在 GPU shader 内解包。

主存缓存默认预算：

```text
budget = min(可用 RAM * 10%, 可用 RAM - 2 GiB)
```

显式设置的非零 host budget 会覆盖自动预算。完整解码数据集能装入预算时会预取所有图片，否则使用按需加载、有限预取和 LRU 淘汰。活动中的 `ImageHandle` 会暂时阻止对应图片被淘汰。

磁盘缓存默认配置：

- 每个 KTX2 array chunk 包含 16 层；
- 每次读取一个完整 chunk，并将其中全部层一起发布到主存缓存；
- 20 GiB 配额；
- 统计区分当前数据集 active chunk 和历史 chunk，超配额时优先淘汰历史 chunk；
- 使用平台对应的用户缓存目录。

显存图片缓存行为：

- 每个驻留 `RGBA8` 图片占用一个对齐的 storage-buffer 槽；
- 支持 `Streaming`、`Partial`、`Full` 三种模式；
- 至少保留一个槽；
- 最大分配量取对齐后的完整数据集槽位大小和 device-local heap budget 四分之一中的较小值；
- 普通预留为 `max(512 MiB, heap budget 的 15%)`，并限制为不超过 heap budget 的一半；
- 稠密化前预留为 `max(1 GiB, heap budget 的 25%)`，同样限制为不超过一半；
- 支持 timeline semaphore 时使用三槽持久映射 staging ring；
- `VULKAN_3DGS_SYNC_IMAGE_UPLOAD=1` 可强制同步上传。

UI 会显示主存、active/history 磁盘缓存、显存 resident/slot/total、staging、上传、命中/未命中、淘汰和 memory budget 数据。

## 训练流程

`GaussianTraining::trainStep()` 执行：

1. 选择并请求 target frame，上传相机数据。
2. 准备 projected Gaussian 和 tile count。
3. 提交并等待 tile-count pass。
4. 必要时扩展 tile-item buffer。
5. 生成并排序 tile item，重建 range，执行 pixel composite。
6. 计算 L1/DSSIM loss 和 backward gradient。
7. 可选执行 densification/pruning。
8. 执行 Adam optimizer。
9. 提交并等待 main pass。
10. 读取紧凑 validation 和 densification 统计。
11. 接管稠密化后的 buffer，并推进训练调度。

调度模式：

- `Sequential`：按顺序使用图片，没有固定总迭代停止值。
- `3DGS Random`：随机不放回 viewpoint stack，总迭代数为 30000。该模式现在是默认值，用于对齐 reference 3DGS 训练循环。

## Loss 和 Optimizer

Loss 组合 L1 和 DSSIM：

```text
(1 - lossDssimWeight) * L1 + lossDssimWeight * (1 - SSIM)
```

SSIM 使用 sigma 1.5 的 11x11 Gaussian window。`SsimBackwardState` 保存优化后的 backward 系数，避免原先 window 内再次遍历 window 的嵌套计算。

Pixel-to-2DGS 投影梯度只对九个可微分量执行原子累加：center XY、conic/opacity XYZW 和 RGB。深度、整数屏幕半径及颜色 padding 保持为零，不执行原子加。

训练面板提供 `Auto`、`Direct`、`Workgroup Shared` 和 `Subgroup` 四种 Pixel-to-2DGS backward 路径。只有 Vulkan 报告 compute stage 同时支持 BASIC 与 SHUFFLE subgroup operation 时，`Auto` 才选择 `Subgroup`，否则回退到 `Direct`。Subgroup shader 使用运行时 wave 宽度，并且只同步同一 subgroup 内的像素。两种协作版本都保持每个像素的逆序遍历和梯度数学不变；`Workgroup Shared` 保留用于显式 profiling。

Tile count 和 emit 共用同一个保守的 ellipse-pixel-rectangle 判定。只有当 valid pixel center 矩形内的最小 conic quadratic 能证明所有位置的 opacity 都低于 `1/255` 贡献阈值时，才会删除该 tile。原有 projected 3-sigma radius 保持不变，继续用于 densification 和 screen-size 逻辑。

Forward compositing 提供 `Direct` 和 `Workgroup Shared` 两种模式。Shared 模式为默认值，以 256 个 Gaussian 为一批协作加载每个 tile 已排序的 projected Gaussian，同时保持每个像素的 front-to-back 顺序、alpha 阈值、transmittance 终止条件和 `processedCount` 不变。

训练使用独立的 prepare/main command buffer 和逐提交 fence。CPU 仍需等待 prepare 的 tile count，以便在 emit、sort、composite 和 optimizer 执行前处理 buffer overflow，但不再调用可能包含同一 queue 上无关工作的 queue-wide `waitIdle()`。vendor radix sorter 虽支持 indirect count，但它按最大 capacity 录制 dispatch，不能为训练图其余 pass 提供 overflow 条件执行。

Densification 将参数、Adam 和 validation partial buffer 作为 ping-pong storage 保留，不再在每次 adopt 后重建两侧。Per-Gaussian scratch buffer 按 16384 个 Gaussian 分块增长并跨 densification 事件复用。64 字节 densification counters 使用持久映射 readback，densification state 通过一次 GPU fill 清零，opacity reset 则根据真实输出数 indirect dispatch，不再按 workspace capacity 调度。

原先逐贡献写入的 `GaussianVisibilityState` 原子操作和 buffer 已移除，因为没有训练 pass 消费其中的 contribution count、alpha sum 或 radius。Densification 继续使用 `GaussianDensificationState`。

Adam-style optimizer 为 position、SH DC/rest、opacity、scale 和 rotation 提供独立学习率。Position learning rate 支持初始值、最终值、delay multiplier、最大步数调度，并乘以 reference 3DGS 使用的 scene extent spatial learning-rate scale。默认 opacity LR、Adam epsilon 和关闭 gradient clipping 的行为与 reference 3DGS 对齐。

## 训练 Buffer

`TrainingBuffers` 负责：

- Gaussian 参数、gradient、Adam state、projection 和 projection gradient；
- tile item、拆分 key、排序 index、scratch storage 和 tile range；
- rendered/target color、pixel gradient、blend state 和 SSIM state；
- visibility 和 densification 统计；
- densification candidate/output buffer 和 counter；
- 紧凑 validation partial/final buffer；
- preview instance 和 camera uniform。

Descriptor getter 返回的是 `vk::DescriptorBufferInfo` 值。将其地址传给 descriptor write 前，必须先保存到生命周期稳定的局部变量中。

## 训练 Validation

Validation 在前 5 次迭代以及之后的 `validationInterval_` 执行：

```cpp
nextIteration <= 5 ||
nextIteration % validationInterval_ == 0
```

周期验证避免下载完整 GPU buffer：

1. `train_loss.comp.slang` 在 binding 29 为每个 256-thread workgroup 写入 loss、渲染检查、反向候选数和实际贡献数 pixel partial。
2. `train_optimizer.comp.slang` 在 binding 30 写入非有限 Gaussian partial。
3. `train_validation_finalize.comp.slang` 在 binding 31 写入固定 80 字节结果。
4. `GaussianTraining` 将结果复制到私有的三槽 host-coherent staging ring，并轮询 fence。

`TrainingValidationStats` 和前端输出保持不变。完整 loss/rendered 下载 API 仍然保留，但不再用于常规 validation。Gaussian 参数下载仍用于 PLY 导出。

## Profiling

CPU 阶段包括 frame upload、image request、target upload、prepare submit、tile-count readback、tile resize、main command record、main submit、validation、densification adoption 和 total step。UI 同时显示每个阶段的实际 sample count。

Tile item count 只在 `GaussianTraining` 中读取一次。prepare command 将 16 字节 counter 复制到 `TrainingBuffers` 持有的持久映射 host-coherent buffer；prepare queue wait 完成后，CPU 直接读取映射内存，并将同一个 count 显式传给 emit/sort/render 流程。

GPU timestamp 阶段包括 tile preparation/emission/sorting、composite、loss、各 backward 阶段、optimizer 和 densification。

Timestamp 回读使用 availability result，不再通过 `VK_QUERY_RESULT_WAIT_BIT` 阻塞。

## 稠密化

默认功能包括 clone、split、opacity pruning、screen/world-size pruning、opacity reset、局部 split offset、scale shrinking 和 optimizer state 处理。

Float atomic 行为：

- 支持时启用 `VK_EXT_shader_atomic_float` buffer atomic add；
- `VK_EXT_shader_atomic_float2` min/max 是可选项；
- 不支持 float min/max 时，`uint_radius` 变体将非负 radius 保存为 uint bits；
- 这些 compute 特性不是全局 physical-device 选择的硬条件。

当前性能限制：

- candidate/output buffer 按最坏增长容量分配；
- 多个大型 buffer 通过临时 staging 从 CPU 全量清零；
- 完整 Gaussian 和 Adam 数据会依次复制到 candidate 和 final buffer；
- 全局 append counter 存在 atomic contention；
- 部分 pass 按最大容量而不是实际输出数量 dispatch；
- adoption 会重新创建多个活动 scratch buffer；
- densification counter 仍通过通用临时 staging 下载；
- 稠密化前调整图片显存缓存可能增加分配和淘汰停顿。

计划中的优化方向是持久化几何增长 workspace、GPU clear、active/output ping-pong、持久化 counter readback、indirect dispatch，以及最终的 classify + prefix-sum + direct scatter。

## Vulkan Loader 和设备说明

vcpkg manifest 包含：

- Vulkan headers、loader、utility libraries 和 validation layers 1.4.350；
- Vulkan loader 的 XCB、Xlib 和 Wayland features；
- GLFW、GLM、ImGui、STB、KTX、Slang 和 COLMAP。

Linux 上，instance/device 创建和 entry point dispatch 必须使用同一套 Vulkan loader。VNC 或 VirtualGL 环境继承的 `LD_PRELOAD` 可能注入冲突的 loader 或 dispatch layer。

常用检查命令：

```bash
ldd ./vulkan-3dgs | grep -E "vulkan|glfw"
env -u LD_PRELOAD ./vulkan-3dgs
vulkaninfo
```

Validation layer 通过运行时 layer manifest 发现。安装 package 并不保证在 `VK_LAYER_PATH` 或 loader 环境错误时仍能找到验证层。

默认跳过 CPU Vulkan device。设置 `VULKAN_3DGS_ALLOW_CPU_VULKAN=1` 可以允许软件 fallback。

## Shader 构建

顶层 CMake 显式编译 Slang shader。新增 pass 时必须注册到 `compile_training_shader(...)` 或 `compile_slang_shader(...)`。

生成的 SPIR-V 会复制到：

```text
build/bin/<Config>/shaders/
```

C++/Slang 结构布局和 descriptor binding 必须保持同步。

## 与 Reference 3DGS 的已知差异

- 随机 viewpoint stack 和 30000 次迭代停止现在通过默认 `3DGS Random` 模式启用；`Sequential` 保留为调试/旧行为模式。
- Forward/backward 数学仍是项目实现，暂时不能描述为与 reference 完全一致。
- Tile sort 使用两次稳定的 32-bit radix pass 表达语义上的 64-bit key。
- 训练循环每次迭代仍会等待 prepare 和 main submission。
- Gaussian 数量较高时，稠密化存在明显的分配和带宽开销。
- 训练结果通过 PLY 导出进入普通渲染器，而不是实时共享训练 render path。

## 验证命令

```powershell
cmake --build build --config Debug
cmake --build build --config Release
ctest --test-dir build -C Debug --output-on-failure
ctest --test-dir build -C Release --output-on-failure
git diff --check
```
