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

训练路径由 `GaussianTraining` 编排，由三个 compute renderer 实现具体 GPU 工作：

- `GaussianForwardRenderer`：投影、tile 构建、排序和前向合成；
- `GaussianBackwardRenderer`：loss、像素梯度、2D Gaussian 梯度、3D Gaussian 梯度和 Adam；
- `GaussianDensificationRenderer`：clone、split、prune、opacity reset 以及稠密化后的 buffer 统计。

它不是普通 `GaussianRenderer` 的 graphics 渲染循环。普通渲染器从 PLY 读取最终参数并绘制实例化四边形；训练路径在 compute shader 中显式构建 tile 列表，并且需要保存前向合成所用的中间量，供反向传播再次访问。

下图把两个 CPU 边界放在顶部的同一个外框内：左侧是 `CPU input`，右侧是 `CPU output`。中间是一次 U 形训练迭代，左列前向传播向下执行，底部计算 Loss/SSIM，右列反向传播向上返回 CPU output；Forward 和 Backward 两个分区名称分别放在对应 pass 列的最底部。pass 框内部只保留算法阶段名称；buffer 的读取、写入、需要保留到反向阶段的前向状态以及 CPU/GPU 传输都直接写在蓝色虚线数据箭头上，不再放在 pass 内部的数据行或独立数据框中。

![Gaussian training logic and data flow](assets/training-flow.svg)

图由 Graphviz 根据本地且被忽略的 `docs/training-flow.dot` 生成。修改 DOT 后可执行 `dot -Tsvg docs/training-flow.dot -o docs/assets/training-flow.svg` 更新 GitHub 直接显示的 SVG。

粗黑实线是连续的训练逻辑通路：顶部左侧 CPU input 的箭头指入单次迭代，前向 pass 从上到下执行，在底部计算唯一一次 `Loss + SSIM`，随后直接进入 `Loss-to-pixel`，再从下到上依次执行 Pixel-to-2DGS、2DGS-to-3DGS 和 optimizer/densification，最后向上指入顶部右侧的 CPU output。蓝色虚线是数据通路：箭头方向从数据产生者或保留状态指向消费者，箭头文字列出该依赖实际携带的 buffer 或持久状态。同列虚线放在相邻 pass 框之间，表示局部数据交接；横向虚线表示反向传播复用的前向中间量。

在 CPU 边界，传入虚线携带 selected frame index、packed target RGBA8、camera uniform 和 iteration constants；传出虚线携带紧凑的 validation/profiling 统计、densification counters 和 output Gaussian count，并不表示把完整 Gaussian 参数下载到 CPU。迭代内部，前向投影写出 `ProjectedGaussian`：一条虚线把它传给 tile 构建，另一条横向虚线把保留的投影状态传给 2DGS-to-3DGS。Graphviz 使用显式 rank 固定左右两列，前向链负责垂直层级，反向逻辑边不参与 rank 计算，因此右列节点保持 `B4` 在上、`B1` 在下，而真实箭头仍从 `B1` 连续向上指向 `B4`。

### 1. 数据集、相机和 target

`TrainingDatasetLoader` 读取 COLMAP 相机、位姿和可选 `points3D.bin`。它使用所选图片文件的真实宽高，并按真实宽高缩放 `fx/fy/cx/cy`，因此 shader 中的 `params.width/height`、camera uniform 和 target buffer 必须来自同一分辨率。`downscale` 只影响 loader 选择的图片目录和内参缩放，不会在训练 shader 中再次除一次分辨率。

每次迭代的 frame 选择由 `TrainingScheduleConfig` 控制：

- `Sequential`：`(current + 1) % dataset.size()`，没有固定总步数；
- `Random`，UI 名称为 `3DGS Random`：使用随机种子初始化 viewpoint stack，每轮随机取出一个索引且不重复，stack 为空时重新填充。默认总步数为 30000，与 reference 3DGS 的训练调度一致。

`ImageStreamer` 在 CPU 侧保存线性 `RGBA8` 图片。它可以从 KTX2 磁盘 chunk 恢复，或由 source 图片解码；`DeviceImageCache` 再把当前 target 上传到 device-local storage buffer 的一个对齐 slot。shader 通过 binding 7 读取 packed `uint` pixel，并在 GPU 内解包为 `[0, 1]` 的颜色。target 不是每步通过 CPU 逐像素转成 float，也不会把整个数据集长期放在 GPU 上。

相机写入 `TrainingForwardCamera`，布局为 15 个 `vec4`，即 240 字节：`view`、`projection`、`model` 各 4 个 `vec4`，以及 `viewport`、`focalTan`、`cameraPosition`。相机通过 binding 11 作为 uniform buffer 提供给投影和反向投影 shader。

### 2. 参数化模型和 GPU 数据结构

训练参数使用与 reference 3DGS 相同的“存储未激活参数、shader 内激活”的思路。C++ `training_types.hpp` 中的结构与 Slang 结构按 16 字节对齐，`static_assert` 检查总大小。

| 结构 | 布局和大小 | 用途 |
| --- | --- | --- |
| `GaussianTrainParam` | `positionOpacity: vec4`、`scale: vec4`、`rotation: vec4`、`sh[16]: vec4`，共 19 个 `vec4`，304 B | 当前 Gaussian 的可训练参数 |
| `GaussianGrad` | 与 `GaussianTrainParam` 相同，304 B | 由反向传播累积的梯度 |
| `AdamState` | `firstMoment + secondMoment` 两个 `GaussianGrad`，608 B | Adam 的一阶、二阶矩 |
| `ProjectedGaussian` | `centerRadius`、`conicOpacity`、`color`，48 B | 当前相机下的 2D 投影 |
| `ProjectedGaussianGrad` | 与 `ProjectedGaussian` 相同，48 B | 像素反向累积的 2D 梯度 |
| `PixelGrad` | 一个 `vec4`，16 B | 每像素 RGB loss 梯度，第四分量为空 |
| `PixelBlendState` | `finalTransmittance`、`accumulatedAlpha`、`processedCount`、`contributionCount`，16 B | 保存 forward 结果给 loss 和 backward |
| `GaussianDensificationState` | screen gradient sum/count、max screen radius、padding，16 B | 稠密化判据统计 |
| `SsimBackwardState` | `base`、`targetCoefficient`、`renderedCoefficient` 三个 `vec4`，48 B | 将 SSIM 的局部导数预计算为线性系数 |

对第 `i` 个 Gaussian，参数化关系是：

```text
opacity_i = sigmoid(positionOpacity_i.w)
scale_i   = exp(scale_i.xyz)
q_i       = normalize(rotation_i)
R_i       = quaternionToMatrix(q_i)
Sigma3D_i = R_i * diag(scale_i^2) * R_i^T
```

因此 buffer 中的 `scale` 不是世界空间半径，`rotation` 也不保证每次写入后已经归一化；投影时归一化 quaternion，优化时更新的是原始参数。opacity 使用 logit 参数化，避免优化器直接把透明度限制在区间内。

### 3. Prepare pass：从 3D Gaussian 到每 Gaussian tile 计数

prepare 使用独立的 `prepareCommandBuffer_`，顺序固定为 `projectGaussians()`、`clearTileRanges()`、`countTileCoverage()`、`prefixGaussianTileRanges()`，最后复制 16 字节 counter 到持久映射的 host-coherent readback buffer。

#### 3.1 前向 buffer 覆盖关系

prepare 不再执行统一的前向 clear。composite 会完整写入每个有效 pixel 的 rendered color 和 pixel blend state，loss 会完整写入每个 pixel 的 loss，Gaussian prefix 会覆盖 CPU 实际读取的 counter 字段，而 Gaussian gradient 会在反向原子累加前统一清零。删除这些在首次读取前必然被覆盖的写入不会改变训练数学。

#### 3.2 投影

`train_forward_project.comp.slang` 每个 invocation 处理一个 Gaussian，并把结果写入 `ProjectedGaussian`。先做 model/view 变换：

```text
p_model  = model * [p, 1]
p_camera = view  * p_model
p_clip   = projection * p_camera
ndc      = p_clip.xyz / p_clip.w
```

`camera.z <= 0.2` 的 Gaussian 被标记为不可见。相机空间协方差由 Jacobian 线性化投影：

```text
Sigma_camera = V * Sigma_world * V^T
J = [ fx/z,   0, -fx*x/z^2 ]
    [  0,  fy/z, -fy*y/z^2 ]
    [  0,    0,       0    ]
Sigma_pixel = (J * Sigma_camera * J^T)[0:2, 0:2] + 0.3 I
```

投影前会把 `x/z`、`y/z` 限制在 `1.3 * focalTan` 范围内。这样可以避免极端视锥边缘导致 Jacobian 爆炸，同时在反向时记录 `gradMask`，被 clamp 的方向不再错误地传播位置梯度。

2D conic 是 `Sigma_pixel` 的逆矩阵。令
`Sigma_pixel = [[a0, b0], [b0, c0]]`，`det = a0*c0-b0*b0`，则写入：

```text
conic = [ c0/det, -b0/det, a0/det ]
```

`ProjectedGaussian.conicOpacity` 的四个 float 是 `A, B, C, opacity`。屏幕 radius 是最大特征值的 3-sigma ceil，主要用于初始 tile bounds 和 densification 的 screen-size 统计。

颜色由当前 Gaussian 到相机的 view direction 评估 SH：

```text
color = max(evaluateSH(sh, viewDirection) + 0.5, 0)
```

shader 使用的 `activeSHDegree` 会逐步开放高阶系数；这影响 forward 颜色和 backward 写入的 SH 梯度数量。

#### 3.3 高斯椭圆与矩形相交的数学原理

问题可以抽象为：给定一个由正定二次型定义的椭圆，以及一个轴对齐矩形，判断二者是否相交。

#### 3.3.1 二次型与有效支撑椭圆

设椭圆中心为 $\boldsymbol\mu$，形状矩阵为对称正定矩阵 $\mathbf M$，并令 $\boldsymbol\delta=\mathbf p-\boldsymbol\mu$：

$$
\mathbf M
=
\begin{bmatrix}
A&B\\
B&C
\end{bmatrix},
\qquad
\boldsymbol\delta=
\begin{bmatrix}
\Delta x\\
\Delta y
\end{bmatrix}
=\mathbf p-\boldsymbol\mu.
$$

对应的二次型为：

$$
q(\boldsymbol\delta)
=\boldsymbol\delta^T\mathbf M\boldsymbol\delta
=A\Delta x^2+2B\Delta x\Delta y+C\Delta y^2.
$$

设高斯函数的幅值为 $\rho>0$，其贡献函数为

$$
g(\boldsymbol\delta)=\exp\!\left(-\frac{1}{2}q(\boldsymbol\delta)\right),
\qquad
f(\boldsymbol\delta)=\rho\,g(\boldsymbol\delta).
$$

给定阈值 $\tau>0$，有效贡献条件 $f\ge\tau$ 等价于

$$
\rho\exp\!\left(-\frac{q}{2}\right)\ge\tau
\quad\Longleftrightarrow\quad
q\le s,
\qquad
s=2\ln\!\left(\frac{\rho}{\tau}\right).
$$

当 $\rho<\tau$ 时有 $s<0$，而正定二次型满足 $q\ge0$，因此不存在满足阈值的点。当 $\rho=\tau$ 时，$s=0$，有效区域退化为中心点。当 $\rho>\tau$ 时，$s>0$，有效支撑区域为非退化椭圆

$$
\mathcal E=\{\boldsymbol\delta\mid q(\boldsymbol\delta)\le s\}.
$$

形状矩阵的正定条件为

$$
A>0,
\qquad C>0,
\qquad \det(\mathbf M)=AC-B^2>0.
$$

这保证 $q$ 是严格凸函数，并且在 $s>0$ 时 $q=s$ 是有限椭圆。

#### 3.3.2 矩形区域上的约束最小值

设矩形在相对坐标中为

$$
\mathcal R=[x_{\min},x_{\max}]\times[y_{\min},y_{\max}].
$$

定义矩形上的二次型最小值

$$
q_{\min}
=\min_{\substack{
x_{\min}\le x\le x_{\max}\\
y_{\min}\le y\le y_{\max}
}}
\left(Ax^2+2Bxy+Cy^2\right).
$$

因为 $\mathbf M$ 正定，$q$ 是严格凸函数，矩形区域上的最小值只能出现在以下位置：

1. 无约束极小点是 $\boldsymbol\delta=(0,0)^T$。如果它不在矩形内，取

   $$
   \boldsymbol\delta_0
   =\Pi_{\mathcal R}(\mathbf 0)
   $$

   其中 $\Pi_{\mathcal R}$ 表示到矩形 $\mathcal R$ 的逐坐标投影，因此 $\boldsymbol\delta_0$ 是矩形中最接近原点的点。

2. 对固定 $x$ 的左右竖直边：

   $$
   \frac{\partial q}{\partial y}=2Bx+2Cy=0
   \quad\Longrightarrow\quad
   y^*(x)=-\frac{B}{C}x.
   $$

   分别代入 $x=x_{\min}$、$x=x_{\max}$，并把 $y^*$ 投影到 $[y_{\min},y_{\max}]$。

3. 对固定 $y$ 的上下水平边：

   $$
   \frac{\partial q}{\partial x}=2Ax+2By=0
   \quad\Longrightarrow\quad
   x^*(y)=-\frac{B}{A}y.
   $$

   分别代入 $y=y_{\min}$、$y=y_{\max}$，并把 $x^*$ 投影到 $[x_{\min},x_{\max}]$。

因此候选集合可以写成：

$$
\begin{aligned}
\mathcal P=\{&\boldsymbol\delta_0,\\
&(x_{\min},\Pi_{[y_{\min},y_{\max}]}(-Bx_{\min}/C)),\\
&(x_{\max},\Pi_{[y_{\min},y_{\max}]}(-Bx_{\max}/C)),\\
&(\Pi_{[x_{\min},x_{\max}]}(-By_{\min}/A),y_{\min}),\\
&(\Pi_{[x_{\min},x_{\max}]}(-By_{\max}/A),y_{\max})\}.
\end{aligned}
$$

计算

$$
q_{\min}=\min_{\mathbf p\in\mathcal P}q(\mathbf p).
$$

如果最小值位于角点，至少一条相邻边上的投影候选会落在该角点，因此不需要额外枚举角点。最终判定为

$$
q_{\min}\le s.
$$

若 $q_{\min}>s$，矩形与椭圆严格分离；若 $q_{\min}\le s$，二者相交。

#### 3.3.3 图像示例

下面的图把 $q(\boldsymbol\delta)=\text{C}$ 看作从高斯中心向外扩张的一族同心椭圆。对给定矩形，候选集合 $\mathcal P$ 包含原点到矩形的投影，以及四条边上的一维驻点。选择其中二次型最小的点 $\mathbf p^*$，就等价于找到从中心扩张时第一个接触矩形的等值椭圆。

![高斯阈值椭圆与矩形的相交判定](assets/gaussian-ellipse-rectangle-example.svg)

左图中，第一次接触矩形的绿色虚线位于蓝色阈值椭圆 $q=s$ 内部，因此 $q_{\min}\le s$，矩形与有效支撑区域相交。右图中，第一次接触矩形的红色虚线位于阈值椭圆外部，因此 $q_{\min}>s$，二者严格分离。图中的空心点是边界候选，实心点是最终选出的 $\mathbf p^*$。

该图也说明了为什么不能只使用矩形中欧氏距离中心最近的点：二次型的等值线通常会旋转并沿两个主轴具有不同尺度，真正需要最小化的是 $q$，而不是普通的圆形距离。

#### 3.3.4 有限搜索域与阈值椭圆

高斯函数在整个平面上具有无限支撑，但阈值 $\tau$ 将有效区域限制为有限椭圆 $\mathcal E$。如果在此基础上再引入一个有限搜索矩形 $\mathcal B$，实际判断的区域是

$$
\mathcal E\cap\mathcal B.
$$

矩形约束最小值只负责判断 $\mathcal E$ 是否与局部矩形相交；有限搜索域则决定哪些局部矩形会被检查。前者是精确的凸优化判定，后者属于对无限支撑函数的截断假设。

#### Tile pair 构建说明

`train_forward_tile_count.comp.slang` 不再对 per-tile count 做 atomic。每个 invocation 处理一个 Gaussian，并把精确保留的 tile 数写入 `gaussianTileRanges[i].y`，`.x` 暂时为零。随后 `train_forward_gaussian_prefix_*` 以 256 个元素为一组做分层 exclusive scan：第一层写每个 Gaussian 的组内 offset 和每组 block sum；当 block sum 数仍大于 256 时，继续在 scratch 中递归生成更高层；顶层由一个 workgroup 完成 scan，并把总 pair 数写入 binding 9；最后 offset-add pass 从上到下把父层 offset 加回子层，再写入 `gaussianTileRanges[i].x`。scratch 容量按所有层元素总和分配，因此跨越 256 和 65536 Gaussian 边界时仍使用同一套路径。

main emit 时，每个 Gaussian invocation 独占 `[range.x, range.x + range.y)`，把自己的 pair 连续写入，不需要 atomic cursor。count 和 emit 共用相同的当前候选遍历方式与精确 ellipse helper，所以 prefixed count 与实际写入数一致。两次稳定 radix sort 后，`train_forward_tile_range_boundaries.comp.slang` 比较相邻 sorted high key，写出每个 tile 的 start/end；`train_forward_tile_sort.comp.slang` 再转换成 `[start,count]` 并把空 tile 保持为零。边界 pass 额外处理末尾 sentinel，因此 tile item 数为 0 或 1 时也有明确行为。

### 训练 shader pass 对照表

下表按 `GaussianForwardRenderer`、`GaussianBackwardRenderer` 和 `GaussianDensificationRenderer` 当前的录制顺序列出主要 pass。除特别说明外，`numthreads(256,1,1)` 的 pass 按 Gaussian、pixel 或 partial 数量 dispatch；forward/composite 和 pixel-to-2DGS 使用 tile 级 `numthreads(16,16,1)`。

| pass 文件 | 阶段 | 输入 | 输出/作用 |
| --- | --- | --- | --- |
| `train_forward_project.comp.slang` | prepare | parameters、camera | 写 `ProjectedGaussian`，同时更新 screen radius 统计 |
| `train_forward_tile_clear.comp.slang` | prepare | tile ranges | 清零后续 range 重建使用的 per-tile start/end |
| `train_forward_tile_count.comp.slang` | prepare | projected、Gaussian ranges | 遍历粗包围矩形、执行精确 tile test、写 per-Gaussian count |
| `train_forward_gaussian_prefix_ranges.comp.slang` | prepare | Gaussian counts、scratch | 每 256 Gaussian 做 exclusive scan，并写第一层 block sum |
| `train_forward_gaussian_prefix_scratch.comp.slang` | prepare | prefix scratch | 递归扫描一层 block sum，并生成父层 |
| `train_forward_gaussian_prefix_top.comp.slang` | prepare | 顶层 prefix、counter | 扫描顶层并写 tile-item 总数 |
| `train_forward_gaussian_prefix_add_scratch.comp.slang` | prepare | 父/子 prefix 层 | 把父层全局 offset 传播到子层 |
| `train_forward_gaussian_prefix_add_ranges.comp.slang` | prepare | Gaussian ranges、第一层 prefix | 把全局 block offset 加到 Gaussian start |
| `train_forward_tile_emit.comp.slang` | main | projected、Gaussian ranges | 连续写 Gaussian index、depth/tile key 和 sort index |
| `train_forward_tile_gather_high.comp.slang` | main | high key、sort index | 为低 key 排序结果 gather high key |
| `train_forward_tile_gather_items.comp.slang` | main | unsorted items、sort index | 生成最终 sorted item 数组 |
| `train_forward_tile_range_boundaries.comp.slang` | main | sorted high key/index | 在相邻 tile key 边界写 start/end |
| `train_forward_tile_sort.comp.slang` | main | tile start/end | 转换成 `[start,count]`，并清零空 tile |
| `train_forward.comp.slang` | forward | projected、sorted items、ranges | Direct 逐 pixel front-to-back composite |
| `train_forward_workgroup.comp.slang` | forward | 同上 | Shared 以 256 Gaussian 批次协作加载后 composite |
| `train_loss.comp.slang` | loss | rendered、target、blend | 写 per-pixel loss、SSIM backward state、pixel validation partial |
| `train_backward_clear.comp.slang` | backward fallback | gradient buffers | Gaussian 和 projected gradient 的 compute 清零回退路径 |
| `train_backward_loss_to_pixel.comp.slang` | backward | rendered、target、SSIM state | 写每 pixel RGB loss gradient |
| `train_backward_pixel_to_2dgs*.comp.slang` | backward | pixel grad、tile lists、blend | 逆序重放 composite，atomic 累加 2D gradient |
| `train_backward_2dgs_to_3dgs.comp.slang` | backward | parameters、projected gradient | 链式求 position/opacity/scale/rotation/SH gradient |
| `train_densify_clear.comp.slang` | densify | counter | 清零 output/统计 counter |
| `train_densify_prune.comp.slang` 或 `train_densify_prune_uint_radius.comp.slang` | densify | parameters、Adam、state | clone/split/keep/prune，并 append output |
| `train_densify_dispatch.comp.slang` | densify | output counter | 生成真实 output count 的 indirect dispatch |
| `train_opacity_reset.comp.slang` | densify | densified params/Adam/counter | 按真实 output count reset opacity |
| `train_optimizer.comp.slang` | main tail | parameters、gradient、Adam | 只执行 Adam 更新 |
| `validation/train_gaussian_validation.comp.slang` | main tail | parameters、可选 densification counter | 分类非有限 Gaussian 字段，每 256 个 Gaussian 写一个 partial |
| `validation/train_validation_finalize.comp.slang` | main tail | pixel/Gaussian partial | 合并为 80-byte final validation result |

`train_backward_pixel_to_2dgs.comp.slang`、`_workgroup`、`_subgroup` 和 `_adaptive` 是同一数学过程的不同内存协作实现；`train_densify_prune_uint_radius.comp.slang` 是没有 float min/max atomic 时的 radius 表示 fallback，不是另一套稠密化规则。CMake 还会编译 `train_project.comp.slang`、`train_pack_render_buffer.comp.slang` 和 `train_densify_finalize_prune.comp.slang`，它们属于旧的通用/预览或兼容 pipeline；当前 `GaussianTraining::trainStep()` 由专用 renderer 录制的 pass 以上表为准。

## Forward、Backward、Loss 和 Optimizer

### 4. Prepare 提交、CPU readback 和容量处理

CPU 结束 prepare command 后提交到 training `computeQueue_`，用 `prepareFence_` 等待该提交完成。等待是必要的，因为下一步需要安全地读取 tile item 总数，并决定是否重建 tile buffer；它不是对所有 Vulkan queue 做 `waitIdle()`。

`TrainingBuffers::recordTileItemCountReadback()` 录制 GPU `copyBuffer`，把 counter buffer 中的 16 字节复制到自身持有的持久映射 readback。fence 完成后 `requiredTileItemCount()` 直接读取这块映射内存，CPU 只获得一个很小的统计结构，不下载 tile ranges、projected Gaussian 或图片。

若 `requiredTileItems > tileItemCapacity()`，CPU 以 `max(required, oldCapacity * 2)` 扩容，并同步准备 unsorted items、两个 key、sort indices、sorted items 以及 radix sort scratch/storage。扩容发生在 main command 录制之前，因此 main 不会因为 emit 越界。capacity 没有超过需求时，本轮不分配。

### 5. Main forward：emit、排序、range 和合成

main 使用独立的 `mainCommandBuffer_`。CPU 把 prepare 得到的 `requiredTileItems` 写入 push constants，然后依次录制：

1. `train_forward_tile_emit.comp.slang`：再次遍历 projected Gaussian 和候选 tile。每个 Gaussian 使用 `gaussianTileRanges` 中自己的 prefixed `[start,count]` 连续写入 `tileItems`、`tileKeyLow`、`tileKeyHigh` 和 `tileSortIndices`，不使用 per-tile atomic cursor；每个 `tileItems` 元素都是一个完整的 `uint` Gaussian index；
2. 对低 32 位 depth key 执行稳定 radix sort；根据排序后的 index gather high key；
3. 对 high tile key 执行第二次稳定 radix sort；再 gather 成最终 `tileItemsSorted`；
4. `train_forward_tile_range_boundaries.comp.slang` 在相邻 high key 变化处写 start/end，随后 `train_forward_tile_sort.comp.slang` 把它转换为每个 tile 的 `[start,count]` range。

逻辑 key 是 `(tileIndex, depth)` 的 64 位 key，但当前 vendored sorter 是 32-bit key/value 接口，所以用“先稳定排序低位、再稳定排序高位”保持相同的字典序语义。依赖点是 radix sort 必须稳定；否则同一 tile 内的 depth 顺序可能被破坏。

`train_forward.comp.slang` 或 `train_forward_workgroup.comp.slang` 以 16x16 threads 处理一个 tile。Direct 版本直接从 sorted tile item 读取；Workgroup Shared 版本按最多 256 个 Gaussian 一批，把 index、center/radius、conic/opacity、color 放入 shared memory，再由 256 个 pixel lane 各自计算。两者都保持：

```text
color       += T * alpha * gaussianColor
alphaSum    += T * alpha
T           *= 1 - alpha
renderAlpha  = 1 - T
```

每个 pixel 对整 tile 保守测试保留下来的 sorted Gaussian index 执行原始 sample test。其中 `alpha = clamp(opacity * exp(power), 0, 0.99)`，且 `power <= 0`、`alpha >= 1/255` 才是贡献。`T * (1-alpha) < 1e-4` 时停止继续处理。每个 pixel 的 `PixelBlendState` 保存最终 `T`、累计 alpha、已处理的 sorted prefix 长度 `processedCount` 和实际贡献数 `contributionCount`，这是反向阶段重建同一前向路径的依据。

### 6. Loss 和 backward

#### 6.1 L1、SSIM 和 pixel gradient

`train_loss.comp.slang` 为每个 pixel 计算：

```text
L1_pixel = (|R-T| + |G-T| + |B-T|) / 3
L_pixel  = ((1-w) * L1_pixel + w * (1-SSIM_pixel)) / pixelCount
```

SSIM 使用半径 5、sigma 1.5 的 11x11 Gaussian window，并使用常数 `C1=0.0001`、`C2=0.0009`。loss 和 loss-to-pixel 都使用 16x16 workgroup；每组协作把 16x16 中心区域加 5-pixel halo 缓存为 26x26 shared-memory 区域，并使用预计算的一维 Gaussian 权重，避免每个 sample 重复全局读取和 `exp()`。越界 halo 按原实现写零，因此边界数学不变。每个中心 pixel 计算 rendered/target 均值、方差和 covariance：

```text
SSIM = ((2*mu_r*mu_t + C1) * (2*cov_rt + C2)) /
       ((mu_r^2 + mu_t^2 + C1) * (var_r + var_t + C2))
```

loss pass 同时写入 `SsimBackwardState`：
`dSSIM/dsample = base + targetCoefficient * target + renderedCoefficient * rendered`。
这样 `loss-to-pixel` 对一个 sample 的反向只需遍历周围 11x11 个中心状态，而不需要对每个窗口重新嵌套计算窗口统计。

`train_backward_loss_to_pixel.comp.slang` 读取 rendered/target 和上述 state，写入 `PixelGrad.color.xyz`。第四分量只作 padding，target buffer 中的 RGBA8 在 shader 内解包。

#### 6.2 pixel-to-2DGS：可微分的前向重放

每个 16x16 tile pixel 在 `processedCount - 1` 到 `0` 的反向顺序遍历 sorted prefix。设当前 sample 的 alpha 为 `a_i`、颜色为 `c_i`，先从最终 transmittance 反推出该 sample 之前的 transmittance：

```text
T_before = T_after / max(1 - a_i, 1e-6)
dL/dc_i = dL/dpixelColor * (T_before * a_i)
dL/da_i = dot(dL/dpixelColor, T_before * (c_i - suffixColor))
suffixColor = a_i*c_i + (1-a_i)*suffixColor
```

随后对 `power = -0.5*(A*dx^2 + 2B*dx*dy + C*dy^2)` 求导。当前 packed symmetric conic 的梯度约定是：

```text
dL/dA = dL/dpower * (-0.5 * dx * dx)
dL/dB = dL/dpower * (-0.5 * dx * dy)
dL/dC = dL/dpower * (-0.5 * dy * dy)
dL/dopacity = dL/dalpha * coverage
```

`B` 只在 `ProjectedGaussian` 中存一次；下游 `backwardConicToCovariance()` 将它作为对称矩阵的非对角项补全。因此不能把 `dL/dB` 写成 `-dx*dy`，否则 covariance 链会把该项重复放大。

每个 pixel 对同一个 Gaussian 的梯度通过 float32 atomic add 累加到 `ProjectedGaussianGrad`。原子操作只写九个可微分字段：center XY、conic/opacity XYZW、RGB；depth、整数 radius 和 color.w 不参与优化，也不应被误认为训练丢失了梯度。

实现提供 `Direct`、`Workgroup Shared`、`Subgroup` 三种固定内核和 `Auto (Adaptive)`。设备不支持 compute subgroup BASIC 与 SHUFFLE 时，Auto 和手动 Subgroup 都回退 Direct；支持时，Auto 先求当前 subgroup 的真实最大 `processedCount`。若它不大于原生 subgroup 宽度，各有效 lane 直接运行自己的逆序循环；只有更长的候选 prefix 才进入 subgroup broadcast/reduction，从而让浅 tile 避免 wave 归约开销，长 tile 继续共享 Gaussian 读取和原子累加。subgroup 内同一轮所有 lane 都处理同一个 broadcast Gaussian，各 lane 保持自己的 pixel transmittance/suffix 状态。它先归约整数 `contributionCount`；总数为零时跳过九个浮点梯度分量的归约，否则再用 wave shuffle 归约梯度，并由 lane 0 对该 Gaussian 执行一次原子累加。该归约改变浮点求和顺序，但不改变每 pixel 的逆序、阈值和梯度公式。

反向累加前，`GaussianBackwardRenderer` 默认对当前有效 `GaussianGrad` 和 `ProjectedGaussianGrad` 前缀各录制一次 `vkCmdFillBuffer`，随后用 buffer barrier 将 transfer write 对 compute shader read/write 可见。`train_backward_clear.comp.slang` 仍作为 compute fallback 保留；设置 `VULKAN_3DGS_COMPUTE_BACKWARD_CLEAR=1` 可强制使用它。GPU profiling 的 `Backward clear` timestamp 会包围实际启用的路径。

#### 6.3 2DGS-to-3DGS 链式反向

`train_backward_2dgs_to_3dgs.comp.slang` 每个 Gaussian 读取 `ProjectedGaussianGrad`，按以下链路写入 `GaussianGrad`：

```text
center/depth -> clip/ndc -> camera/object position
conic       -> inverse(Sigma_pixel) -> Sigma_pixel
Sigma_pixel -> J * V * Sigma_world * V^T * J^T
Sigma_world -> model * R * diag(exp(scale)^2) * R^T * model^T
opacity     -> sigmoid(rawOpacity)
color       -> SH(viewDirection) + 0.5
```

协方差逆矩阵使用解析导数，`scale` 梯度再乘 `d exp(rawScale)/d rawScale = exp(rawScale)`；quaternion 梯度包含 rotation matrix 导数和 normalize 的投影导数。SH 只对 `activeSHDegree` 以内的系数写入梯度，并把 view direction 的导数继续传回 Gaussian position。该 pass 同时把 screen-space center gradient 的 magnitude 累加到 `GaussianDensificationState`。

### 7. Densification、pruning 和 Adam

`shouldRunDensification()` 要求 enabled、迭代处于 `[densifyFromIteration, densifyUntilIteration)`，并满足 `densificationInterval`。默认参数为 from=500、until=15000、interval=100、opacity reset=3000、max Gaussian=10000000、split children=2、grad threshold=0.0002、min opacity=0.005、screen threshold=20、world threshold=0.1。

稠密化在本轮 backward 之后、optimizer 之前执行：

1. GPU clear binding 19 的 64-byte counter；
2. `train_densify_prune.comp.slang` 为每个 active Gaussian 判断 opacity、平均 screen gradient、screen radius 和 world scale；
3. 根据 clone/split/keep/prune 结果 atomic append 到 `densifiedParams` 和 `densifiedAdamStates`，同时写出 kept/clone/split/prune 统计；split child 使用 Gaussian 局部尺度和 rotation 变换后的随机 normal offset，并缩小 child scale；
4. `train_densify_dispatch.comp.slang` 根据实际 output count 生成 indirect dispatch；
5. 若到了 opacity reset 周期，`train_opacity_reset.comp.slang` 通过 `dispatchIndirect` 只处理真实输出数量，并同步清理或调整对应 Adam state；
6. 复制 64 字节统计到持久映射 readback，main fence 完成后 CPU 读取；
7. `adoptDensifiedGaussians()` 交换 active/output 的参数、Adam 和 partial buffer，更新 `trainableGaussianCount_`，而不是把完整参数下载到 CPU。

稠密化 workspace 以当前 capacity 的 1.5 倍几何增长，再向上对齐到 16384 个 Gaussian 的块并复用，减少 Gaussian 数量连续增长时反复重建 gradient、projection、densification state 和 preview buffer。`GaussianDensificationState` 在一次稠密化结束时用 `vkCmdFillBuffer` 清零，供下一轮 backward 继续累积。支持 `VK_EXT_shader_atomic_float` 时使用 buffer float32 atomic add；float min/max 不可用时，`uint_radius` shader 变体保存非负 radius 的 uint 表示。上述扩展是训练能力的可选实现，不参与全局 presentation GPU 的硬性筛选。

Adam pass 对每个 Gaussian 使用 bias correction：

```text
m_t = beta1*m_(t-1) + (1-beta1)*g_t
v_t = beta2*v_(t-1) + (1-beta2)*g_t^2
mhat = m_t / (1-beta1^t)
vhat = v_t / (1-beta2^t)
param -= lr * mhat / (sqrt(vhat) + epsilon)
```

position、SH DC、SH rest、opacity、scale、rotation 有独立学习率。CPU 每轮预计算 position 的 init/final log-linear schedule、可选 delay multiplier、scene-extent scale，以及 Adam 的 `(1-beta)` 和两个 bias-correction 倒数，再通过 push constants 传入；shader 不再为每个 Gaussian 重复执行 `pow()` 和 position schedule。默认 epsilon 为 `1e-15`，gradient clipping 为 0 表示关闭。当前迭代执行 densification 时 `optimizerEnabled` 会关闭，避免在 active 参数已经被输出 buffer 替换的同一提交中再次更新旧数组；非最终迭代才执行 optimizer。

## 训练 Buffer

`TrainingBuffers` 是训练路径的资源所有者，按逻辑分成以下几组：

| 组 | buffer | 生命周期和读写方向 |
| --- | --- | --- |
| 参数 | `gaussianParams_`、`gaussianGrads_`、`adamStates_` | 初始化后常驻 GPU；参数由 project/optimizer 读写，梯度由 backward 原子累加 |
| 投影 | `projected_`、`projectedGrads_` | 每步覆盖；project 写投影，pixel-to-2DGS 写梯度 |
| tile | `gaussianTileRanges_`、`gaussianTilePrefixScratch_`、`tileItems_`、`tileKeyLow_`、`tileKeyHigh_`、`tileSortIndices_`、`tileSortScratch_`、`tileItemsSorted_`、`tileSortStorage_`、`tileRanges_` | prepare 生成 per-Gaussian count/start 和总数，main 连续写 item、排序并重建 per-tile 范围 |
| 像素 | `renderedColor_`、`targetColor_`、`pixelGrads_`、`pixelBlendStates_`、`loss_`、`ssimBackwardStates_` | 每步清理或覆盖；target 由图片缓存上传，其余由 forward/backward 写入 |
| 稠密化 | `densificationStates_`、`densifiedParams_`、`densifiedAdamStates_`、`densificationCounters_` | backward 累积判据，densify 生成 output，adopt 后交换 active/output |
| validation | `pixelValidationPartials_`、`gaussianValidationPartials_`、`densifiedGaussianValidationPartials_`、`validationFinalResult_` | 周期性 GPU reduction，不下载完整数组 |
| 辅助 | `counters_`、`previewInstances_`、`camera_` | Gaussian prefix 总数、预览渲染实例、camera uniform |

这些 buffer 的 capacity 与 `gaussianCapacity_`、`gaussianWorkspaceCapacity_`、`densificationCapacity_`、`tileItemCapacity_`、`extent_` 分开管理。Gaussian 参数是 AoS：一个 `GaussianTrainParam` 内连续存放 position/opacity、scale、rotation 和 16 个 SH `vec4`；tile item 则是按 tile 排序后的完整 `uint` Gaussian index，避免复制完整 Gaussian 到每个贡献项。

## Descriptor Binding

所有训练 pass 使用 set 0 的 storage/uniform buffer。binding 号码是 C++ `updateDescriptorSets()` 和 Slang `[[vk::binding(N, 0)]]` 之间的 ABI，修改一侧必须同步修改另一侧。

| binding | 主要对象 | 使用阶段 |
| ---: | --- | --- |
| 0 | `GaussianTrainParam` | project、2DGS-to-3DGS、densify、optimizer |
| 1 | `GaussianGrad` | backward clear、2DGS-to-3DGS、optimizer |
| 2 | `AdamState` | densify、optimizer |
| 3 | `ProjectedGaussian` | emit、composite、pixel-to-2DGS、2DGS-to-3DGS |
| 4 | sorted `tileItems` | composite、pixel-to-2DGS |
| 5 | per-tile `tileRanges` | clear、range boundary/build、composite、backward |
| 6 | rendered `float4` | clear、composite、loss、loss-to-pixel |
| 7 | target packed `uint` | loss、loss-to-pixel |
| 8 | per-pixel loss | clear、loss |
| 9 | general/tile counters | Gaussian prefix 总数、其他统计 |
| 11 | `TrainingForwardCamera` uniform | projection 和 2DGS-to-3DGS |
| 12 | `PixelGrad` | backward clear、loss-to-pixel、pixel-to-2DGS |
| 13 | `ProjectedGaussianGrad` | backward clear、pixel-to-2DGS、2DGS-to-3DGS |
| 14 | `PixelBlendState` | clear、composite、loss、pixel-to-2DGS |
| 16 | `GaussianDensificationState` | project、2DGS-to-3DGS、densify |
| 17/18 | densified params/Adam | densify output、opacity reset |
| 19 | densification counters | densify、opacity reset、optimizer validation |
| 20/21/22 | low/high sort key、sort indices | tile emit、radix gather、sort |
| 23/24 | sort scratch、sorted tile items | radix gather |
| 25/26/27 | candidate params/Adam/state | densification finalize |
| 28 | `SsimBackwardState` | loss、loss-to-pixel |
| 29/30/31 | pixel partial、Gaussian partial、80-byte final result | validation |
| 32/33 | per-Gaussian `{start,count}`、hierarchical prefix scratch | tile count/prefix/emit |

`TrainingBuffers` 的 descriptor getter 返回 `vk::DescriptorBufferInfo` 值。传给 `vk::WriteDescriptorSet::pBufferInfo` 前必须保存在生命周期稳定的局部变量中，不能取得临时返回值的地址。

## Command Buffer、队列和同步

训练设备可以有专用 compute queue，也可以使用同时支持 transfer 的 compute queue。训练 buffer 创建时显式使用 training device 的 queue-family index；presentation device、GLFW surface、swapchain 和 ImGui 不参与训练提交。

每个 step 的同步边界是：

```text
CPU record prepare
  -> submit prepare + prepareFence
  -> wait prepareFence
  -> CPU read tile count / optional resize
CPU record main
  -> submit main + mainFence 或 validation slot fence
  -> wait selected fence
  -> CPU read compact stats / adopt
```

同一个 command buffer 内不依赖隐式顺序来解决 storage buffer hazard。renderer 在 pass 之间插入 `vkCmdPipelineBarrier`：典型源访问是 `ShaderWrite`，目标访问按下一个 pass 是 `ShaderRead | ShaderWrite`、`TransferRead/Write` 或 `IndirectCommandRead`，目标 stage 至少包含 `ComputeShader`，间接 dispatch 还包含 `DrawIndirect` stage。`vkCmdFillBuffer` 后使用 `TransferWrite -> ComputeShader` barrier，保证清零数据可见。

图片上传是另一条异步链路。`DeviceImageCache` 在 timeline semaphore 可用时用三槽 persistently mapped staging ring 上传 target，并记录 `pendingImageUploadValue_`。main submit 通过 `vk::TimelineSemaphoreSubmitInfo` 等待该 value，等待 stage 为 `ComputeShader`；因此 shader 读取 binding 7 前 target upload 已完成。设置 `VULKAN_3DGS_SYNC_IMAGE_UPLOAD=1` 会选择同步 fallback，便于排查 upload race，但会牺牲吞吐。

prepare/main 使用逐提交 fence，不再调用 queue-wide `waitIdle()`。代价是训练循环仍然是同步迭代，尤其 prepare 必须等待一个小 readback；validation 使用三槽 fence ring，避免非验证迭代下载大 buffer。

## CPU/GPU 数据传递

CPU 到 GPU 的热路径只有：camera uniform、push constants、当前 target 的 upload，以及初始化/导出时的 Gaussian 参数传输。每步的 Gaussian 参数、gradient、Adam、projected、tile item、rendered color 和 pixel gradient 都留在 GPU。

GPU 到 CPU 的常规路径分为三类：

1. **容量控制**：prepare 只回读 16 字节 tile counter，决定 tile item buffer 是否扩容；
2. **统计和诊断**：稠密化只回读 64 字节 counters，validation 最终结果固定 80 字节；
3. **显式导出**：PLY 导出时才下载完整 `GaussianTrainParam` 数组。完整 loss/rendered download API 是工具接口，不属于常规训练循环。

validation 的 reduction 过程是：loss pass 每 256 pixels 写一个 `TrainingPixelValidationPartial`，独立 Gaussian validation pass 每 256 Gaussians 写一个 `TrainingGaussianValidationPartial`，finalize pass 用一个 workgroup 合并为 `TrainingValidationGpuResult`。它统计 loss、alpha、luminance、invalid loss/pixel、平均候选数、平均 contributor、最大 processed candidate 和各类 non-finite Gaussian。C++ 把 80 字节复制到私有三槽 host-coherent staging ring，通过 fence poll 取回，最终填充 `TrainingValidationStats`。

## 训练 Validation

Validation 在前 5 次迭代以及之后的 `validationInterval_` 执行。ImGui 的 `Validation Interval` 默认值为 `100`；设为 `0` 时只保留前 5 次检查：

```cpp
nextIteration <= 5u ||
(validationInterval_ > 0u && nextIteration % validationInterval_ == 0u)
```

`invalidLossCount` 表示 loss 非有限，`invalidRenderedPixelCount` 表示 rendered RGBA 非有限，`nonFiniteGaussianCount` 再按 position、raw opacity、raw scale、`exp(scale)`、rotation 和 SH 分类。`nonFiniteActivatedScaleCount` 很重要：raw scale 可能有限，但 `exp(rawScale)` 已经溢出，后续 covariance/conic 就会产生无穷大。

周期验证避免下载完整 GPU buffer：

1. `train_loss.comp.slang` 在 binding 29 为每个 256-thread workgroup 写入 loss、render 检查、反向候选数和实际贡献数的 `TrainingPixelValidationPartial`。
2. `train_gaussian_validation.comp.slang` 在 binding 30 为每个 256-Gaussian workgroup 写入非有限分类的 `TrainingGaussianValidationPartial`。稠密化迭代中，它读取 GPU output count 并检查 densified output buffer。
3. `train_validation_finalize.comp.slang` 在 binding 31 将 partial 合并为固定 80 字节 `TrainingValidationGpuResult`。
4. `GaussianTraining` 将结果复制到私有的三槽 host-coherent staging ring，并轮询 slot fence。

`TrainingValidationStats` 和前端输出保持不变。完整 loss/rendered 下载 API 仍然保留，但不再用于常规 validation。Gaussian 参数下载仍用于 PLY 导出。

## Profiling

CPU 阶段包括 frame upload、image request、target upload、prepare submit、tile-count readback、tile resize、main command record、main submit、validation、densification adoption 和 total step。UI 同时显示每个阶段的 last、average 和 sample count。

Tile item count 只在 `GaussianTraining` 中读取一次。prepare command 将 16 字节 counter 复制到 `TrainingBuffers` 持有的持久映射 host-coherent buffer；prepare fence 完成后，CPU 直接读取映射内存，并将同一个 count 显式传给 emit/sort/render 流程。

GPU timestamp 阶段包括 Gaussian projection、tile coverage count、Gaussian tile prefix、tile emission、sort/ranges、composite、loss、backward clear、loss-to-pixel、pixel-to-2DGS、2DGS-to-3DGS、optimizer、validation 和 densification。UI 标签仍显示 `Tile prefix`，实际测量的是分层 per-Gaussian prefix passes。`Optimizer` 现在只包含 Adam dispatch；`Validation` 包含 Gaussian 有限性归约和最终归约。像素 partial 的条件归约仍位于 `Loss` 内，因此 validation 迭代也会让 `Loss` 样本变慢。该拆分只在 validation 迭代增加一个独立 Gaussian-validation dispatch，不改变检查内容或 optimizer 数学。

Timestamp 回读使用 availability result，不再通过 `VK_QUERY_RESULT_WAIT_BIT` 阻塞。

例如 `pixel-to-2DGS` 高，优先看每 pixel 的 `processedCandidate` 和 `contributor`；tile item 高，优先看 Gaussian radius、分辨率、tile culling 和近景大 Gaussian；`Main submit/wait` 高，则要区分 GPU timestamp、image upload timeline wait、validation readback 和其他 queue 上的任务，不能只看单个 shader 时间。

## 稠密化

默认功能包括 clone、split、opacity pruning、screen/world-size pruning、opacity reset、局部 split offset、scale shrinking 和 optimizer state 处理。

Float atomic 行为：

- 支持时启用 `VK_EXT_shader_atomic_float` buffer atomic add；
- `VK_EXT_shader_atomic_float2` min/max 是可选项；
- 不支持 float min/max 时，`uint_radius` 变体将非负 radius 保存为 uint bits；
- 这些 compute 特性不是全局 physical-device 选择的硬条件。

当前实现的性能特征是：candidate/output payload 仍需复制完整的 Gaussian 和 Adam 结构，append counter 有 atomic contention，稠密化发生时仍可能触发大 buffer capacity 检查和 image-cache 预算调整；但 active/output 参数和 Adam 已经 ping-pong，workspace 按 16384 Gaussian 分块复用，state 用 GPU fill 清零，counter 使用持久映射 readback，opacity reset 使用真实 output count 的 indirect dispatch。

GPU-driven tile count/sort 暂不强行移除 prepare 的 CPU readback：当前 emit buffer 必须在主 command 录制前拥有足够容量，而 vendored radix sorter 的 indirect count 接口不能让后续整条图在容量不足时条件执行。若要彻底 GPU-driven，需要预留固定上限或增加 GPU overflow/resize 协议，前者浪费显存，后者仍需要 CPU 介入，不能仅删除一次 readback。

Gaussian 参数继续使用 AoS。投影、2DGS-to-3DGS、Adam 和 PLY 导出都会同时读取一个 Gaussian 的多个字段，直接 SoA 会同时改变 C++/Slang layout、descriptor ABI、densification copy、Adam state 和导出路径；在当前 profile 中没有足够收益证据支持这一高风险重构。后续若继续优化，应优先 profiling `Densify/prune`、`Densification adopt` 和 cache resize 的实际占比，再考虑 classify + prefix-sum + direct scatter，以减少全局 append 和 candidate payload。不能把已经移除的 `GaussianVisibilityState` 或“每步 CPU 全量清零/下载”写成当前实现。

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
