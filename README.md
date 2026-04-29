# Vulkan + 3D Gaussian Splatting 项目框架

这是一个基于 Vulkan 图形 API 和 3D 高斯散射（3DGS）技术的现代渲染框架，专为实时 3D 场景渲染和点云可视化设计。

##  项目结构

```
vk-gs/
├── CMakeLists.txt              # 根项目配置
├── README.md                   # 项目说明
├── sandbox/                    # 沙盒测试环境
│   ├── CMakeLists.txt
│   └── sandbox.cpp            # 快速测试入口
└── src/                       # 核心源代码
    ├── CMakeLists.txt         # 核心库配置
    │
    ├── vulkan/                # 通用 Vulkan 资源封装层
    │   ├── buffer.hpp/cpp     # 缓冲区管理（支持 Staging 上传）
    │   ├── shader.hpp/cpp     # 着色器模块加载
    │   ├── swapchain.hpp/cpp  # 交换链管理
    │   └── command_pool.hpp/cpp # 命令池与命令缓冲区管理
    │
    ├── context/               # Vulkan 基础环境
    │   ├── context.hpp/cpp    # Instance, Surface, Device 管理
    │   └── device.hpp/cpp     # 物理/逻辑设备、队列家族选择
    │
    ├── gaussian_render_compute/ # 高斯渲染特定实现
    │   ├── gaussian_model.hpp/cpp      # 高斯点云数据模型
    │   ├── gaussian_renderer.hpp/cpp   # 高斯渲染器
    │   ├── renderpass.hpp/cpp          # 渲染通道配置
    │   └── pipeline.hpp/cpp            # 图形管线状态
    │
    ├── utils/                 # 工具类
    │   ├── camera.hpp/cpp     # 相机控制（View/Projection 矩阵）
    │   ├── file_utils.hpp     # 文件 I/O 操作
    │   └── logger.hpp         # 分级日志系统
    │
    ├── application.hpp/cpp    # 应用框架（主循环、事件处理）
    ├── window.hpp/cpp         # GLFW 窗口封装
    └── renderer.hpp/cpp       # 渲染器基类接口
```

##  架构设计

### 分层架构
1. **Utils 层**: 基础工具（日志、文件、数学、相机）
2. **Vulkan 层**: 封装 Vulkan API 底层对象（Buffer, Shader, Swapchain, CommandPool）
3. **Context 层**: Vulkan 实例/设备/队列管理
4. **GS 渲染层**: 业务逻辑层，处理高斯模型数据与渲染逻辑
5. **Application 层**: 应用框架层，整合窗口、输入、渲染循环

### 核心设计原则
- **职责分离**: Context 仅负责基础环境，渲染资源由 Renderer 管理
- **模块化隔离**: 通用资源（vulkan/）与特定实现（gaussian_render_grapics/）物理隔离
- **RAII 管理**: 所有 Vulkan 资源通过 C++ 对象生命周期自动管理
- **可扩展性**: 通过替换 Renderer 子类实现不同渲染方法切换

##  核心模块

### 1. 应用程序框架 (Application)
- 基于 GLFW 的窗口管理与事件处理
- Vulkan 上下文初始化与管理
- 模板方法模式：子类重写 `initialize()` 和 `render()`
- 支持运行时渲染器动态切换

### 2. Vulkan 基础环境 (Context)
- **Context**: 单例模式，管理 Instance、Surface、Device
- **Device**: 
  - 物理设备选择与特性检查
  - 逻辑设备创建与队列家族管理
  - **专用传输队列支持**（自动回退到图形队列）
- **Queue Management**: 支持 Graphics、Compute、Transfer 多队列架构

### 3. Vulkan 资源封装 (vulkan/)
- **Buffer**: 
  - 支持 Staging Buffer 异步数据传输
  - RAII 内存管理与移动语义
  - 自动查找最优内存类型
- **Shader**: SPIR-V 着色器模块加载与管理
- **Swapchain**: 
  - 自适应表面格式与呈现模式选择
  - 帧缓冲与图像视图管理
  - 支持窗口 resize 重建
- **CommandPool**: 
  - 命令缓冲区分配与回收
  - 支持临时与重置标志
  - 线程安全扩展预留

### 4. 3D 高斯散射核心 (gaussian_render_grapics/)
- **GaussianModel**: 高斯点云数据结构与加载
- **GaussianRenderer**: 
  - 继承自 Renderer 基类
  - 管理 RenderPass、Pipeline、DescriptorSet
  - 深度排序与混合渲染
- **RenderPass**: 渲染通道附件配置
- **Pipeline**: 图形管线状态对象（顶点输入、混合、深度测试）

### 5. 工具模块 (utils/)
- **Camera**: 第一人称相机控制，提供 View/Projection 矩阵
- **FileUtils**: 跨平台文件路径解析（支持相对路径）
- **Logger**: 分级日志系统（INFO/WARN/ERROR），支持格式化输出

##  依赖项

- **Vulkan SDK**: 图形 API（需支持 Vulkan 1.0+）
- **GLFW3**: 窗口与输入管理
- **GLM**: OpenGL Mathematics 数学库
- **STB**: 图像加载库（stb_image）
- **CMake**: 3.26+ 构建系统

##  构建说明

### 环境要求
- **编译器**: 支持 C++20 标准（MSVC 2019+, GCC 10+, Clang 10+）
- **CMake**: >= 3.26
- **Vulkan SDK**: 已配置环境变量或系统路径
- **依赖库**: GLFW3、GLM、STB（可通过 vcpkg 或系统包管理器安装）

### 构建步骤
```bash
# 1. 克隆项目
git clone <repository-url>
cd vk-gs

# 2. 创建构建目录
mkdir build && cd build

# 3. 配置项目（可选指定生成器）
cmake .. 
# Windows MSVC: cmake .. -G "Visual Studio 17 2022"
# Linux: cmake .. -G "Unix Makefiles"

# 4. 编译（Release 模式）
cmake --build . --config Release

# 5. 运行沙盒测试
./bin/Release/vk_gs_sandbox.exe  # Windows
./bin/vk_gs_sandbox              # Linux
```

### 着色器编译
项目自动编译 GLSL 着色器为 SPIR-V：
- 使用 **glslc** 编译器（性能更优，错误信息友好）
- 着色器源码位于 `src/shaders/` 目录
- 编译后的 `.spv` 文件自动复制到构建目录

## 使用示例

### 创建自定义应用
```cpp
#include "application.hpp"
#include "gaussian_render_grapics/gaussian_renderer.hpp"
#include "utils/camera.hpp"

class MyGaussianApp : public vk_gs::Application {
public:
    MyGaussianApp() : Application("3DGS Viewer", 1920, 1080) {}
    
protected:
    void initialize() override {
        // 1. 创建高斯渲染器
        auto& context = vk_gs::Context::Instance();
        renderer_ = std::make_unique<vk_gs::GaussianRenderer>(
            context.getDevice(), 
            context.getSurface()
        );
        
        // 2. 加载高斯模型
        gaussian_model_ = std::make_unique<vk_gs::GaussianModel>();
        gaussian_model_->load_from_ply("data/splat.ply");
        
        // 3. 初始化相机
        camera_ = std::make_unique<vk_gs::Camera>();
        camera_->setPosition(glm::vec3(0.0f, 0.0f, 5.0f));
    }
    
    void render() override {
        // 渲染高斯点云
        if (renderer_ && gaussian_model_) {
            renderer_->render(
                *gaussian_model_,
                camera_->getViewMatrix(),
                camera_->getProjectionMatrix()
            );
        }
    }
    
    void onResize(uint32_t width, uint32_t height) override {
        Application::onResize(width, height);
        if (camera_) {
            camera_->setAspectRatio(static_cast<float>(width) / height);
        }
    }
    
private:
    std::unique_ptr<vk_gs::GaussianRenderer> renderer_;
    std::unique_ptr<vk_gs::GaussianModel> gaussian_model_;
    std::unique_ptr<vk_gs::Camera> camera_;
};

int main() {
    try {
        MyGaussianApp app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
```

## 特性

### 已实现功能
- [x] **Vulkan 基础框架**: Instance/Device/Surface 完整初始化流程
- [x] **多队列架构**: 支持专用 Transfer Queue（自动回退机制）
- [x] **资源 RAII 管理**: Buffer/Shader/Swapchain/CommandPool 自动化生命周期
- [x] **Staging Buffer**: 高效的 GPU 数据传输机制
- [x] **模块化架构**: 通用资源与特定渲染逻辑物理隔离
- [x] **相机系统**: 第一人称相机控制与矩阵计算
- [x] **日志系统**: 分级日志与格式化输出
- [x] **跨平台路径**: 自动解析相对路径为可执行文件目录基准
- [x] **CMake 构建**: 自动着色器编译与资源部署

### TODO
- [ ] **完整渲染管线**: 深度预传递、透明混合优化
- [ ] **高斯着色器**: 顶点/片段着色器实现球体光栅化
- [ ] **文件格式支持**: 原生 .ply/.splat 格式解析器
- [ ] **GPU 加速排序**: Compute Shader 实现深度排序
- [ ] **性能优化**: 视锥剔除、LOD、批处理渲染
- [ ] **VMA 内存管理**: 优化资源分配与释放，避免内存泄漏

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。

---

*这是一个活跃开发中的项目，欢迎参与贡献！* 