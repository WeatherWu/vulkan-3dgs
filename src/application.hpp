#pragma once

#include <memory>
#include <string>
#include <glm/glm.hpp>

namespace vk_gs {

class Window;
class Context;
class Renderer;
class GaussianModel;

// 渲染模式枚举
enum class RenderMode {
    GaussianGraphics,  // 高斯图形渲染
    GaussianCompute,   // 高斯计算渲染
    StopthePoping
    // 未来可以添加其他渲染模式
    // RayTracing,      // 光线追踪
    // ForwardRendering,// 前向渲染
};

class Application {
public:
    Application(const std::string& title, int width, int height, RenderMode mode = RenderMode::GaussianGraphics);
    virtual ~Application();
    
    void run();
    
    // 运行时切换渲染模式
    void switchRenderMode(RenderMode mode);
    RenderMode getCurrentRenderMode() const { return current_mode_; }
    
    // 设置模型数据（由外部调用）
    void setModel(const GaussianModel* model);
    
    // 设置相机参数（由外部调用）
    void setCamera(const glm::mat4& view, const glm::mat4& projection);
    
protected:
    virtual void initialize();
    virtual void update(float delta_time);
    virtual void render();
    virtual void cleanup();
    
protected:
    std::unique_ptr<Window> window_;
    Context* vulkan_context_ = nullptr;  // 使用指针指向单例
    std::unique_ptr<Renderer> renderer_;
    
    // 渲染数据
    const GaussianModel* current_model_ = nullptr;
    glm::mat4 view_matrix_ = glm::mat4(1.0f);
    glm::mat4 projection_matrix_ = glm::mat4(1.0f);

    RenderMode current_mode_;
    bool running_ = true;
    
private:
    // 工厂方法：根据渲染模式创建对应的渲染器实例
    std::unique_ptr<Renderer> createRenderer(RenderMode mode);
};

} // namespace vk_gs
