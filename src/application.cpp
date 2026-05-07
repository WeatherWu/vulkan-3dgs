#include "application.hpp"
#include "window.hpp"
#include "context/context.hpp"
#include "gaussian_renderer_compute/gaussian_renderer.hpp"
#include "gaussian_renderer_compute/gaussian_model.hpp"
#include "utils/logger.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

namespace vk_gs {

Application::Application(const std::string& title, int width, int height, RenderMode mode) 
    : current_mode_(mode) {
    // 设置日志级别为DEBUG（仅在Debug模式下）
#ifdef _DEBUG
    vk_gs::Logger::get_instance().set_level(LogLevel::DEBUG_VKGS);
#endif
    
    LOG_INFO("Initializing Vulkan+3DGS Application");
    
    // 初始化GLFW
    if (!glfwInit()) {
        LOG_ERROR("Failed to initialize GLFW");
        throw std::runtime_error("GLFW initialization failed");
    }
    
    // 创建窗口
    window_ = std::make_unique<Window>(title, width, height);
    
    // 获取Vulkan上下文单例并初始化
    vulkan_context_ = &Context::Instance();
    vulkan_context_->initialize(window_->get_handle());
    
    // 使用工厂方法创建具体的渲染器实例
    renderer_ = createRenderer(current_mode_);
    renderer_->initialize(window_->get_handle());
    
    initialize();
}

Application::~Application() {
    cleanup();
    glfwTerminate();
}

void Application::run() {
    LOG_INFO("Starting application main loop");
    
    while (!window_->should_close() && running_) {
        tick();
    }
}

void Application::tick() {
    window_->poll_events();
    render();
}

void Application::switchRenderMode(RenderMode mode) {
    if (current_mode_ == mode) {
        return;
    }
    
    LOG_INFO("Switching render mode from {} to {}", 
             static_cast<int>(current_mode_), 
             static_cast<int>(mode));
    
    // 1. 清理旧渲染器
    if (renderer_) {
        renderer_->cleanup();
        renderer_.reset();
    }
    
    // 2. 创建新渲染器
    current_mode_ = mode;
    renderer_ = createRenderer(current_mode_);
    
    // 3. 重新初始化
    if (renderer_) {
        renderer_->initialize(window_->get_handle());
        LOG_INFO("Successfully switched to new render mode");
    } else {
        LOG_ERROR("Failed to create renderer for mode: {}", static_cast<int>(mode));
    }
}

std::unique_ptr<Renderer> Application::createRenderer(RenderMode mode) {
    switch (mode) {
        case RenderMode::GaussianGraphics:
            LOG_INFO("Creating Gaussian Graphics Renderer");
            return std::make_unique<GaussianRenderer>();
            
        // 未来可以添加其他渲染模式
        // case RenderMode::RayTracing:
        //     LOG_INFO("Creating Ray Tracing Renderer");
        //     return std::make_unique<RayTracingRenderer>();
            
        default:
            LOG_ERROR("Unknown render mode: {}", static_cast<int>(mode));
            return nullptr;
    }
}

void Application::initialize() {
    // Context已经在构造函数中初始化过了，这里不需要再次初始化
    LOG_INFO("Application initialization completed");
}

void Application::update(float delta_time) {
    // 子类可以重写此方法
    // 默认实现：使用简单的旋转相机
    static float time = 0.0f;
    time += delta_time;
    
    // 简单的环绕相机
    float radius = 3.0f;
    float camX = std::sin(time * 0.5f) * radius;
    float camZ = std::cos(time * 0.5f) * radius;
    
    glm::vec3 cameraPos(camX, 0.0f, camZ);
    glm::vec3 target(0.0f, 0.0f, 0.0f);
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    
    view_matrix_ = glm::lookAt(cameraPos, target, up);
    projection_matrix_ = glm::perspective(glm::radians(45.0f), 
                                          1280.0f / 720.0f, 
                                          0.1f, 100.0f);
    projection_matrix_[1][1] *= -1.0f;
}

void Application::render() {
    // 调用具体渲染器的渲染逻辑
    if (renderer_ && current_model_) {
        LOG_INFO("Starting render frame");
        
        // 传递模型数据和相机参数
        auto* gsRenderer = dynamic_cast<GaussianRenderer*>(renderer_.get());
        if (gsRenderer) {
            LOG_INFO("Setting render data with {} points", 
                     std::distance(current_model_->begin(), current_model_->end()));
            gsRenderer->setRenderData(current_model_, view_matrix_, projection_matrix_, camera_);
        } else {
            LOG_WARN("Renderer is not a GaussianRenderer, skipping data setup");
        }
        
        LOG_INFO("Calling renderer_->render()");
        renderer_->render();
        LOG_INFO("Frame rendered successfully");
    } else {
        LOG_WARN("Skipping render: renderer={} model={}", 
                renderer_ ? "valid" : "null", 
                current_model_ ? "valid" : "null");
    }
}

void Application::setModel(const GaussianModel* model) {
    current_model_ = model;
    LOG_INFO("Model set for rendering");
}

void Application::setCamera(const glm::mat4& view, const glm::mat4& projection) {
    view_matrix_ = view;
    projection_matrix_ = projection;
}

void Application::setTrueCamera(const vk_gs::Camera& camera) {
    camera_ = camera;
}

void Application::cleanup() {
    LOG_INFO("Cleaning up application");
    if (renderer_) {
        renderer_->cleanup();
    }
    // 注意：vulkan_context_是单例，不需要在这里清理
}

}
