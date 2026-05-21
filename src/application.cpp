#include "application.hpp"
#include "window.hpp"
#include "context/context.hpp"
#include "gaussian_renderer_compute/gaussian_renderer.hpp"
#include "gaussian_renderer_compute/gaussian_model.hpp"
#include "utils/logger.hpp"

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
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
    renderer_->setImGuiDrawCallback([this]() {
        drawImGuiControls();
    });

    window_->set_resize_callback([this](int width, int height) {
        if (width <= 0 || height <= 0) {
            return;
        }

        if (renderer_) {
            renderer_->onResize(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
        }

        if (has_true_camera_) {
            float aspect = static_cast<float>(width) / static_cast<float>(height);
            view_matrix_ = camera_.get_view_matrix();
            projection_matrix_ = camera_.get_projection_matrix(aspect, camera_.get_fov());
        }
    });
    
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

    double currentTime = glfwGetTime();
    float deltaTime = 0.0f;
    if (has_last_tick_time_) {
        deltaTime = static_cast<float>(currentTime - last_tick_time_);
        deltaTime = std::clamp(deltaTime, 0.0f, 0.1f);
    } else {
        has_last_tick_time_ = true;
    }
    last_tick_time_ = currentTime;

    update(deltaTime);
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
        renderer_->setImGuiDrawCallback([this]() {
            drawImGuiControls();
        });
        LOG_INFO("Successfully switched to new render mode");
    } else {
        LOG_ERROR("Failed to create renderer for mode: {}", static_cast<int>(mode));
    }
}

std::unique_ptr<Renderer> Application::createRenderer(RenderMode mode) {
    switch (mode) {
        case RenderMode::GaussianGraphics:
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
}

void Application::update(float delta_time) {
    updateOrbitCamera(delta_time);
}

void Application::render() {
    // 调用具体渲染器的渲染逻辑
    if (renderer_ && current_model_) {
        if (has_true_camera_ && window_) {
            int framebufferWidth = 0;
            int framebufferHeight = 0;
            glfwGetFramebufferSize(window_->get_handle(), &framebufferWidth, &framebufferHeight);
            if (framebufferWidth > 0 && framebufferHeight > 0) {
                float aspect = static_cast<float>(framebufferWidth) / static_cast<float>(framebufferHeight);
                view_matrix_ = camera_.get_view_matrix();
                projection_matrix_ = camera_.get_projection_matrix(aspect, camera_.get_fov());
            }
        }
        
        // 传递模型数据和相机参数
        auto* gsRenderer = dynamic_cast<GaussianRenderer*>(renderer_.get());
        if (gsRenderer) {
            gsRenderer->setRenderData(current_model_, view_matrix_, projection_matrix_, camera_);
        } else {
            LOG_WARN("Renderer is not a GaussianRenderer, skipping data setup");
        }
        
        renderer_->render();
    } else {
        LOG_WARN("Skipping render: renderer={} model={}", 
                renderer_ ? "valid" : "null", 
                current_model_ ? "valid" : "null");
    }
}

void Application::setModel(const GaussianModel* model) {
    current_model_ = model;
    resetOrbitFromModel();
}

void Application::setCamera(const glm::mat4& view, const glm::mat4& projection) {
    view_matrix_ = view;
    projection_matrix_ = projection;
}

void Application::setTrueCamera(const vk_gs::Camera& camera) {
    camera_ = camera;
    has_true_camera_ = true;

    glm::vec3 offset = camera_.get_position() - orbit_center_;
    float horizontalDistance = std::sqrt(offset.x * offset.x + offset.z * offset.z);
    float distance = glm::length(offset);
    if (distance > 0.001f) {
        orbit_radius_ = distance;
        orbit_angle_ = std::atan2(offset.z, offset.x);
        orbit_pitch_ = std::asin(std::clamp(offset.y / distance, -1.0f, 1.0f));
    } else if (horizontalDistance > 0.001f) {
        orbit_angle_ = std::atan2(offset.z, offset.x);
    }
}

void Application::cleanup() {
    if (renderer_) {
        renderer_->cleanup();
    }
    // 注意：vulkan_context_是单例，不需要在这里清理
}

void Application::resetOrbitFromModel() {
    if (!current_model_ || current_model_->isEmpty()) {
        orbit_center_ = glm::vec3(0.0f);
        orbit_radius_ = 5.0f;
        orbit_angle_ = 0.0f;
        orbit_pitch_ = 0.0f;
        return;
    }

    orbit_center_ = current_model_->get_center();
    orbit_radius_ = std::max(current_model_->get_radius() * 2.5f, 1.0f);
    orbit_angle_ = 0.0f;
    orbit_pitch_ = 0.0f;
}

void Application::updateOrbitCamera(float delta_time) {
    if (!orbit_camera_enabled_ || !current_model_) {
        return;
    }

    updateOrbitInput(delta_time);

    float cosPitch = std::cos(orbit_pitch_);

    glm::vec3 offset(
        std::cos(orbit_angle_) * cosPitch * orbit_radius_,
        std::sin(orbit_pitch_) * orbit_radius_,
        std::sin(orbit_angle_) * cosPitch * orbit_radius_
    );

    camera_.set_position(orbit_center_ + offset);
    camera_.set_target(orbit_center_);
    has_true_camera_ = true;

    if (!window_) {
        return;
    }

    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetFramebufferSize(window_->get_handle(), &framebufferWidth, &framebufferHeight);
    if (framebufferWidth <= 0 || framebufferHeight <= 0) {
        return;
    }

    float aspect = static_cast<float>(framebufferWidth) / static_cast<float>(framebufferHeight);
    view_matrix_ = camera_.get_view_matrix();
    projection_matrix_ = camera_.get_projection_matrix(aspect, camera_.get_fov());
}

void Application::updateOrbitInput(float delta_time) {
    (void)delta_time;

    if (!window_) {
        return;
    }

    bool imguiCapturingMouse = false;
    if (ImGui::GetCurrentContext()) {
        imguiCapturingMouse = ImGui::GetIO().WantCaptureMouse;
    }

    bool leftDown = glfwGetMouseButton(window_->get_handle(), GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    double mouseX = 0.0;
    double mouseY = 0.0;
    glfwGetCursorPos(window_->get_handle(), &mouseX, &mouseY);

    if (!leftDown || imguiCapturingMouse) {
        orbit_dragging_ = false;
        last_mouse_x_ = mouseX;
        last_mouse_y_ = mouseY;
        return;
    }

    if (!orbit_dragging_) {
        orbit_dragging_ = true;
        last_mouse_x_ = mouseX;
        last_mouse_y_ = mouseY;
        return;
    }

    double deltaX = mouseX - last_mouse_x_;
    double deltaY = mouseY - last_mouse_y_;
    last_mouse_x_ = mouseX;
    last_mouse_y_ = mouseY;

    orbit_angle_ -= static_cast<float>(deltaX) * orbit_mouse_sensitivity_;
    orbit_pitch_ -= static_cast<float>(deltaY) * orbit_mouse_sensitivity_;
    orbit_pitch_ = std::clamp(orbit_pitch_, glm::radians(-89.0f), glm::radians(89.0f));
}

void Application::drawImGuiControls() {
    ImGui::Begin("Camera");
    ImGui::Checkbox("Orbit", &orbit_camera_enabled_);
    ImGui::SliderFloat("Sensitivity", &orbit_mouse_sensitivity_, 0.001f, 0.02f, "%.3f");

    float radiusLimit = std::max(orbit_radius_ * 3.0f, 20.0f);
    ImGui::SliderFloat("Radius", &orbit_radius_, 0.1f, radiusLimit, "%.2f");

    float pitchDegrees = glm::degrees(orbit_pitch_);
    if (ImGui::SliderFloat("Pitch", &pitchDegrees, -89.0f, 89.0f, "%.1f deg")) {
        orbit_pitch_ = glm::radians(pitchDegrees);
    }

    if (ImGui::Button("Reset")) {
        resetOrbitFromModel();
    }

    const glm::vec3& position = camera_.get_position();
    ImGui::Text("Position %.2f %.2f %.2f", position.x, position.y, position.z);
    ImGui::End();
}

}
