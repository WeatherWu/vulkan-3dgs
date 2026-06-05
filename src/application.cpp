#include "application.hpp"
#include "window.hpp"
#include "context/context.hpp"
#include "gaussian_renderer/gaussian_renderer.hpp"
#include "gaussian_renderer/gaussian_model.hpp"
#include "utils/logger.hpp"

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
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

    window_->set_drop_callback([this](const std::vector<std::string>& paths) {
        handleDroppedFiles(paths);
    });

    window_->set_scroll_callback([this](double xoffset, double yoffset) {
        handleScroll(xoffset, yoffset);
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
        if (auto* gsRenderer = dynamic_cast<GaussianRenderer*>(renderer_.get())) {
            gsRenderer->setPresentModePreference(present_mode_preference_);
        }
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
    if (renderer_) {
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
        if (gsRenderer && present_mode_dirty_) {
            gsRenderer->setPresentModePreference(present_mode_preference_);
            present_mode_dirty_ = false;
        }

        if (gsRenderer && current_model_) {
            gsRenderer->setRenderData(current_model_, view_matrix_, projection_matrix_, camera_, model_matrix_);
        } else if (!gsRenderer) {
            LOG_WARN("Renderer is not a GaussianRenderer, skipping data setup");
        }
        
        renderer_->render();
    } else {
        static bool warnedMissingModel = false;
        if (!warnedMissingModel) {
            LOG_INFO("No model loaded. Drag a .ply file into the window to load it.");
            warnedMissingModel = true;
        }
    }
}

void Application::setModel(const GaussianModel* model) {
    owned_model_.reset();
    current_model_ = model;
    resetOrbitFromModel();
}

bool Application::loadModelFromFile(const std::string& filename) {
    auto model = std::make_unique<GaussianModel>();
    if (!model->loadFromFile(filename)) {
        LOG_ERROR("Failed to load model: {}", filename);
        return false;
    }

    owned_model_ = std::move(model);
    owned_model_path_ = filename;
    current_model_ = owned_model_.get();
    resetOrbitFromModel();
    updateModelMatrix();

    LOG_INFO("Loaded model from dropped file: {}", filename);
    return true;
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
        orbit_offset_ = offset;
        orbit_up_ = camera_.get_up();
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
        orbit_offset_ = glm::vec3(orbit_radius_, 0.0f, 0.0f);
        orbit_up_ = glm::vec3(0.0f, 1.0f, 0.0f);
        return;
    }

    orbit_center_ = current_model_->get_center();
    orbit_radius_ = std::max(current_model_->get_radius() * 2.5f, 1.0f);
    orbit_angle_ = 0.0f;
    orbit_pitch_ = 0.0f;
    orbit_offset_ = glm::vec3(orbit_radius_, 0.0f, 0.0f);
    orbit_up_ = glm::vec3(0.0f, 1.0f, 0.0f);
}

void Application::updateOrbitCamera(float delta_time) {
    if (!orbit_camera_enabled_ || !current_model_) {
        return;
    }

    updateOrbitInput(delta_time);

    float offsetLength = glm::length(orbit_offset_);
    if (offsetLength <= 1e-5f) {
        orbit_offset_ = glm::vec3(orbit_radius_, 0.0f, 0.0f);
    } else {
        orbit_offset_ = glm::normalize(orbit_offset_) * orbit_radius_;
    }

    camera_.set_position(orbit_center_ + orbit_offset_);
    camera_.look_at(orbit_center_, orbit_up_);
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
    syncOrbitAnglesFromOffset();
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

    glm::vec2 drag(static_cast<float>(deltaX), static_cast<float>(deltaY));
    float dragLength = glm::length(drag);
    if (dragLength <= 1e-5f) {
        return;
    }

    glm::vec3 viewDirection = glm::normalize(orbit_center_ - camera_.get_position());
    glm::vec3 cameraUp = glm::normalize(orbit_up_);
    glm::vec3 cameraRight = glm::cross(viewDirection, cameraUp);
    if (glm::length(cameraRight) <= 1e-5f) {
        cameraRight = camera_.get_right();
    }
    cameraRight = glm::normalize(cameraRight);
    cameraUp = glm::normalize(glm::cross(cameraRight, viewDirection));

    glm::vec3 axisWorld = drag.y * cameraRight + drag.x * cameraUp;
    float axisLength = glm::length(axisWorld);
    if (axisLength <= 1e-5f) {
        return;
    }
    axisWorld /= axisLength;

    // Move the camera in the inverse direction so the object appears to rotate
    // with the mouse, matching a direct model-rotation trackball.
    float angle = -dragLength * orbit_mouse_sensitivity_;
    glm::quat deltaRotation = glm::angleAxis(angle, axisWorld);
    orbit_offset_ = deltaRotation * orbit_offset_;
    orbit_up_ = glm::normalize(deltaRotation * cameraUp);
    orbit_radius_ = std::max(glm::length(orbit_offset_), 0.05f);
    syncOrbitAnglesFromOffset();
}

void Application::syncOrbitAnglesFromOffset() {
    float distance = glm::length(orbit_offset_);
    if (distance <= 1e-5f) {
        orbit_angle_ = 0.0f;
        orbit_pitch_ = 0.0f;
        return;
    }

    orbit_angle_ = std::atan2(orbit_offset_.z, orbit_offset_.x);
    orbit_pitch_ = std::asin(std::clamp(orbit_offset_.y / distance, -1.0f, 1.0f));
}

void Application::rebuildOrbitOffsetFromAngles() {
    float cosPitch = std::cos(orbit_pitch_);
    orbit_offset_ = glm::vec3(
        std::cos(orbit_angle_) * cosPitch * orbit_radius_,
        std::sin(orbit_pitch_) * orbit_radius_,
        std::sin(orbit_angle_) * cosPitch * orbit_radius_
    );
}

void Application::drawImGuiControls() {
    ImGui::Begin("Camera");

    const ImGuiIO& io = ImGui::GetIO();
    ImGui::Text("FPS %.1f", io.Framerate);
    if (io.Framerate > 0.0f) {
        ImGui::Text("Frame %.2f ms", 1000.0f / io.Framerate);
    }
    ImGui::Separator();

    int presentModeIndex = static_cast<int>(present_mode_preference_);
    const char* presentModeLabels[] = {
        "最高帧率",
        "低延迟",
        "垂直同步",
    };
    if (ImGui::Combo("帧率模式", &presentModeIndex, presentModeLabels, 3)) {
        present_mode_preference_ = static_cast<PresentModePreference>(presentModeIndex);
        present_mode_dirty_ = true;
    }
    ImGui::Separator();

    bool flipY = flip_model_y_;
    bool flipZ = flip_model_z_;
    if (ImGui::Checkbox("Flip Y", &flipY)) {
        flip_model_y_ = flipY;
        updateModelMatrix();
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("Flip Z", &flipZ)) {
        flip_model_z_ = flipZ;
        updateModelMatrix();
    }

    ImGui::Checkbox("Orbit", &orbit_camera_enabled_);
    ImGui::SliderFloat("Sensitivity", &orbit_mouse_sensitivity_, 0.001f, 0.02f, "%.3f");
    ImGui::SliderFloat("Dolly Speed", &orbit_zoom_sensitivity_, 0.02f, 0.5f, "%.2f");

    float radiusLimit = std::max(orbit_radius_ * 3.0f, 20.0f);
    ImGui::SliderFloat("Distance", &orbit_radius_, 0.1f, radiusLimit, "%.2f");

    float pitchDegrees = glm::degrees(orbit_pitch_);
    if (ImGui::SliderFloat("Pitch", &pitchDegrees, -89.0f, 89.0f, "%.1f deg")) {
        orbit_pitch_ = glm::radians(pitchDegrees);
        rebuildOrbitOffsetFromAngles();
    }

    if (ImGui::Button("Reset")) {
        resetOrbitFromModel();
    }

    const glm::vec3& position = camera_.get_position();
    ImGui::Text("Position %.2f %.2f %.2f", position.x, position.y, position.z);
    ImGui::End();
}

void Application::handleDroppedFiles(const std::vector<std::string>& paths) {
    if (paths.empty()) {
        return;
    }

    if (paths.size() > 1) {
        LOG_WARN("Multiple files dropped; loading the first one only");
    }

    loadModelFromFile(paths.front());
}

void Application::handleScroll(double xoffset, double yoffset) {
    (void)xoffset;

    if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse) {
        return;
    }

    float zoomFactor = 1.0f - static_cast<float>(yoffset) * orbit_zoom_sensitivity_;
    zoomFactor = std::clamp(zoomFactor, 0.1f, 4.0f);
    orbit_radius_ = std::max(0.05f, orbit_radius_ * zoomFactor);
    if (glm::length(orbit_offset_) > 1e-5f) {
        orbit_offset_ = glm::normalize(orbit_offset_) * orbit_radius_;
    }

    if (!orbit_camera_enabled_ && has_true_camera_) {
        glm::vec3 toCenter = orbit_center_ - camera_.get_position();
        float currentDistance = glm::length(toCenter);
        if (currentDistance > 1e-4f) {
            camera_.set_position(orbit_center_ - glm::normalize(toCenter) * orbit_radius_);
            camera_.look_at(orbit_center_, orbit_up_);
            orbit_offset_ = camera_.get_position() - orbit_center_;
            syncOrbitAnglesFromOffset();
            view_matrix_ = camera_.get_view_matrix();
        }
    }
}

void Application::updateModelMatrix() {
    model_matrix_ = glm::mat4(1.0f);
    if (!current_model_ || current_model_->isEmpty()) {
        return;
    }

    glm::vec3 scaling(1.0f);
    scaling.y = flip_model_y_ ? -1.0f : 1.0f;
    scaling.z = flip_model_z_ ? -1.0f : 1.0f;

    glm::vec3 center = current_model_->get_center();
    model_matrix_ = glm::translate(glm::mat4(1.0f), center) *
                    glm::scale(glm::mat4(1.0f), scaling) *
                    glm::translate(glm::mat4(1.0f), -center);
}

}
