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
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <exception>

#include <ImGuiFileDialog.h>
namespace vk_gs {

namespace {

void copyToInputBuffer(std::array<char, 512>& buffer, const std::string& value) {
    std::fill(buffer.begin(), buffer.end(), '\0');
    std::copy_n(value.data(), std::min(value.size(), buffer.size() - 1), buffer.data());
}

void copyToInputBuffer(std::array<char, 256>& buffer, const std::string& value) {
    std::fill(buffer.begin(), buffer.end(), '\0');
    std::copy_n(value.data(), std::min(value.size(), buffer.size() - 1), buffer.data());
}

std::string inputBufferString(const std::array<char, 512>& buffer) {
    return std::string(buffer.data());
}

std::string inputBufferString(const std::array<char, 256>& buffer) {
    return std::string(buffer.data());
}

bool hasExtension(const std::filesystem::path& path, const std::string& extension) {
    std::string actual = path.extension().string();
    std::transform(actual.begin(), actual.end(), actual.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return actual == extension;
}

std::filesystem::path existingDirectoryOrCurrent(const std::string& value) {
    std::filesystem::path path(value);
    if (std::filesystem::is_directory(path)) {
        return path;
    }
    return std::filesystem::current_path();
}

} // namespace

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

    copyToInputBuffer(training_dataset_path_, "data/mipnerf360/bicycle");
    copyToInputBuffer(training_output_dir_, "output");
    copyToInputBuffer(training_output_name_, "bicycle.ply");
    training_status_ = "Training dataset is not loaded";
    
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

    if (training_running_) {
        try {
            for (uint32_t i = 0; i < std::max(training_steps_per_frame_, 1u); ++i) {
                runTrainingStepFromUi();
            }
        } catch (const std::exception& error) {
            training_running_ = false;
            setTrainingError(error.what());
        }
    }
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
    training_.cleanup();
    training_initialized_ = false;

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

    drawTrainingControls();
}

void Application::handleDroppedFiles(const std::vector<std::string>& paths) {
    if (paths.empty()) {
        return;
    }

    if (paths.size() > 1) {
        LOG_WARN("Multiple files dropped; loading the first one only");
    }

    std::filesystem::path droppedPath(paths.front());
    if (std::filesystem::is_directory(droppedPath)) {
        copyToInputBuffer(training_dataset_path_, droppedPath.string());
        syncDefaultOutputNameFromDataset();
        validateTrainingDatasetFromUi();
        return;
    }

    if (hasExtension(droppedPath, ".ply")) {
        loadModelFromFile(paths.front());
        return;
    }

    setTrainingError("Unsupported dropped file. Drop a .ply model or a dataset folder.");
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

void Application::drawTrainingControls() {
    ImGui::Begin("Training");

    if (training_running_) {
        ImGui::BeginDisabled();
    }

    ImGui::InputText("Dataset Folder", training_dataset_path_.data(), training_dataset_path_.size());
    ImGui::SameLine();
    if (ImGui::Button("Browse##DatasetFolder")) {
        chooseTrainingDatasetFolderFromUi();
    }
    ImGui::InputInt("Downscale", &training_downscale_);
    training_downscale_ = std::clamp(training_downscale_, 1, 16);

    const bool canEditTrainingSetup = !training_running_;

    if (ImGui::CollapsingHeader("Initialization", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!canEditTrainingSetup) {
            ImGui::BeginDisabled();
        }
        ImGui::InputScalar("Fallback Gaussians", ImGuiDataType_U32, &training_initial_gaussians_);
        training_initial_gaussians_ = std::clamp(training_initial_gaussians_, 1u, 1000000u);
        ImGui::InputScalar("Random Seed", ImGuiDataType_U32, &training_random_seed_);
        training_random_seed_ = std::max(training_random_seed_, 1u);
        ImGui::InputFloat("Initial Opacity", &training_initial_opacity_, 0.01f, 0.05f, "%.4f");
        training_initial_opacity_ = std::clamp(training_initial_opacity_, 0.001f, 0.99f);
        ImGui::InputFloat("Scene Radius Scale", &training_scene_radius_scale_, 0.1f, 1.0f, "%.3f");
        training_scene_radius_scale_ = std::clamp(training_scene_radius_scale_, 0.01f, 100.0f);
        if (!canEditTrainingSetup) {
            ImGui::EndDisabled();
        }
    }

    if (ImGui::Button("Validate")) {
        validateTrainingDatasetFromUi();
    }
    ImGui::SameLine();
    if (ImGui::Button("Load Dataset")) {
        loadTrainingDatasetFromUi();
    }

    if (training_running_) {
        ImGui::EndDisabled();
    }

    if (training_dataset_valid_) {
        ImGui::Text("Frames %u", training_frame_count_);
        ImGui::Text("Resolution %ux%u", training_width_, training_height_);
    }

    ImGui::Separator();
    if (ImGui::CollapsingHeader("Training", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::InputScalar("Steps/Frame", ImGuiDataType_U32, &training_steps_per_frame_);
        training_steps_per_frame_ = std::max(training_steps_per_frame_, 1u);
        if (!canEditTrainingSetup) {
            ImGui::BeginDisabled();
        }
        ImGui::InputFloat("Learning Rate", &training_learning_rate_, 0.0001f, 0.001f, "%.6g");
        training_learning_rate_ = std::max(training_learning_rate_, 0.0f);
        ImGui::InputFloat("Adam Beta1", &training_adam_beta1_, 0.01f, 0.05f, "%.6g");
        training_adam_beta1_ = std::clamp(training_adam_beta1_, 0.0f, 0.999999f);
        ImGui::InputFloat("Adam Beta2", &training_adam_beta2_, 0.001f, 0.01f, "%.6g");
        training_adam_beta2_ = std::clamp(training_adam_beta2_, 0.0f, 0.999999f);
        ImGui::InputFloat("Adam Epsilon", &training_adam_epsilon_, 0.00000001f, 0.0000001f, "%.6g");
        training_adam_epsilon_ = std::max(training_adam_epsilon_, 1e-12f);
        ImGui::InputFloat("Gradient Clip", &training_grad_clip_, 10.0f, 100.0f, "%.6g");
        training_grad_clip_ = std::max(training_grad_clip_, 1e-8f);
        if (!canEditTrainingSetup) {
            ImGui::EndDisabled();
        }
    }

    if (ImGui::CollapsingHeader("Densification / Pruning", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!canEditTrainingSetup) {
            ImGui::BeginDisabled();
        }
        ImGui::Checkbox("Enable Densification", &training_densification_enabled_);
        ImGui::InputScalar("Densify From", ImGuiDataType_U32, &training_densify_from_iteration_);
        ImGui::InputScalar("Densify Until", ImGuiDataType_U32, &training_densify_until_iteration_);
        training_densify_until_iteration_ = std::max(training_densify_until_iteration_,
                                                     training_densify_from_iteration_);
        ImGui::InputScalar("Densify Interval", ImGuiDataType_U32, &training_densification_interval_);
        training_densification_interval_ = std::max(training_densification_interval_, 1u);
        ImGui::InputScalar("Opacity Reset Interval", ImGuiDataType_U32, &training_opacity_reset_interval_);
        training_opacity_reset_interval_ = std::max(training_opacity_reset_interval_, 1u);
        ImGui::InputScalar("Max Gaussians", ImGuiDataType_U32, &training_max_gaussians_);
        training_max_gaussians_ = std::clamp(training_max_gaussians_, 1u, 10000000u);
        ImGui::InputScalar("Split Children", ImGuiDataType_U32, &training_split_children_);
        training_split_children_ = std::clamp(training_split_children_, 2u, 8u);
        ImGui::InputFloat("Gradient Threshold", &training_densify_grad_threshold_, 0.00001f, 0.0001f, "%.6g");
        training_densify_grad_threshold_ = std::max(training_densify_grad_threshold_, 0.0f);
        ImGui::InputFloat("Min Opacity", &training_min_opacity_, 0.001f, 0.01f, "%.6g");
        training_min_opacity_ = std::clamp(training_min_opacity_, 0.0f, 0.99f);
        ImGui::InputFloat("Percent Dense", &training_percent_dense_, 0.001f, 0.01f, "%.6g");
        training_percent_dense_ = std::clamp(training_percent_dense_, 0.0f, 1.0f);
        ImGui::InputFloat("Screen Prune Size", &training_screen_size_prune_threshold_, 1.0f, 10.0f, "%.3f");
        training_screen_size_prune_threshold_ = std::max(training_screen_size_prune_threshold_, 0.0f);
        if (!canEditTrainingSetup) {
            ImGui::EndDisabled();
        }
    }

    bool canTrain = training_dataset_loaded_;
    if (!canTrain) {
        ImGui::BeginDisabled();
    }

    if (ImGui::Button(training_running_ ? "Pause Training" : "Start Training")) {
        if (training_running_) {
            training_running_ = false;
        } else {
            try {
                applyTrainingConfigFromUi();
                initializeTrainingIfNeeded();
                training_running_ = true;
            } catch (const std::exception& error) {
                training_running_ = false;
                setTrainingError(error.what());
            }
        }
    }

    if (!canTrain) {
        ImGui::EndDisabled();
    }

    ImGui::Text("Steps %llu", static_cast<unsigned long long>(training_steps_done_));
    ImGui::Text("Gaussians %u", training_.gaussianCount());
    ImGui::Text("Frame %llu / %u",
                static_cast<unsigned long long>(training_.hasDataset() ? training_.currentFrameIndex() : 0),
                training_frame_count_);

    ImGui::Separator();
    ImGui::InputText("Output Folder", training_output_dir_.data(), training_output_dir_.size());
    ImGui::SameLine();
    if (ImGui::Button("Browse##OutputFolder")) {
        chooseTrainingOutputFolderFromUi();
    }
    ImGui::InputText("PLY Name", training_output_name_.data(), training_output_name_.size());
    if (ImGui::Button("Save PLY")) {
        exportTrainingModelFromUi();
    }
    ImGui::SameLine();
    if (ImGui::Button("Save As")) {
        saveTrainingPlyAsFromUi();
    }

    if (!training_status_.empty()) {
        ImGui::TextWrapped("%s", training_status_.c_str());
    }

    if (training_error_popup_pending_) {
        ImGui::OpenPopup("Training Error");
        training_error_popup_pending_ = false;
    }
    if (ImGui::BeginPopupModal("Training Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped("%s", training_error_.c_str());
        if (ImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    drawTrainingFileDialogs();
    ImGui::End();
}

void Application::validateTrainingDatasetFromUi() {
    try {
        auto validation = TrainingDatasetLoader::validateMipNeRF360Scene(inputBufferString(training_dataset_path_),
                                                                         static_cast<uint32_t>(training_downscale_));
        training_dataset_valid_ = validation.valid;
        training_frame_count_ = validation.frameCount;
        training_width_ = validation.width;
        training_height_ = validation.height;
        if (!validation.valid) {
            setTrainingError(validation.message);
            return;
        }
        syncDefaultOutputNameFromDataset();
        setTrainingStatus("Dataset valid: " + std::to_string(training_frame_count_) +
                          " frames, " + std::to_string(training_width_) +
                          "x" + std::to_string(training_height_));
    } catch (const std::exception& error) {
        training_dataset_valid_ = false;
        setTrainingError(error.what());
    }
}

void Application::loadTrainingDatasetFromUi() {
    validateTrainingDatasetFromUi();
    if (!training_dataset_valid_) {
        return;
    }

    try {
        auto& context = Context::Instance();
        auto& device = context.getDevice();
        auto transferFamily = device.getQueueFamilyIndices().transferIndex.value_or(
            device.getQueueFamilyIndices().graphicsIndex.value());
        if (training_.isInitialized()) {
            training_.cleanup();
            training_initialized_ = false;
        }
        training_.initialize(context.Device(),
                             context.PhysicalDevice(),
                             context.getTransferQueue(),
                             transferFamily);
        training_.loadMipNeRF360Dataset(inputBufferString(training_dataset_path_),
                                        static_cast<uint32_t>(training_downscale_));
        training_initialized_ = false;
        training_dataset_loaded_ = true;
        training_steps_done_ = 0;
        setTrainingStatus("Dataset loaded. Press Start Training to initialize and train.");
    } catch (const std::exception& error) {
        training_dataset_loaded_ = false;
        setTrainingError(error.what());
    }
}

void Application::initializeTrainingIfNeeded() {
    if (training_initialized_) {
        return;
    }

    if (!renderer_) {
        throw std::runtime_error("Renderer is not initialized");
    }

    auto& context = Context::Instance();
    auto& device = context.getDevice();
    auto computeFamily = device.getQueueFamilyIndices().computeIndex.value_or(
        device.getQueueFamilyIndices().graphicsIndex.value());
    auto transferFamily = device.getQueueFamilyIndices().transferIndex.value_or(
        device.getQueueFamilyIndices().graphicsIndex.value());

    training_.initialize(context.Device(),
                         context.PhysicalDevice(),
                         context.getTransferQueue(),
                         transferFamily);
    training_.initializeTrainingRenderers(context.Device(),
                                          context.PhysicalDevice(),
                                          device.getComputeQueue(),
                                          computeFamily,
                                          std::max(training_.gaussianCount(), 1u),
                                          TrainingExtent{std::max(training_width_, 1u), std::max(training_height_, 1u)});
    training_initialized_ = true;
}

void Application::runTrainingStepFromUi() {
    if (!training_dataset_loaded_) {
        throw std::runtime_error("Load a training dataset before starting training");
    }
    training_.trainStep();
    ++training_steps_done_;
}

void Application::exportTrainingModelFromUi() {
    if (!training_.hasTrainableModel()) {
        setTrainingError("No trained Gaussian model is available to export");
        return;
    }

    try {
        exportTrainingModelToPath(outputPlyPath());
    } catch (const std::exception& error) {
        setTrainingError(error.what());
    }
}

void Application::exportTrainingModelToPath(const std::filesystem::path& path) {
    if (!training_.hasTrainableModel()) {
        throw std::runtime_error("No trained Gaussian model is available to export");
    }

    std::filesystem::path outputPath = path;
    if (outputPath.extension().empty()) {
        outputPath += ".ply";
    }

    if (!training_.exportToPLY(outputPath)) {
        throw std::runtime_error("Failed to export PLY: " + outputPath.string());
    }

    setTrainingStatus("Saved PLY: " + outputPath.string());
}

void Application::chooseTrainingDatasetFolderFromUi() {
    IGFD::FileDialogConfig config;
    config.path = existingDirectoryOrCurrent(inputBufferString(training_dataset_path_)).string();
    config.flags = ImGuiFileDialogFlags_Modal;
    ImGuiFileDialog::Instance()->OpenDialog("TrainingDatasetFolderDialog",
                                            "Select Training Dataset Folder",
                                            nullptr,
                                            config);
}

void Application::chooseTrainingOutputFolderFromUi() {
    IGFD::FileDialogConfig config;
    config.path = existingDirectoryOrCurrent(inputBufferString(training_output_dir_)).string();
    config.flags = ImGuiFileDialogFlags_Modal;
    ImGuiFileDialog::Instance()->OpenDialog("TrainingOutputFolderDialog",
                                            "Select Output Folder",
                                            nullptr,
                                            config);
}

void Application::saveTrainingPlyAsFromUi() {
    IGFD::FileDialogConfig config;
    config.path = existingDirectoryOrCurrent(inputBufferString(training_output_dir_)).string();
    config.fileName = inputBufferString(training_output_name_);
    config.flags = ImGuiFileDialogFlags_Modal;
    ImGuiFileDialog::Instance()->OpenDialog("TrainingSavePlyDialog",
                                            "Save Training Result",
                                            ".ply",
                                            config);
}

void Application::drawTrainingFileDialogs() {
    const ImVec2 minSize(620.0f, 360.0f);
    const ImVec2 maxSize(FLT_MAX, FLT_MAX);

    if (ImGuiFileDialog::Instance()->Display("TrainingDatasetFolderDialog",
                                             ImGuiWindowFlags_NoCollapse,
                                             minSize,
                                             maxSize)) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            copyToInputBuffer(training_dataset_path_, ImGuiFileDialog::Instance()->GetCurrentPath());
            syncDefaultOutputNameFromDataset();
            validateTrainingDatasetFromUi();
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (ImGuiFileDialog::Instance()->Display("TrainingOutputFolderDialog",
                                             ImGuiWindowFlags_NoCollapse,
                                             minSize,
                                             maxSize)) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            copyToInputBuffer(training_output_dir_, ImGuiFileDialog::Instance()->GetCurrentPath());
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (ImGuiFileDialog::Instance()->Display("TrainingSavePlyDialog",
                                             ImGuiWindowFlags_NoCollapse,
                                             minSize,
                                             maxSize)) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::filesystem::path path(ImGuiFileDialog::Instance()->GetFilePathName());
            copyToInputBuffer(training_output_dir_, path.parent_path().string());
            copyToInputBuffer(training_output_name_, path.filename().string());
            try {
                exportTrainingModelToPath(path);
            } catch (const std::exception& error) {
                setTrainingError(error.what());
            }
        }
        ImGuiFileDialog::Instance()->Close();
    }
}

void Application::applyTrainingConfigFromUi() {
    if (!training_dataset_loaded_) {
        throw std::runtime_error("Load a training dataset before starting training");
    }

    if (!training_.hasTrainableModel()) {
        TrainingInitializationConfig initConfig{};
        initConfig.randomGaussianCount = training_initial_gaussians_;
        initConfig.randomSeed = training_random_seed_;
        initConfig.initialOpacity = training_initial_opacity_;
        initConfig.sceneRadiusScale = training_scene_radius_scale_;
        training_.initializeModelFromDataset(initConfig);
        training_initialized_ = false;

        const std::string initSource = training_.usedRandomInitialization()
            ? "random fallback"
            : "COLMAP sparse points";
        setTrainingStatus("Training initialized with " +
                          std::to_string(training_.gaussianCount()) +
                          " gaussians from " + initSource);
    }

    TrainingOptimizerConfig optimizerConfig{};
    optimizerConfig.learningRate = training_learning_rate_;
    optimizerConfig.beta1 = training_adam_beta1_;
    optimizerConfig.beta2 = training_adam_beta2_;
    optimizerConfig.epsilon = training_adam_epsilon_;
    optimizerConfig.gradClip = training_grad_clip_;
    training_.setOptimizerConfig(optimizerConfig);

    applyTrainingDensificationConfigFromUi();
}

void Application::applyTrainingDensificationConfigFromUi() {
    TrainingDensificationConfig config{};
    config.enabled = training_densification_enabled_;
    config.densifyFromIteration = training_densify_from_iteration_;
    config.densifyUntilIteration = std::max(training_densify_until_iteration_,
                                            training_densify_from_iteration_);
    config.densificationInterval = std::max(training_densification_interval_, 1u);
    config.opacityResetInterval = std::max(training_opacity_reset_interval_, 1u);
    config.maxGaussianCount = std::max(training_max_gaussians_, 1u);
    config.splitChildren = std::clamp(training_split_children_, 2u, 8u);
    config.densifyGradThreshold = std::max(training_densify_grad_threshold_, 0.0f);
    config.minOpacity = std::clamp(training_min_opacity_, 0.0f, 0.99f);
    config.percentDense = std::clamp(training_percent_dense_, 0.0f, 1.0f);
    config.screenSizePruneThreshold = std::max(training_screen_size_prune_threshold_, 0.0f);
    training_.setDensificationConfig(config);
}

void Application::setTrainingStatus(const std::string& message) {
    training_status_ = message;
    LOG_INFO("{}", message);
}

void Application::setTrainingError(const std::string& message) {
    training_error_ = message;
    training_status_ = message;
    training_error_popup_pending_ = true;
    LOG_ERROR("{}", message);
}

std::filesystem::path Application::outputPlyPath() const {
    std::filesystem::path outputDir(inputBufferString(training_output_dir_));
    std::filesystem::path outputName(inputBufferString(training_output_name_));
    if (outputName.extension().empty()) {
        outputName += ".ply";
    }
    return outputDir / outputName;
}

void Application::syncDefaultOutputNameFromDataset() {
    std::filesystem::path datasetPath(inputBufferString(training_dataset_path_));
    std::string name = datasetPath.filename().string();
    if (name.empty()) {
        name = "trained";
    }
    copyToInputBuffer(training_output_name_, name + ".ply");
}

}
