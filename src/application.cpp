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
#include <iomanip>
#include <sstream>

#include <ImGuiFileDialog.h>
namespace vulkan3DGS {

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

std::string gpuDisplayLabel(const Device::PhysicalDeviceInfo& info) {
    constexpr double bytesPerGiB = 1024.0 * 1024.0 * 1024.0;
    std::ostringstream stream;
    stream << '[' << info.vulkanIndex << "] " << info.name << " (" << info.typeName
           << ", " << std::fixed << std::setprecision(1)
           << static_cast<double>(info.deviceLocalMemoryBytes) / bytesPerGiB << " GiB)";
    return stream.str();
}

} // namespace

Application::Application(const std::string& title,
                         int width,
                         int height,
                         RenderMode mode,
                         std::optional<std::string> gpuSelector)
    : current_mode_(mode) {
    // 设置日志级别为DEBUG（仅在Debug模式下）
#ifdef _DEBUG
    vulkan3DGS::Logger::get_instance().set_level(LogLevel::DEBUG_VULKAN_3DGS);
#endif
    
    LOG_INFO("Initializing Vulkan+3DGS Application");
    
    Context::initializeVulkanLoader();

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

    training_device_ = std::make_unique<Device>(vk::SurfaceKHR{},
                                                std::move(gpuSelector),
                                                DeviceRole::Training);
    training_device_->createDevice();
    syncTrainingGpuSelection();

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

    if (pending_training_gpu_selector_.has_value()) {
        applyPendingTrainingGpuSelection();
    }

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
            for (uint32_t i = 0; training_running_ && i < std::max(training_steps_per_frame_, 1u); ++i) {
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

void Application::setTrueCamera(const vulkan3DGS::Camera& camera) {
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
    training_device_.reset();

    if (vulkan_context_ && vulkan_context_->isInitialized()) {
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vulkan_context_->Device());
    }

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

void Application::drawTrainingGpuControl() {
    if (!training_device_) {
        ImGui::TextDisabled("Training GPU unavailable");
        return;
    }

    const auto& availableGpus = training_device_->getAvailablePhysicalDeviceInfos();
    const auto& selectedGpu = training_device_->getSelectedPhysicalDeviceInfo();
    if (availableGpus.empty()) {
        ImGui::TextDisabled("Training GPU unavailable");
        return;
    }

    training_gpu_ui_selection_ = std::min(training_gpu_ui_selection_, availableGpus.size() - 1);
    const std::string previewLabel = gpuDisplayLabel(availableGpus[training_gpu_ui_selection_]);
    if (training_running_) {
        ImGui::BeginDisabled();
    }
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::BeginCombo("Training GPU", previewLabel.c_str())) {
        for (size_t index = 0; index < availableGpus.size(); ++index) {
            const Device::PhysicalDeviceInfo& gpu = availableGpus[index];
            const std::string label = gpuDisplayLabel(gpu);
            const bool isSelected = index == training_gpu_ui_selection_;
            if (ImGui::Selectable(label.c_str(), isSelected)) {
                training_gpu_ui_selection_ = index;
                if (gpu.vulkanIndex != selectedGpu.vulkanIndex) {
                    pending_training_gpu_selector_ = gpu.uuid.empty()
                        ? std::to_string(gpu.vulkanIndex)
                        : gpu.uuid;
                }
            }
            if (isSelected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    if (training_running_) {
        ImGui::EndDisabled();
    }
    if (ImGui::IsItemHovered()) {
        const Device::PhysicalDeviceInfo& gpu = availableGpus[training_gpu_ui_selection_];
        ImGui::BeginTooltip();
        ImGui::Text("Vulkan %s", gpu.apiVersion.c_str());
        ImGui::Text("Driver %s", gpu.driverVersion.c_str());
        if (!gpu.uuid.empty()) {
            ImGui::Text("UUID %s", gpu.uuid.c_str());
        }
        ImGui::EndTooltip();
    }
}

void Application::syncTrainingGpuSelection() {
    if (!training_device_) {
        training_gpu_ui_selection_ = 0;
        return;
    }
    const auto& availableGpus = training_device_->getAvailablePhysicalDeviceInfos();
    const uint32_t selectedIndex =
        training_device_->getSelectedPhysicalDeviceInfo().vulkanIndex;
    const auto selected = std::find_if(
        availableGpus.begin(), availableGpus.end(), [selectedIndex](const Device::PhysicalDeviceInfo& info) {
            return info.vulkanIndex == selectedIndex;
        });
    training_gpu_ui_selection_ = selected == availableGpus.end()
        ? 0
        : static_cast<size_t>(std::distance(availableGpus.begin(), selected));
}

void Application::applyPendingTrainingGpuSelection() {
    const std::string selector = *pending_training_gpu_selector_;
    pending_training_gpu_selector_.reset();
    training_running_ = false;

    try {
        training_.cleanup();
        training_initialized_ = false;
        training_dataset_loaded_ = false;
        training_steps_done_ = 0;

        auto replacement = std::make_unique<Device>(vk::SurfaceKHR{}, selector, DeviceRole::Training);
        replacement->createDevice();
        training_device_ = std::move(replacement);
        syncTrainingGpuSelection();

        const auto& gpu = training_device_->getSelectedPhysicalDeviceInfo();
        setTrainingStatus("Training GPU selected: [" + std::to_string(gpu.vulkanIndex) +
                          "] " + gpu.name + ". Dataset must be loaded again.");
    } catch (const std::exception& error) {
        syncTrainingGpuSelection();
        setTrainingError(std::string("Failed to switch training GPU: ") + error.what());
    }
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

    drawTrainingGpuControl();
    ImGui::Separator();

    if (training_running_) {
        ImGui::BeginDisabled();
    }

    if (ImGui::InputText("Dataset Folder", training_dataset_path_.data(), training_dataset_path_.size())) {
        training_dataset_valid_ = false;
        training_dataset_loaded_ = false;
        training_initialized_ = false;
        training_running_ = false;
        training_frame_count_ = 0;
        training_width_ = 0;
        training_height_ = 0;
        training_status_ = "Dataset path changed. Validate or start training to load it.";
    }
    ImGui::SameLine();
    if (ImGui::Button("Browse##DatasetFolder")) {
        chooseTrainingDatasetFolderFromUi();
    }
    const int previousDownscale = training_downscale_;
    ImGui::InputInt("Downscale", &training_downscale_);
    training_downscale_ = std::clamp(training_downscale_, 1, 16);
    if (training_downscale_ != previousDownscale) {
        training_dataset_valid_ = false;
        training_dataset_loaded_ = false;
        training_initialized_ = false;
        training_running_ = false;
        training_frame_count_ = 0;
        training_width_ = 0;
        training_height_ = 0;
        training_status_ = "Dataset downscale changed. Validate or start training to load it.";
    }

    const bool canEditTrainingSetup = !training_running_;

    if (ImGui::CollapsingHeader("Initialization", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!canEditTrainingSetup) {
            ImGui::BeginDisabled();
        }
        ImGui::InputScalar("Fallback Gaussians", ImGuiDataType_U32, &training_initial_gaussians_);
        training_initial_gaussians_ = std::clamp(training_initial_gaussians_, 1u, 1000000u);
        ImGui::InputScalar("Random Seed", ImGuiDataType_U32, &training_random_seed_);
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
        if (!canEditTrainingSetup) {
            ImGui::BeginDisabled();
        }
        const char* trainingModes[] = {
            "Sequential",
            "3DGS Random"
        };
        ImGui::Combo("Mode", &training_mode_, trainingModes, IM_ARRAYSIZE(trainingModes));
        training_mode_ = std::clamp(training_mode_, 0, 1);
        const char* pixelBackwardModes[] = {
            "Auto (Adaptive)",
            "Direct",
            "Workgroup Shared",
            "Subgroup"
        };
        if (ImGui::Combo("Pixel Backward", &training_pixel_to_2dgs_mode_,
                         pixelBackwardModes, IM_ARRAYSIZE(pixelBackwardModes))) {
            training_pixel_to_2dgs_mode_ = std::clamp(training_pixel_to_2dgs_mode_, 0, 3);
            training_.setPixelTo2DGSMode(
                static_cast<TrainingPixelTo2DGSMode>(training_pixel_to_2dgs_mode_));
        }
        if (ImGui::SliderFloat("Auto Min Subgroup Utilization",
                               &training_pixel_to_2dgs_min_subgroup_utilization_,
                               0.0f,
                               1.0f,
                               "%.2f")) {
            training_.setPixelTo2DGSMinSubgroupUtilization(
                training_pixel_to_2dgs_min_subgroup_utilization_);
        }
        if (training_.isRendererInitialized()) {
            const int activePixelBackwardMode = static_cast<int>(training_.activePixelTo2DGSMode());
            ImGui::Text("Active Pixel Backward %s", pixelBackwardModes[activePixelBackwardMode]);
        }
        const char* forwardCompositeModes[] = {
            "Direct",
            "Workgroup Shared"
        };
        if (ImGui::Combo("Forward Composite", &training_forward_composite_mode_,
                         forwardCompositeModes, IM_ARRAYSIZE(forwardCompositeModes))) {
            training_forward_composite_mode_ = std::clamp(training_forward_composite_mode_, 0, 1);
            training_.setForwardCompositeMode(
                static_cast<TrainingForwardCompositeMode>(training_forward_composite_mode_));
        }
        if (training_mode_ == 1) {
            training_total_iterations_ = 30000;
            ImGui::BeginDisabled();
            ImGui::InputScalar("Total Iterations", ImGuiDataType_U32, &training_total_iterations_);
            ImGui::EndDisabled();
        }
        if (!canEditTrainingSetup) {
            ImGui::EndDisabled();
        }
        ImGui::InputScalar("Steps/Frame", ImGuiDataType_U32, &training_steps_per_frame_);
        training_steps_per_frame_ = std::max(training_steps_per_frame_, 1u);
        if (ImGui::InputScalar("Validation Interval",
                               ImGuiDataType_U32,
                               &training_validation_interval_)) {
            training_.setValidationInterval(training_validation_interval_);
        }
        if (!canEditTrainingSetup) {
            ImGui::BeginDisabled();
        }
        ImGui::InputFloat("Position LR", &training_position_lr_, 0.00001f, 0.0001f, "%.6g");
        training_position_lr_ = std::max(training_position_lr_, 0.0f);
        ImGui::InputFloat("Position LR Final", &training_position_lr_final_, 0.000001f, 0.00001f, "%.6g");
        training_position_lr_final_ = std::max(training_position_lr_final_, 0.0f);
        ImGui::InputFloat("Position LR Delay Mult", &training_position_lr_delay_mult_, 0.01f, 0.05f, "%.6g");
        training_position_lr_delay_mult_ = std::clamp(training_position_lr_delay_mult_, 0.0f, 1.0f);
        ImGui::InputFloat("Position LR Delay Steps", &training_position_lr_delay_steps_, 100.0f, 1000.0f, "%.0f");
        training_position_lr_delay_steps_ = std::max(training_position_lr_delay_steps_, 0.0f);
        ImGui::InputFloat("Position LR Steps", &training_position_lr_max_steps_, 1000.0f, 5000.0f, "%.0f");
        training_position_lr_max_steps_ = std::max(training_position_lr_max_steps_, 1.0f);
        ImGui::InputFloat("Feature LR", &training_feature_lr_, 0.0001f, 0.001f, "%.6g");
        training_feature_lr_ = std::max(training_feature_lr_, 0.0f);
        ImGui::InputFloat("Feature Rest LR", &training_feature_rest_lr_, 0.00001f, 0.0001f, "%.6g");
        training_feature_rest_lr_ = std::max(training_feature_rest_lr_, 0.0f);
        ImGui::InputFloat("Opacity LR", &training_opacity_lr_, 0.001f, 0.01f, "%.6g");
        training_opacity_lr_ = std::max(training_opacity_lr_, 0.0f);
        ImGui::InputFloat("Scale LR", &training_scale_lr_, 0.0001f, 0.001f, "%.6g");
        training_scale_lr_ = std::max(training_scale_lr_, 0.0f);
        ImGui::InputFloat("Rotation LR", &training_rotation_lr_, 0.0001f, 0.001f, "%.6g");
        training_rotation_lr_ = std::max(training_rotation_lr_, 0.0f);
        ImGui::InputScalar("Max SH Degree", ImGuiDataType_U32, &training_max_sh_degree_);
        training_max_sh_degree_ = std::min(training_max_sh_degree_, 3u);
        ImGui::InputScalar("SH Degree Interval", ImGuiDataType_U32, &training_sh_degree_interval_);
        training_sh_degree_interval_ = std::max(training_sh_degree_interval_, 1u);
        ImGui::InputFloat("Adam Beta1", &training_adam_beta1_, 0.01f, 0.05f, "%.6g");
        training_adam_beta1_ = std::clamp(training_adam_beta1_, 0.0f, 0.999999f);
        ImGui::InputFloat("Adam Beta2", &training_adam_beta2_, 0.001f, 0.01f, "%.6g");
        training_adam_beta2_ = std::clamp(training_adam_beta2_, 0.0f, 0.999999f);
        ImGui::InputFloat("Adam Epsilon", &training_adam_epsilon_, 1e-15f, 1e-12f, "%.3e");
        training_adam_epsilon_ = std::max(training_adam_epsilon_, 1e-15f);
        ImGui::InputFloat("Gradient Clip", &training_grad_clip_, 10.0f, 100.0f, "%.6g");
        training_grad_clip_ = std::max(training_grad_clip_, 0.0f);
        ImGui::InputFloat("DSSIM Weight", &training_loss_dssim_weight_, 0.01f, 0.05f, "%.6g");
        training_loss_dssim_weight_ = std::clamp(training_loss_dssim_weight_, 0.0f, 1.0f);
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
        ImGui::InputFloat("World Prune Size", &training_world_size_prune_threshold_, 0.001f, 0.01f, "%.6g");
        training_world_size_prune_threshold_ = std::max(training_world_size_prune_threshold_, 0.0f);
        if (!canEditTrainingSetup) {
            ImGui::EndDisabled();
        }
    }

    const bool hasDatasetPath = !inputBufferString(training_dataset_path_).empty();
    bool canTrain = hasDatasetPath;
    if (!canTrain) {
        ImGui::BeginDisabled();
    }

    const bool benchmarkActive = training_.isFixedWorkloadBenchmarkActive();
    const char* primaryTrainingButtonLabel = benchmarkActive
        ? "Stop Benchmark"
        : (training_running_ ? "Pause Training" : "Start Training");
    if (ImGui::Button(primaryTrainingButtonLabel)) {
        if (training_running_) {
            if (benchmarkActive) {
                training_.stopFixedWorkloadBenchmark();
            }
            training_running_ = false;
            setTrainingStatus(benchmarkActive ? "Fixed workload benchmark stopped."
                                              : "Training paused.");
        } else {
            try {
                if (!training_dataset_loaded_) {
                    loadTrainingDatasetFromUi();
                    if (!training_dataset_loaded_) {
                        return;
                    }
                }
                applyTrainingConfigFromUi();
                initializeTrainingIfNeeded();
                training_running_ = true;
                setTrainingStatus("Training started.");
            } catch (const std::exception& error) {
                training_running_ = false;
                setTrainingError(error.what());
            }
        }
    }

    if (!training_running_) {
        ImGui::SameLine();
        if (ImGui::Button("Start Fixed Benchmark")) {
            try {
                if (!training_dataset_loaded_) {
                    loadTrainingDatasetFromUi();
                    if (!training_dataset_loaded_) {
                        return;
                    }
                }
                applyTrainingConfigFromUi();
                initializeTrainingIfNeeded();
                TrainingFixedBenchmarkConfig benchmarkConfig{};
                benchmarkConfig.frameIndex = training_benchmark_frame_;
                benchmarkConfig.warmupSteps = training_benchmark_warmup_steps_;
                benchmarkConfig.measuredSteps = training_benchmark_measured_steps_;
                training_.startFixedWorkloadBenchmark(benchmarkConfig);
                training_running_ = true;
                setTrainingStatus("Fixed workload benchmark started.");
            } catch (const std::exception& error) {
                training_running_ = false;
                setTrainingError(error.what());
            }
        }
    }

    if (!canTrain) {
        ImGui::EndDisabled();
    }

    ImGui::Text("Dataset valid %s, loaded %s",
                training_dataset_valid_ ? "yes" : "no",
                training_dataset_loaded_ ? "yes" : "no");
    ImGui::Text("Training initialized %s, running %s",
                training_initialized_ ? "yes" : "no",
                training_running_ ? "yes" : "no");
    if (!hasDatasetPath) {
        ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.25f, 1.0f),
                           "Choose a dataset folder before starting training.");
    } else if (!training_dataset_loaded_) {
        ImGui::TextWrapped("Start Training will validate and load the dataset first.");
    }

    if (training_.totalIterations() > 0) {
        ImGui::Text("Iterations %u / %u", training_.trainingIteration(), training_.totalIterations());
    } else {
        ImGui::Text("Iterations %u", training_.trainingIteration());
    }
    ImGui::Text("Steps %llu", static_cast<unsigned long long>(training_steps_done_));
    ImGui::Text("Gaussians %u", training_.gaussianCount());
    if (!training_running_) {
        ImGui::InputScalar("Benchmark Frame", ImGuiDataType_U32, &training_benchmark_frame_);
        if (training_frame_count_ > 0u) {
            training_benchmark_frame_ = std::min(training_benchmark_frame_, training_frame_count_ - 1u);
        } else {
            training_benchmark_frame_ = 0u;
        }
        ImGui::InputScalar("Benchmark Warmup", ImGuiDataType_U32,
                           &training_benchmark_warmup_steps_);
        ImGui::InputScalar("Benchmark Measured", ImGuiDataType_U32,
                           &training_benchmark_measured_steps_);
        training_benchmark_measured_steps_ = std::max(training_benchmark_measured_steps_, 1u);
    }
    const auto& benchmark = training_.fixedWorkloadBenchmarkStats();
    if (benchmark.active || benchmark.complete || benchmark.completedWarmupSteps > 0u ||
        benchmark.completedMeasuredSteps > 0u) {
        ImGui::Text("Fixed benchmark frame %u, warmup %u/%u, measured %u/%u, %s",
                    benchmark.frameIndex,
                    benchmark.completedWarmupSteps,
                    benchmark.warmupSteps,
                    benchmark.completedMeasuredSteps,
                    benchmark.measuredSteps,
                    benchmark.active ? "running" : (benchmark.complete ? "complete" : "stopped"));
    }
    const auto& densification = training_.densificationStats();
    const uint32_t densificationIteration = training_.densificationStatsIteration();
    if (densificationIteration > 0) {
        ImGui::Text("Densify/prune iter %u, output %u", densificationIteration, densification.outputCount);
        ImGui::Text("Densify kept %u, cloned %u, split %u, pruned %u",
                    densification.keptSources,
                    densification.cloneSources,
                    densification.splitSources,
                    densification.prunedSources);
        ImGui::Text("Prune hits opacity %u, screen %u, world %u",
                    densification.pruneOpacityHits,
                    densification.pruneScreenHits,
                    densification.pruneWorldHits);
    } else {
        ImGui::Text("Densify/prune not run yet");
    }
    const auto& validation = training_.validationStats();
    ImGui::Text("Loss %.6g", validation.meanLoss);
    ImGui::Text("Render alpha %.6g", validation.meanRenderedAlpha);
    ImGui::Text("Tile items %u", validation.tileItemCount);
    ImGui::Text("Backward candidates avg %.2f, contributors avg %.2f, max candidates %u",
                validation.meanProcessedCandidatesPerPixel,
                validation.meanContributorsPerPixel,
                validation.maxProcessedCandidatesPerPixel);
    const auto& candidateProfile = training_.candidateProfileStats();
    if (candidateProfile.sampleCount > 0u) {
        const float emptyCandidatePercent = candidateProfile.meanProcessedCandidatesPerPixel > 0.0f
            ? 100.0f * (1.0f - candidateProfile.meanContributorsPerPixel /
                                  candidateProfile.meanProcessedCandidatesPerPixel)
            : 0.0f;
        ImGui::Text("Validation history n=%u: candidates %.2f, contributors %.2f, empty %.2f%%, max %u",
                    candidateProfile.sampleCount,
                    candidateProfile.meanProcessedCandidatesPerPixel,
                    candidateProfile.meanContributorsPerPixel,
                    emptyCandidatePercent,
                    candidateProfile.maxProcessedCandidatesPerPixel);
        ImGui::Text("Processed pixels 0/1-32/33-64/65-128: %.1f%% / %.1f%% / %.1f%% / %.1f%%",
                    100.0f * candidateProfile.meanPixelFractionByProcessedBucket[0],
                    100.0f * candidateProfile.meanPixelFractionByProcessedBucket[1],
                    100.0f * candidateProfile.meanPixelFractionByProcessedBucket[2],
                    100.0f * candidateProfile.meanPixelFractionByProcessedBucket[3]);
        ImGui::Text("Processed pixels 129-256/257-512/513-1024/>1024: %.1f%% / %.1f%% / %.1f%% / %.1f%%",
                    100.0f * candidateProfile.meanPixelFractionByProcessedBucket[4],
                    100.0f * candidateProfile.meanPixelFractionByProcessedBucket[5],
                    100.0f * candidateProfile.meanPixelFractionByProcessedBucket[6],
                    100.0f * candidateProfile.meanPixelFractionByProcessedBucket[7]);
    }
    if (!validation.valid && training_steps_done_ > 0) {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.25f, 1.0f),
                           "Validation issue: invalid loss %u, invalid pixels %u, non-finite gaussians %u",
                           validation.invalidLossCount,
                           validation.invalidRenderedPixelCount,
                           validation.nonFiniteGaussianCount);
        if (validation.nonFiniteGaussianCount > 0u) {
            ImGui::Text("First non-finite %u; pos %u, opacity %u, raw/active scale %u/%u, rotation %u, SH %u",
                        validation.firstNonFiniteGaussianIndex,
                        validation.nonFinitePositionCount,
                        validation.nonFiniteOpacityCount,
                        validation.nonFiniteRawScaleCount,
                        validation.nonFiniteActivatedScaleCount,
                        validation.nonFiniteRotationCount,
                        validation.nonFiniteSHCount);
        }
    }
    if (ImGui::CollapsingHeader("Training Profiling", ImGuiTreeNodeFlags_DefaultOpen)) {
        static constexpr const char* cpuStageNames[] = {
            "Frame upload",
            "Image request/wait",
            "Target RGBA8 upload",
            "Prepare submit/wait",
            "Tile count readback",
            "Tile buffer resize",
            "Main command record",
            "Main submit/wait",
            "Validation",
            "Densify adopt",
            "Total step",
        };
        static constexpr const char* gpuStageNames[] = {
            "Gaussian projection",
            "Tile coverage count",
            "Tile prefix",
            "Tile emit",
            "Tile sort/ranges",
            "Composite",
            "Loss",
            "Backward clear",
            "Loss to pixel",
            "Pixel to 2DGS",
            "2DGS to 3DGS",
            "Optimizer",
            "Validation",
            "Densify/prune",
        };
        static_assert(sizeof(cpuStageNames) / sizeof(cpuStageNames[0]) == kTrainingCpuProfileStageCount);
        static_assert(sizeof(gpuStageNames) / sizeof(gpuStageNames[0]) == kTrainingGpuProfileStageCount);
        const auto& profiling = training_.profilingStats();
        ImGui::Text("CPU last / average (ms)");
        for (size_t i = 0; i < kTrainingCpuProfileStageCount; ++i) {
            const auto& timing = profiling.cpu[i];
            ImGui::Text("%s %.2f / %.2f (n=%u)",
                        cpuStageNames[i], timing.lastMs, timing.averageMs, timing.sampleCount);
        }
        if (profiling.gpuTimestampsAvailable) {
            ImGui::Text("GPU last / average (ms)");
            for (size_t i = 0; i < kTrainingGpuProfileStageCount; ++i) {
                const auto& timing = profiling.gpu[i];
                ImGui::Text("%s %.2f / %.2f (n=%u)",
                            gpuStageNames[i], timing.lastMs, timing.averageMs, timing.sampleCount);
            }
        } else {
            ImGui::TextDisabled("GPU timestamps unavailable on the compute queue");
        }
    }
    if (ImGui::CollapsingHeader("Image Cache", ImGuiTreeNodeFlags_DefaultOpen)) {
        const ImageStreamerStats cacheStats = training_.imageCacheStats();
        constexpr double bytesPerMiB = 1024.0 * 1024.0;
        ImGui::Text("Host %.1f / %.1f MiB, %llu images",
                    static_cast<double>(cacheStats.hostCachedBytes) / bytesPerMiB,
                    static_cast<double>(cacheStats.hostBudgetBytes) / bytesPerMiB,
                    static_cast<unsigned long long>(cacheStats.hostCachedImages));
        ImGui::Text("Host hit %llu, miss %llu, wait %llu, evict %llu",
                    static_cast<unsigned long long>(cacheStats.hostHits),
                    static_cast<unsigned long long>(cacheStats.hostMisses),
                    static_cast<unsigned long long>(cacheStats.hostWaits),
                    static_cast<unsigned long long>(cacheStats.hostEvictions));
        ImGui::Text("Prefetch %llu, KTX chunk hit %llu, miss %llu, write %llu",
                    static_cast<unsigned long long>(cacheStats.prefetchRequests),
                    static_cast<unsigned long long>(cacheStats.disk.hits),
                    static_cast<unsigned long long>(cacheStats.disk.misses),
                    static_cast<unsigned long long>(cacheStats.disk.writes));
        ImGui::Text("KTX active %.1f MiB / %llu chunks, history %.1f MiB / %llu chunks",
                    static_cast<double>(cacheStats.disk.activeBytes) / bytesPerMiB,
                    static_cast<unsigned long long>(cacheStats.disk.activeFiles),
                    static_cast<double>(cacheStats.disk.historicalBytes) / bytesPerMiB,
                    static_cast<unsigned long long>(cacheStats.disk.historicalFiles));
        ImGui::Text("KTX total %.1f MiB / %llu chunks, recover %llu, evict %llu",
                    static_cast<double>(cacheStats.disk.cachedBytes) / bytesPerMiB,
                    static_cast<unsigned long long>(cacheStats.disk.cachedFiles),
                    static_cast<unsigned long long>(cacheStats.disk.recoveries),
                    static_cast<unsigned long long>(cacheStats.disk.evictions));
        const DeviceImageCacheStats deviceCacheStats = training_.deviceImageCacheStats();
        static constexpr const char* deviceCacheModes[] = {"Streaming", "Partial", "Full"};
        const uint32_t deviceMode = std::min(static_cast<uint32_t>(deviceCacheStats.mode), 2u);
        ImGui::Text("GPU %s %.1f / %.1f MiB, %u resident / %u slots / %u total",
                    deviceCacheModes[deviceMode],
                    static_cast<double>(deviceCacheStats.allocatedBytes) / bytesPerMiB,
                    static_cast<double>(deviceCacheStats.budgetBytes) / bytesPerMiB,
                    deviceCacheStats.residentImages,
                    deviceCacheStats.slotCount,
                    deviceCacheStats.totalImages);
        ImGui::Text("GPU hit %llu, miss %llu, upload %llu, evict %llu",
                    static_cast<unsigned long long>(deviceCacheStats.hits),
                    static_cast<unsigned long long>(deviceCacheStats.misses),
                    static_cast<unsigned long long>(deviceCacheStats.uploads),
                    static_cast<unsigned long long>(deviceCacheStats.evictions));
        ImGui::Text("Upload %s, staging %.1f MiB, pending %u, ring waits %llu",
                    deviceCacheStats.asynchronousUploads ? "async" : "sync",
                    static_cast<double>(deviceCacheStats.stagingBytes) / bytesPerMiB,
                    deviceCacheStats.pendingUploads,
                    static_cast<unsigned long long>(deviceCacheStats.uploadWaits));
        if (deviceCacheStats.heapBudgetBytes > 0) {
            ImGui::Text("VRAM heap %.1f / %.1f MiB%s, cache resizes %llu",
                        static_cast<double>(deviceCacheStats.heapUsageBytes) / bytesPerMiB,
                        static_cast<double>(deviceCacheStats.heapBudgetBytes) / bytesPerMiB,
                        deviceCacheStats.memoryBudgetAvailable ? " (budget)" : " (size only)",
                        static_cast<unsigned long long>(deviceCacheStats.budgetResizes));
        }
    }
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
            training_dataset_loaded_ = false;
            training_initialized_ = false;
            training_running_ = false;
            setTrainingError(validation.message);
            return;
        }
        syncDefaultOutputNameFromDataset();
        setTrainingStatus("Dataset valid: " + std::to_string(training_frame_count_) +
                          " frames, " + std::to_string(training_width_) +
                          "x" + std::to_string(training_height_));
    } catch (const std::exception& error) {
        training_dataset_valid_ = false;
        training_dataset_loaded_ = false;
        training_initialized_ = false;
        training_running_ = false;
        training_frame_count_ = 0;
        training_width_ = 0;
        training_height_ = 0;
        setTrainingError(error.what());
    }
}

void Application::loadTrainingDatasetFromUi() {
    training_running_ = false;
    training_dataset_loaded_ = false;
    training_initialized_ = false;

    validateTrainingDatasetFromUi();
    if (!training_dataset_valid_) {
        return;
    }

    try {
        if (!training_device_) {
            throw std::runtime_error("Training GPU is not initialized");
        }
        auto& device = *training_device_;
        const uint32_t computeFamily = device.getQueueFamilyIndices().computeIndex.value();
        auto transferFamily = device.getQueueFamilyIndices().transferIndex.value_or(
            computeFamily);
        if (training_.isInitialized()) {
            training_.cleanup();
            training_initialized_ = false;
        }
        training_.initialize(device.getDevice(),
                             device.getPhysicalDevice(),
                             device.getTransferQueue(),
                             transferFamily,
                             computeFamily);
        training_.loadMipNeRF360Dataset(inputBufferString(training_dataset_path_),
                                        static_cast<uint32_t>(training_downscale_));
        training_initialized_ = false;
        training_dataset_loaded_ = true;
        training_steps_done_ = 0;
        setTrainingStatus("Dataset loaded: " + std::to_string(training_frame_count_) +
                          " frames, " + std::to_string(training_width_) +
                          "x" + std::to_string(training_height_) +
                          ". Press Start Training to initialize and train.");
    } catch (const std::exception& error) {
        training_dataset_loaded_ = false;
        training_initialized_ = false;
        training_running_ = false;
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

    if (!training_device_) {
        throw std::runtime_error("Training GPU is not initialized");
    }
    auto& device = *training_device_;
    auto computeFamily = device.getQueueFamilyIndices().computeIndex.value();
    auto transferFamily = device.getQueueFamilyIndices().transferIndex.value_or(
        computeFamily);

    training_.initialize(device.getDevice(),
                         device.getPhysicalDevice(),
                         device.getTransferQueue(),
                         transferFamily,
                         computeFamily);
    training_.initializeTrainingRenderers(device.getDevice(),
                                          device.getPhysicalDevice(),
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
    if (!training_.isFixedWorkloadBenchmarkActive() && training_.isTrainingComplete()) {
        training_running_ = false;
        setTrainingStatus("Training completed at " +
                          std::to_string(training_.trainingIteration()) +
                          " iterations.");
        return;
    }
    training_.trainStep();
    ++training_steps_done_;
    const auto& benchmark = training_.fixedWorkloadBenchmarkStats();
    if (!benchmark.active && benchmark.complete) {
        training_running_ = false;
        setTrainingStatus("Fixed workload benchmark completed: " +
                          std::to_string(benchmark.measuredSteps) +
                          " measured steps.");
        return;
    }
    if (training_.isTrainingComplete()) {
        training_running_ = false;
        setTrainingStatus("Training completed at " +
                          std::to_string(training_.trainingIteration()) +
                          " iterations.");
    }
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

    TrainingScheduleConfig scheduleConfig{};
    scheduleConfig.imageSelectionMode = training_mode_ == 1
        ? TrainingImageSelectionMode::Random
        : TrainingImageSelectionMode::Sequential;
    scheduleConfig.totalIterations = training_mode_ == 1 ? 30000u : 0u;
    scheduleConfig.randomSeed = training_random_seed_;
    training_total_iterations_ = scheduleConfig.totalIterations > 0 ? scheduleConfig.totalIterations : 30000u;
    training_.setScheduleConfig(scheduleConfig);
    training_.setPixelTo2DGSMode(
        static_cast<TrainingPixelTo2DGSMode>(std::clamp(training_pixel_to_2dgs_mode_, 0, 3)));
    training_.setPixelTo2DGSMinSubgroupUtilization(
        training_pixel_to_2dgs_min_subgroup_utilization_);
    training_.setForwardCompositeMode(
        static_cast<TrainingForwardCompositeMode>(
            std::clamp(training_forward_composite_mode_, 0, 1)));
    training_.setValidationInterval(training_validation_interval_);

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
    optimizerConfig.positionLearningRate = training_position_lr_;
    optimizerConfig.positionLearningRateFinal = training_position_lr_final_;
    optimizerConfig.positionLearningRateDelayMult = training_position_lr_delay_mult_;
    optimizerConfig.positionLearningRateDelaySteps = training_position_lr_delay_steps_;
    optimizerConfig.positionLearningRateMaxSteps = training_position_lr_max_steps_;
    optimizerConfig.featureLearningRate = training_feature_lr_;
    optimizerConfig.featureRestLearningRate = training_feature_rest_lr_;
    optimizerConfig.opacityLearningRate = training_opacity_lr_;
    optimizerConfig.scaleLearningRate = training_scale_lr_;
    optimizerConfig.rotationLearningRate = training_rotation_lr_;
    optimizerConfig.beta1 = training_adam_beta1_;
    optimizerConfig.beta2 = training_adam_beta2_;
    optimizerConfig.epsilon = training_adam_epsilon_;
    optimizerConfig.gradClip = training_grad_clip_;
    optimizerConfig.lossDssimWeight = training_loss_dssim_weight_;
    optimizerConfig.maxSHDegree = std::min(training_max_sh_degree_, 3u);
    optimizerConfig.shDegreeInterval = std::max(training_sh_degree_interval_, 1u);
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
    config.worldSizePruneThreshold = std::max(training_world_size_prune_threshold_, 0.0f);
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
