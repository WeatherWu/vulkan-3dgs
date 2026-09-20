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
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>

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

std::string formatTrainingDuration(double seconds) {
    const uint64_t totalMilliseconds = static_cast<uint64_t>(
        std::max(seconds, 0.0) * 1000.0);
    const uint64_t milliseconds = totalMilliseconds % 1000u;
    const uint64_t totalSeconds = totalMilliseconds / 1000u;
    const uint64_t secondsPart = totalSeconds % 60u;
    const uint64_t totalMinutes = totalSeconds / 60u;
    const uint64_t minutesPart = totalMinutes % 60u;
    const uint64_t hours = totalMinutes / 60u;

    std::ostringstream stream;
    stream << std::setfill('0') << std::setw(2) << hours << ':'
           << std::setw(2) << minutesPart << ':'
           << std::setw(2) << secondsPart << '.'
           << std::setw(3) << milliseconds;
    return stream.str();
}

std::optional<std::filesystem::path> environmentDirectory(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    size_t length = 0;
    if (_dupenv_s(&value, &length, name) != 0 || !value || length <= 1u) {
        std::free(value);
        return std::nullopt;
    }
    std::filesystem::path path(value);
    std::free(value);
    return path;
#else
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return std::nullopt;
    }
    return std::filesystem::path(value);
#endif
}

std::filesystem::path trainingSettingsPath() {
#ifdef _WIN32
    if (auto appData = environmentDirectory("APPDATA")) {
        return *appData / "vulkan-3dgs" / "training-settings.cfg";
    }
#else
    if (auto configHome = environmentDirectory("XDG_CONFIG_HOME")) {
        return *configHome / "vulkan-3dgs" / "training-settings.cfg";
    }
    if (auto home = environmentDirectory("HOME")) {
        return *home / ".config" / "vulkan-3dgs" / "training-settings.cfg";
    }
#endif
    return std::filesystem::current_path() / ".vulkan-3dgs-training-settings.cfg";
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

    copyToInputBuffer(training_dataset_path_, "data/mipnerf360/bicycle");
    copyToInputBuffer(training_output_dir_, "output");
    copyToInputBuffer(training_output_name_, "bicycle.ply");
    loadTrainingSettings();

    const bool usePersistedGpu = !gpuSelector.has_value() &&
                                 !persisted_training_gpu_selector_.empty();
    if (usePersistedGpu) {
        gpuSelector = persisted_training_gpu_selector_;
    }
    
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
    applyRenderProfile();
    renderer_->setImGuiDrawCallback([this]() {
        drawImGuiControls();
    });

    try {
        training_device_ = std::make_unique<Device>(vk::SurfaceKHR{},
                                                    std::move(gpuSelector),
                                                    DeviceRole::Training);
        training_device_->createDevice();
    } catch (const std::exception& error) {
        if (!usePersistedGpu) {
            throw;
        }
        LOG_WARN("Saved training GPU is unavailable ({}); selecting the default GPU",
                 error.what());
        training_device_ = std::make_unique<Device>(vk::SurfaceKHR{},
                                                    std::nullopt,
                                                    DeviceRole::Training);
        training_device_->createDevice();
    }
    syncTrainingGpuSelection();

    window_->set_resize_callback([this](int width, int height) {
        if (width <= 0 || height <= 0) {
            return;
        }

        if (renderer_) {
            renderer_->onResize(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
        }

        if (has_true_camera_) {
            updateCameraMatrices(width, height);
        }
    });

    window_->set_drop_callback([this](const std::vector<std::string>& paths) {
        handleDroppedFiles(paths);
    });

    window_->set_scroll_callback([this](double xoffset, double yoffset) {
        handleScroll(xoffset, yoffset);
    });

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

    consumeTrainingWorkerResult();

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
    bool renderUi = true;
    if (training_async_active_ && training_pure_active_) {
        constexpr double pureUiIntervalSeconds = 0.1;
        renderUi = currentTime - training_pure_last_ui_render_time_ >=
                   pureUiIntervalSeconds;
    }
    if (renderUi) {
        training_pure_last_ui_render_time_ = currentTime;
        render();
    } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
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
            gsRenderer->setRenderProfile(gaussian_render_profile_);
            gsRenderer->setSHBands(gaussian_sh_bands_);
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
    updateSuperSplatCamera(delta_time);

    if (training_running_ && !training_async_active_) {
        try {
            const uint32_t stepBudget = std::max(training_steps_per_frame_, 1u);
            for (uint32_t i = 0; training_running_ && i < stepBudget; ++i) {
                runTrainingStepFromUi();
            }
        } catch (const std::exception& error) {
            training_running_ = false;
            training_pure_active_ = false;
            stopTrainingTimer();
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
                updateCameraMatrices(framebufferWidth, framebufferHeight);
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
    resetCameraFromModel();
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
    resetCameraFromModel();
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
    const glm::vec3 focalPoint = current_model_
        ? current_model_->get_focus_center()
        : camera_.get_position() + camera_.get_front() * 5.0f;
    const float sceneRadius = current_model_
        ? current_model_->get_focus_radius()
        : 2.0f;
    supersplat_camera_controller_.syncFromCamera(
        camera_, focalPoint, sceneRadius);
}

void Application::cleanup() {
    stopTrainingWorkerAndJoin();
    saveTrainingSettings();
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

void Application::resetCameraFromModel() {
    int framebufferWidth = 1;
    int framebufferHeight = 1;
    if (window_) {
        glfwGetFramebufferSize(
            window_->get_handle(), &framebufferWidth, &framebufferHeight);
    }
    if (!current_model_ || current_model_->isEmpty()) {
        supersplat_camera_controller_.reset(
            glm::vec3(0.0f), 1.0f,
            static_cast<float>(std::max(framebufferWidth, 1)),
            static_cast<float>(std::max(framebufferHeight, 1)));
        return;
    }
    supersplat_camera_controller_.reset(
        current_model_->get_focus_center(), current_model_->get_focus_radius(),
        static_cast<float>(std::max(framebufferWidth, 1)),
        static_cast<float>(std::max(framebufferHeight, 1)));
}

void Application::focusCameraOnModel() {
    if (!current_model_ || current_model_->isEmpty()) {
        return;
    }
    int framebufferWidth = 1;
    int framebufferHeight = 1;
    if (window_) {
        glfwGetFramebufferSize(
            window_->get_handle(), &framebufferWidth, &framebufferHeight);
    }
    supersplat_camera_controller_.focus(
        current_model_->get_focus_center(), current_model_->get_focus_radius(),
        static_cast<float>(std::max(framebufferWidth, 1)),
        static_cast<float>(std::max(framebufferHeight, 1)));
}

void Application::updateSuperSplatCamera(float deltaTime) {
    if (!current_model_ || current_model_->isEmpty()) {
        return;
    }

    updateSuperSplatInput(deltaTime);
    supersplat_camera_controller_.update(deltaTime, camera_);
    has_true_camera_ = true;

    if (!window_) {
        return;
    }
    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetFramebufferSize(window_->get_handle(), &framebufferWidth, &framebufferHeight);
    if (framebufferWidth > 0 && framebufferHeight > 0) {
        updateCameraMatrices(framebufferWidth, framebufferHeight);
    }
}

void Application::updateSuperSplatInput(float deltaTime) {
    if (!window_) {
        return;
    }

    GLFWwindow* handle = window_->get_handle();
    const ImGuiIO* io = ImGui::GetCurrentContext() ? &ImGui::GetIO() : nullptr;
    const bool captureMouse = io && io->WantCaptureMouse;
    const bool captureKeyboard = io && io->WantCaptureKeyboard;

    double mouseX = 0.0;
    double mouseY = 0.0;
    glfwGetCursorPos(handle, &mouseX, &mouseY);
    const bool leftDown =
        glfwGetMouseButton(handle, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    const bool rightDown =
        glfwGetMouseButton(handle, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    const int activeButton = rightDown
        ? GLFW_MOUSE_BUTTON_RIGHT
        : (leftDown ? GLFW_MOUSE_BUTTON_LEFT : -1);

    if (captureMouse || activeButton < 0) {
        supersplat_dragging_ = false;
        supersplat_drag_button_ = -1;
        last_mouse_x_ = mouseX;
        last_mouse_y_ = mouseY;
    } else if (!supersplat_dragging_ || supersplat_drag_button_ != activeButton) {
        supersplat_dragging_ = true;
        supersplat_drag_button_ = activeButton;
        last_mouse_x_ = mouseX;
        last_mouse_y_ = mouseY;
    } else {
        const float deltaX = static_cast<float>(mouseX - last_mouse_x_);
        const float deltaY = static_cast<float>(mouseY - last_mouse_y_);
        last_mouse_x_ = mouseX;
        last_mouse_y_ = mouseY;

        if (activeButton == GLFW_MOUSE_BUTTON_RIGHT) {
            int framebufferWidth = 0;
            int framebufferHeight = 0;
            glfwGetFramebufferSize(handle, &framebufferWidth, &framebufferHeight);
            supersplat_camera_controller_.pan(
                deltaX, deltaY,
                static_cast<float>(framebufferWidth),
                static_cast<float>(framebufferHeight));
        } else if (supersplat_camera_controller_.mode() == CameraControlMode::Fly) {
            supersplat_camera_controller_.look(deltaX, deltaY);
        } else {
            supersplat_camera_controller_.orbit(deltaX, deltaY);
        }
    }

    const bool toggleDown =
        glfwGetKey(handle, GLFW_KEY_V) == GLFW_PRESS;
    if (!captureKeyboard && toggleDown && !supersplat_toggle_key_down_) {
        supersplat_camera_controller_.toggleMode();
    }
    supersplat_toggle_key_down_ = toggleDown;

    const bool focusDown =
        glfwGetKey(handle, GLFW_KEY_F) == GLFW_PRESS;
    if (!captureKeyboard && focusDown && !supersplat_focus_key_down_) {
        const bool resetAngles =
            glfwGetKey(handle, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
            glfwGetKey(handle, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
        if (resetAngles) {
            resetCameraFromModel();
        } else {
            focusCameraOnModel();
        }
    }
    supersplat_focus_key_down_ = focusDown;

    if (!captureKeyboard &&
        supersplat_camera_controller_.mode() == CameraControlMode::Fly) {
        glm::vec3 localMotion(0.0f);
        if (glfwGetKey(handle, GLFW_KEY_D) == GLFW_PRESS) localMotion.x += 1.0f;
        if (glfwGetKey(handle, GLFW_KEY_A) == GLFW_PRESS) localMotion.x -= 1.0f;
        if (glfwGetKey(handle, GLFW_KEY_E) == GLFW_PRESS) localMotion.y += 1.0f;
        if (glfwGetKey(handle, GLFW_KEY_Q) == GLFW_PRESS) localMotion.y -= 1.0f;
        if (glfwGetKey(handle, GLFW_KEY_W) == GLFW_PRESS) localMotion.z += 1.0f;
        if (glfwGetKey(handle, GLFW_KEY_S) == GLFW_PRESS) localMotion.z -= 1.0f;

        float speedMultiplier = 1.0f;
        if (glfwGetKey(handle, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
            glfwGetKey(handle, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
            speedMultiplier *= 10.0f;
        }
        if (glfwGetKey(handle, GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
            glfwGetKey(handle, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS) {
            speedMultiplier *= 0.1f;
        }
        supersplat_camera_controller_.fly(
            localMotion, deltaTime, speedMultiplier);
    }
}

void Application::updateCameraMatrices(int framebufferWidth,
                                       int framebufferHeight) {
    if (framebufferWidth <= 0 || framebufferHeight <= 0) {
        return;
    }

    camera_near_plane_ = 0.1f;
    camera_far_plane_ = 100.0f;
    if (current_model_ && !current_model_->isEmpty()) {
        const auto clipping = supersplat_camera_controller_.fitClippingPlanes(
            camera_, current_model_->get_clip_center(), current_model_->get_clip_radius());
        camera_near_plane_ = clipping.first;
        camera_far_plane_ = clipping.second;
    }

    const float aspect = static_cast<float>(framebufferWidth) /
                         static_cast<float>(framebufferHeight);
    float verticalFov = camera_.get_fov();
    if (aspect > 1.0f) {
        verticalFov = glm::degrees(
            2.0f * std::atan(
                std::tan(glm::radians(camera_.get_fov()) * 0.5f) / aspect));
    }
    view_matrix_ = camera_.get_view_matrix();
    projection_matrix_ = camera_.get_projection_matrix(
        aspect, verticalFov, camera_near_plane_, camera_far_plane_);
}

void Application::applyRenderProfile() {
    if (auto* gsRenderer = dynamic_cast<GaussianRenderer*>(renderer_.get())) {
        gsRenderer->setRenderProfile(gaussian_render_profile_);
        gsRenderer->setSHBands(gaussian_sh_bands_);
    }
    camera_.set_fov(supersplat_camera_controller_.fovDegrees());
    supersplat_dragging_ = false;
}

void Application::drawImGuiControls() {
    ImGui::Begin("Camera");

    const ImGuiIO& io = ImGui::GetIO();
    ImGui::Text("FPS %.1f", io.Framerate);
    if (io.Framerate > 0.0f) {
        ImGui::Text("Frame %.2f ms", 1000.0f / io.Framerate);
    }
    ImGui::Separator();

    int renderProfileIndex = static_cast<int>(gaussian_render_profile_);
    const char* renderProfileLabels[] = {
        "Legacy",
        "SuperSplat Compatible",
    };
    if (ImGui::Combo("Render Profile", &renderProfileIndex,
                     renderProfileLabels, IM_ARRAYSIZE(renderProfileLabels))) {
        gaussian_render_profile_ = static_cast<GaussianRenderProfile>(
            std::clamp(renderProfileIndex, 0, 1));
        applyRenderProfile();
        saveTrainingSettings();
    }
    ImGui::TextDisabled("Both profiles share the same camera controls.");
    int shBands = static_cast<int>(gaussian_sh_bands_);
    if (ImGui::SliderInt("SH Bands", &shBands, 0, 3)) {
        gaussian_sh_bands_ = static_cast<uint32_t>(std::clamp(shBands, 0, 3));
        if (auto* gsRenderer = dynamic_cast<GaussianRenderer*>(renderer_.get())) {
            gsRenderer->setSHBands(gaussian_sh_bands_);
        }
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

    int cameraMode = static_cast<int>(supersplat_camera_controller_.mode());
    const char* cameraModes[] = {"Orbit", "Fly"};
    if (ImGui::Combo("Camera Mode", &cameraMode,
                     cameraModes, IM_ARRAYSIZE(cameraModes))) {
        supersplat_camera_controller_.setMode(
            static_cast<CameraControlMode>(std::clamp(cameraMode, 0, 1)));
    }

    float fov = supersplat_camera_controller_.fovDegrees();
    if (ImGui::SliderFloat("Field of View (larger axis)", &fov,
                           10.0f, 120.0f, "%.1f deg")) {
        supersplat_camera_controller_.setFovDegrees(fov);
    }
    float damping = supersplat_camera_controller_.dampingSeconds();
    if (ImGui::SliderFloat("Damping", &damping, 0.0f, 0.5f, "%.3f s")) {
        supersplat_camera_controller_.setDampingSeconds(damping);
    }
    float orbitSensitivity = supersplat_camera_controller_.orbitSensitivity();
    if (ImGui::SliderFloat("Orbit Sensitivity", &orbitSensitivity,
                           0.001f, 0.02f, "%.3f")) {
        supersplat_camera_controller_.setOrbitSensitivity(orbitSensitivity);
    }
    float panSensitivity = supersplat_camera_controller_.panSensitivity();
    if (ImGui::SliderFloat("Pan Sensitivity", &panSensitivity,
                           0.1f, 3.0f, "%.2f")) {
        supersplat_camera_controller_.setPanSensitivity(panSensitivity);
    }
    float zoomSensitivity = supersplat_camera_controller_.zoomSensitivity();
    if (ImGui::SliderFloat("Dolly Speed", &zoomSensitivity,
                           0.02f, 0.5f, "%.2f")) {
        supersplat_camera_controller_.setZoomSensitivity(zoomSensitivity);
    }
    float flySpeed = supersplat_camera_controller_.flySpeed();
    if (ImGui::SliderFloat("Fly Speed", &flySpeed,
                           0.1f, 10.0f, "%.2f")) {
        supersplat_camera_controller_.setFlySpeed(flySpeed);
    }

    float distance = supersplat_camera_controller_.distance();
    const float distanceLimit = std::max(distance * 3.0f, 20.0f);
    if (ImGui::SliderFloat("Distance", &distance, 0.01f,
                           distanceLimit, "%.3f")) {
        supersplat_camera_controller_.setDistance(distance);
    }
    float azimuth = supersplat_camera_controller_.azimuthDegrees();
    if (ImGui::SliderFloat("Azimuth", &azimuth, -180.0f, 180.0f, "%.1f deg")) {
        supersplat_camera_controller_.setAzimuthDegrees(azimuth);
    }
    float elevation = supersplat_camera_controller_.elevationDegrees();
    if (ImGui::SliderFloat("Elevation", &elevation, -89.0f, 89.0f, "%.1f deg")) {
        supersplat_camera_controller_.setElevationDegrees(elevation);
    }

    if (ImGui::Button("Focus")) {
        focusCameraOnModel();
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        resetCameraFromModel();
    }

    const glm::vec3& target = supersplat_camera_controller_.focalPoint();
    ImGui::Text("Target %.3f %.3f %.3f", target.x, target.y, target.z);
    ImGui::Text("Clip %.6g .. %.6g", camera_near_plane_, camera_far_plane_);
    if (current_model_ && !current_model_->isEmpty()) {
        ImGui::Text("Bounds focus %.4g, clip %.4g",
                    current_model_->get_focus_radius(),
                    current_model_->get_clip_radius());
    }
    ImGui::TextDisabled("LMB orbit/look, RMB pan, wheel dolly, V mode, F focus");
    ImGui::TextDisabled("Fly: WASDQE, Shift 10x, Alt 0.1x; Shift+F resets view");

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
    stopTrainingWorkerAndJoin();
    const std::string selector = *pending_training_gpu_selector_;
    pending_training_gpu_selector_.reset();
    training_running_ = false;
    training_pure_active_ = false;
    resetTrainingTimer();

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
        saveTrainingSettings();
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

    supersplat_camera_controller_.dolly(static_cast<float>(yoffset));
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
    const TrainingUiSnapshot trainingSnapshot = trainingSnapshotForUi();

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
        training_pure_active_ = false;
        resetTrainingTimer();
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
        training_pure_active_ = false;
        resetTrainingTimer();
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
            "Subgroup",
            "Tile Gaussian Atomic",
            "VkSplat Per-Splat",
            "VkSplat Tensor (Adapted)"
        };
        if (ImGui::Combo("Pixel Backward", &training_pixel_to_2dgs_mode_,
                         pixelBackwardModes, IM_ARRAYSIZE(pixelBackwardModes))) {
            training_pixel_to_2dgs_mode_ = std::clamp(training_pixel_to_2dgs_mode_, 0, 6);
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
        if (trainingSnapshot.rendererInitialized) {
            const int activePixelBackwardMode = static_cast<int>(trainingSnapshot.activePixelMode);
            ImGui::Text("Active Pixel Backward %s", pixelBackwardModes[activePixelBackwardMode]);
            if (training_pixel_to_2dgs_mode_ == 4 &&
                !trainingSnapshot.tileGaussianSupported) {
                ImGui::TextDisabled(
                    "Tile Gaussian unavailable on this GPU; using Direct.");
            }
            if (training_pixel_to_2dgs_mode_ == 5 &&
                !trainingSnapshot.vkSplatPerSplatSupported) {
                ImGui::TextDisabled(
                    "VkSplat Per-Splat unavailable on this GPU; using Direct.");
            }
            if (training_pixel_to_2dgs_mode_ == 6 &&
                !trainingSnapshot.vkSplatTensorSupported) {
                ImGui::TextDisabled(
                    "VkSplat Tensor requires 45 KiB shared memory; using Direct.");
            }
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
            ImGui::BeginDisabled();
        }
        ImGui::InputScalar("Benchmark Steps/Frame", ImGuiDataType_U32, &training_steps_per_frame_);
        training_steps_per_frame_ = std::max(training_steps_per_frame_, 1u);
        ImGui::Checkbox("Pure Training", &training_pure_mode_);
        if (training_pure_mode_) {
            ImGui::TextDisabled(
                "Only elapsed time refreshes; the final PLY is exported automatically.");
        }
        if (ImGui::InputScalar("Validation Interval",
                               ImGuiDataType_U32,
                               &training_validation_interval_)) {
            training_.setValidationInterval(training_validation_interval_);
        }
        if (!canEditTrainingSetup) {
            ImGui::EndDisabled();
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

    const bool benchmarkActive = trainingSnapshot.fixedBenchmark.active;
    const char* primaryTrainingButtonLabel = training_stop_requested_
        ? "Stopping Training..."
        : benchmarkActive
        ? "Stop Benchmark"
        : (training_running_ ? "Pause Training" : "Start Training");
    if (training_stop_requested_) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button(primaryTrainingButtonLabel)) {
        if (training_running_) {
            if (benchmarkActive) {
                training_.stopFixedWorkloadBenchmark();
                training_running_ = false;
                setTrainingStatus("Fixed workload benchmark stopped.");
            } else if (training_async_active_) {
                requestTrainingWorkerStop();
                setTrainingStatus("Training pause requested; waiting for the current step.");
            } else {
                stopTrainingTimer();
                training_running_ = false;
                training_pure_active_ = false;
                setTrainingStatus("Training paused.");
            }
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
                if (trainingSnapshot.trainingIteration == 0u) {
                    resetTrainingTimer();
                }
                startTrainingTimer();
                saveTrainingSettings();
                startTrainingWorker(training_pure_mode_);
                setTrainingStatus(training_pure_mode_
                    ? "Pure asynchronous training started; only elapsed time updates until completion."
                    : "Asynchronous training started.");
            } catch (const std::exception& error) {
                stopTrainingWorkerAndJoin();
                setTrainingError(error.what());
            }
        }
    }
    if (training_stop_requested_) {
        ImGui::EndDisabled();
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
                training_pure_active_ = false;
                training_running_ = true;
                setTrainingStatus("Fixed workload benchmark started.");
            } catch (const std::exception& error) {
                training_running_ = false;
                training_pure_active_ = false;
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

    const bool pureWorkerRunning = training_async_active_ && training_pure_active_;
    if (!pureWorkerRunning) {
        if (trainingSnapshot.totalIterations > 0) {
            ImGui::Text("Iterations %u / %u", trainingSnapshot.trainingIteration, trainingSnapshot.totalIterations);
        } else {
            ImGui::Text("Iterations %u", trainingSnapshot.trainingIteration);
        }
        const uint64_t displayedSteps = training_async_active_
            ? static_cast<uint64_t>(trainingSnapshot.trainingIteration)
            : training_steps_done_;
        ImGui::Text("Steps %llu", static_cast<unsigned long long>(displayedSteps));
    }
    const std::string trainingElapsed = formatTrainingDuration(trainingElapsedSeconds());
    ImGui::Text("Total training time %s%s",
                trainingElapsed.c_str(),
                training_timer_running_ ? " (running)" : "");
    if (!pureWorkerRunning) {
        ImGui::Text("Gaussians %u", trainingSnapshot.gaussianCount);
    } else {
        ImGui::TextDisabled(
            "Pure mode: detailed training data will be published after completion.");
    }
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
    const auto& benchmark = trainingSnapshot.fixedBenchmark;
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
    if (!pureWorkerRunning) {
    const auto& densification = trainingSnapshot.densification;
    const uint32_t densificationIteration = trainingSnapshot.densificationIteration;
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
    const auto& validation = trainingSnapshot.validation;
    ImGui::Text("Loss %.6g", validation.meanLoss);
    ImGui::Text("Render alpha %.6g", validation.meanRenderedAlpha);
    ImGui::Text("Tile items %u", validation.tileItemCount);
    ImGui::Text("Backward candidates avg %.2f, contributors avg %.2f, max candidates %u",
                validation.meanProcessedCandidatesPerPixel,
                validation.meanContributorsPerPixel,
                validation.maxProcessedCandidatesPerPixel);
    const auto& candidateProfile = trainingSnapshot.candidateProfile;
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
    if (!validation.valid && trainingSnapshot.trainingIteration > 0u) {
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
            "Tile-local backward",
            "2DGS to 3DGS",
            "Fused projection/optimizer",
            "Optimizer",
            "Validation",
            "Densify/prune",
        };
        static_assert(sizeof(cpuStageNames) / sizeof(cpuStageNames[0]) == kTrainingCpuProfileStageCount);
        static_assert(sizeof(gpuStageNames) / sizeof(gpuStageNames[0]) == kTrainingGpuProfileStageCount);
        const auto& profiling = trainingSnapshot.profiling;
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
        const ImageStreamerStats cacheStats = trainingSnapshot.imageCache;
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
        const DeviceImageCacheStats deviceCacheStats = trainingSnapshot.deviceImageCache;
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
                static_cast<unsigned long long>(trainingSnapshot.currentFrameIndex),
                training_frame_count_);
    }

    ImGui::Separator();
    if (training_running_) {
        ImGui::BeginDisabled();
    }
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
    if (training_running_) {
        ImGui::EndDisabled();
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

    if (!training_running_) {
        drawTrainingFileDialogs();
    }
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
    stopTrainingWorkerAndJoin();
    training_running_ = false;
    training_pure_active_ = false;
    resetTrainingTimer();
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

Application::TrainingUiSnapshot Application::captureTrainingSnapshot() const {
    TrainingUiSnapshot snapshot{};
    snapshot.rendererInitialized = training_.isRendererInitialized();
    snapshot.hasDataset = training_.hasDataset();
    snapshot.hasTrainableModel = training_.hasTrainableModel();
    snapshot.trainingComplete = training_.isTrainingComplete();
    snapshot.subgroupSupported = training_.subgroupPixelTo2DGSSupported();
    snapshot.tileGaussianSupported = training_.tileGaussianPixelTo2DGSSupported();
    snapshot.vkSplatPerSplatSupported = training_.vkSplatPerSplatSupported();
    snapshot.vkSplatTensorSupported = training_.vkSplatTensorSupported();
    snapshot.activePixelMode = training_.activePixelTo2DGSMode();
    snapshot.gaussianCount = training_.gaussianCount();
    snapshot.trainingIteration = training_.trainingIteration();
    snapshot.totalIterations = training_.totalIterations();
    snapshot.densificationIteration = training_.densificationStatsIteration();
    snapshot.currentFrameIndex = snapshot.hasDataset
        ? training_.currentFrameIndex()
        : 0u;
    snapshot.fixedBenchmark = training_.fixedWorkloadBenchmarkStats();
    snapshot.validation = training_.validationStats();
    snapshot.candidateProfile = training_.candidateProfileStats();
    snapshot.densification = training_.densificationStats();
    snapshot.profiling = training_.profilingStats();
    snapshot.imageCache = training_.imageCacheStats();
    snapshot.deviceImageCache = training_.deviceImageCacheStats();
    return snapshot;
}

void Application::publishTrainingSnapshot() {
    TrainingUiSnapshot snapshot = captureTrainingSnapshot();
    std::lock_guard lock(training_snapshot_mutex_);
    training_snapshot_ = std::move(snapshot);
    training_snapshot_valid_ = true;
}

Application::TrainingUiSnapshot Application::trainingSnapshotForUi() const {
    {
        std::lock_guard lock(training_snapshot_mutex_);
        if (training_snapshot_valid_ && training_async_active_) {
            return training_snapshot_;
        }
    }
    return captureTrainingSnapshot();
}

void Application::startTrainingWorker(bool pure) {
    if (training_worker_.joinable()) {
        training_worker_.join();
    }
    training_worker_result_ready_.store(false, std::memory_order_release);
    training_async_active_ = true;
    training_stop_requested_ = false;
    training_pure_active_ = pure;
    training_running_ = true;
    training_pure_last_ui_render_time_ = 0.0;
    publishTrainingSnapshot();
    training_worker_ = std::jthread(
        [this, pure](std::stop_token stopToken) {
            trainingWorkerMain(stopToken, pure);
        });
}

void Application::trainingWorkerMain(std::stop_token stopToken, bool pure) {
    TrainingWorkerResult result{};
    result.pure = pure;
    try {
        uint32_t stepsSinceSnapshot = 0u;
        while (!stopToken.stop_requested() && !training_.isTrainingComplete()) {
            training_.trainStep();
            ++stepsSinceSnapshot;
            if (!pure && stepsSinceSnapshot >= 32u) {
                publishTrainingSnapshot();
                stepsSinceSnapshot = 0u;
            }
        }
        result.completed = training_.isTrainingComplete();
    } catch (const std::exception& error) {
        result.error = error.what();
    } catch (...) {
        result.error = "Unknown exception in asynchronous training worker";
    }

    result.finishedAt = std::chrono::steady_clock::now();
    try {
        publishTrainingSnapshot();
    } catch (const std::exception& error) {
        if (result.error.empty()) {
            result.error = std::string("Failed to publish final training snapshot: ") +
                           error.what();
        }
    }
    {
        std::lock_guard lock(training_worker_result_mutex_);
        training_worker_result_ = std::move(result);
    }
    training_worker_result_ready_.store(true, std::memory_order_release);
}

void Application::requestTrainingWorkerStop() {
    if (!training_async_active_ || !training_worker_.joinable()) {
        return;
    }
    training_stop_requested_ = true;
    training_worker_.request_stop();
}

void Application::stopTrainingWorkerAndJoin() {
    if (training_worker_.joinable()) {
        training_worker_.request_stop();
        training_worker_.join();
    }
    training_worker_result_ready_.store(false, std::memory_order_release);
    training_async_active_ = false;
    training_stop_requested_ = false;
    training_running_ = false;
    training_pure_active_ = false;
    stopTrainingTimer();
}

void Application::consumeTrainingWorkerResult() {
    if (!training_worker_result_ready_.load(std::memory_order_acquire)) {
        return;
    }
    if (training_worker_.joinable()) {
        training_worker_.join();
    }

    TrainingWorkerResult result{};
    {
        std::lock_guard lock(training_worker_result_mutex_);
        result = training_worker_result_;
    }
    training_worker_result_ready_.store(false, std::memory_order_release);
    training_async_active_ = false;
    training_stop_requested_ = false;
    stopTrainingTimerAt(result.finishedAt);

    TrainingUiSnapshot snapshot{};
    {
        std::lock_guard lock(training_snapshot_mutex_);
        snapshot = training_snapshot_;
    }
    training_steps_done_ = snapshot.trainingIteration;

    if (!result.error.empty()) {
        training_running_ = false;
        training_pure_active_ = false;
        setTrainingError(result.error);
        return;
    }
    if (result.completed) {
        try {
            finishTrainingRun();
        } catch (const std::exception& error) {
            training_running_ = false;
            training_pure_active_ = false;
            setTrainingError(error.what());
        }
        return;
    }

    training_running_ = false;
    training_pure_active_ = false;
    setTrainingStatus("Training paused at " +
                      std::to_string(snapshot.trainingIteration) +
                      " iterations after " +
                      formatTrainingDuration(trainingElapsedSeconds()) + ".");
}

void Application::runTrainingStepFromUi() {
    if (!training_dataset_loaded_) {
        throw std::runtime_error("Load a training dataset before starting training");
    }
    if (!training_.isFixedWorkloadBenchmarkActive() && training_.isTrainingComplete()) {
        finishTrainingRun();
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
        finishTrainingRun();
    }
}

void Application::finishTrainingRun() {
    const bool completedPureTraining = training_pure_active_;
    training_running_ = false;
    training_pure_active_ = false;
    stopTrainingTimer();

    std::string status = "Training completed at " +
                         std::to_string(training_.trainingIteration()) +
                         " iterations in " +
                         formatTrainingDuration(trainingElapsedSeconds()) + ".";
    if (completedPureTraining) {
        std::filesystem::path outputPath = outputPlyPath();
        if (outputPath.extension().empty()) {
            outputPath += ".ply";
        }
        if (!training_.exportToPLY(outputPath)) {
            throw std::runtime_error("Failed to export PLY: " + outputPath.string());
        }
        status += " Saved PLY: " + outputPath.string();
    }
    setTrainingStatus(status);
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
        static_cast<TrainingPixelTo2DGSMode>(std::clamp(training_pixel_to_2dgs_mode_, 0, 6)));
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

void Application::loadTrainingSettings() {
    const std::filesystem::path path = trainingSettingsPath();
    std::ifstream input(path);
    if (!input.is_open()) {
        return;
    }

    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        std::string key;
        row >> key;
        if (key.empty() || key[0] == '#') {
            continue;
        }

        if (key == "dataset") {
            std::string value;
            if (row >> std::quoted(value)) copyToInputBuffer(training_dataset_path_, value);
        } else if (key == "output_dir") {
            std::string value;
            if (row >> std::quoted(value)) copyToInputBuffer(training_output_dir_, value);
        } else if (key == "output_name") {
            std::string value;
            if (row >> std::quoted(value)) copyToInputBuffer(training_output_name_, value);
        } else if (key == "gpu_selector") {
            row >> std::quoted(persisted_training_gpu_selector_);
        } else if (key == "render_profile") {
            int value = 0;
            if (row >> value) {
                gaussian_render_profile_ = static_cast<GaussianRenderProfile>(
                    std::clamp(value, 0, 1));
            }
        } else if (key == "gaussian_sh_bands") {
            row >> gaussian_sh_bands_;
        } else if (key == "camera_mode" || key == "supersplat_camera_mode") {
            int value = 0;
            if (row >> value) {
                supersplat_camera_controller_.setMode(
                    static_cast<CameraControlMode>(std::clamp(value, 0, 1)));
            }
        } else if (key == "camera_fov" || key == "supersplat_camera_fov") {
            float value = 75.0f;
            if (row >> value) supersplat_camera_controller_.setFovDegrees(value);
        } else if (key == "camera_damping" || key == "supersplat_camera_damping") {
            float value = 0.12f;
            if (row >> value) supersplat_camera_controller_.setDampingSeconds(value);
        } else if (key == "camera_orbit_sensitivity" || key == "supersplat_orbit_sensitivity") {
            float value = 0.005f;
            if (row >> value) supersplat_camera_controller_.setOrbitSensitivity(value);
        } else if (key == "camera_pan_sensitivity" || key == "supersplat_pan_sensitivity") {
            float value = 1.0f;
            if (row >> value) supersplat_camera_controller_.setPanSensitivity(value);
        } else if (key == "camera_zoom_sensitivity" || key == "supersplat_zoom_sensitivity") {
            float value = 0.12f;
            if (row >> value) supersplat_camera_controller_.setZoomSensitivity(value);
        } else if (key == "camera_fly_speed" || key == "supersplat_fly_speed") {
            float value = 1.0f;
            if (row >> value) supersplat_camera_controller_.setFlySpeed(value);
        } else if (key == "downscale") row >> training_downscale_;
        else if (key == "initial_gaussians") row >> training_initial_gaussians_;
        else if (key == "random_seed") row >> training_random_seed_;
        else if (key == "initial_opacity") row >> training_initial_opacity_;
        else if (key == "scene_radius_scale") row >> training_scene_radius_scale_;
        else if (key == "training_mode") row >> training_mode_;
        else if (key == "pixel_to_2dgs_mode") row >> training_pixel_to_2dgs_mode_;
        else if (key == "pixel_to_2dgs_min_subgroup_utilization") row >> training_pixel_to_2dgs_min_subgroup_utilization_;
        else if (key == "forward_composite_mode") row >> training_forward_composite_mode_;
        else if (key == "total_iterations") row >> training_total_iterations_;
        else if (key == "steps_per_frame") row >> training_steps_per_frame_;
        else if (key == "validation_interval") row >> training_validation_interval_;
        else if (key == "benchmark_frame") row >> training_benchmark_frame_;
        else if (key == "benchmark_warmup") row >> training_benchmark_warmup_steps_;
        else if (key == "benchmark_measured") row >> training_benchmark_measured_steps_;
        else if (key == "pure_training") {
            int value = 0;
            if (row >> value) training_pure_mode_ = value != 0;
        } else if (key == "position_lr") row >> training_position_lr_;
        else if (key == "position_lr_final") row >> training_position_lr_final_;
        else if (key == "position_lr_delay_mult") row >> training_position_lr_delay_mult_;
        else if (key == "position_lr_delay_steps") row >> training_position_lr_delay_steps_;
        else if (key == "position_lr_max_steps") row >> training_position_lr_max_steps_;
        else if (key == "feature_lr") row >> training_feature_lr_;
        else if (key == "feature_rest_lr") row >> training_feature_rest_lr_;
        else if (key == "opacity_lr") row >> training_opacity_lr_;
        else if (key == "scale_lr") row >> training_scale_lr_;
        else if (key == "rotation_lr") row >> training_rotation_lr_;
        else if (key == "adam_beta1") row >> training_adam_beta1_;
        else if (key == "adam_beta2") row >> training_adam_beta2_;
        else if (key == "adam_epsilon") row >> training_adam_epsilon_;
        else if (key == "grad_clip") row >> training_grad_clip_;
        else if (key == "loss_dssim_weight") row >> training_loss_dssim_weight_;
        else if (key == "max_sh_degree") row >> training_max_sh_degree_;
        else if (key == "sh_degree_interval") row >> training_sh_degree_interval_;
        else if (key == "densification_enabled") {
            int value = 0;
            if (row >> value) training_densification_enabled_ = value != 0;
        } else if (key == "densify_from_iteration") row >> training_densify_from_iteration_;
        else if (key == "densify_until_iteration") row >> training_densify_until_iteration_;
        else if (key == "densification_interval") row >> training_densification_interval_;
        else if (key == "opacity_reset_interval") row >> training_opacity_reset_interval_;
        else if (key == "max_gaussians") row >> training_max_gaussians_;
        else if (key == "split_children") row >> training_split_children_;
        else if (key == "densify_grad_threshold") row >> training_densify_grad_threshold_;
        else if (key == "min_opacity") row >> training_min_opacity_;
        else if (key == "percent_dense") row >> training_percent_dense_;
        else if (key == "screen_prune_size") row >> training_screen_size_prune_threshold_;
        else if (key == "world_prune_size") row >> training_world_size_prune_threshold_;
    }

    training_downscale_ = std::clamp(training_downscale_, 1, 16);
    gaussian_sh_bands_ = std::min(gaussian_sh_bands_, 3u);
    training_mode_ = std::clamp(training_mode_, 0, 1);
    training_pixel_to_2dgs_mode_ = std::clamp(training_pixel_to_2dgs_mode_, 0, 6);
    training_pixel_to_2dgs_min_subgroup_utilization_ = std::clamp(
        training_pixel_to_2dgs_min_subgroup_utilization_, 0.0f, 1.0f);
    training_forward_composite_mode_ = std::clamp(training_forward_composite_mode_, 0, 1);
    training_steps_per_frame_ = std::max(training_steps_per_frame_, 1u);
    training_benchmark_measured_steps_ = std::max(training_benchmark_measured_steps_, 1u);
    training_max_sh_degree_ = std::min(training_max_sh_degree_, 3u);
    training_sh_degree_interval_ = std::max(training_sh_degree_interval_, 1u);
    training_split_children_ = std::clamp(training_split_children_, 2u, 8u);

    LOG_INFO("Loaded training settings from {}", path.string());
}

void Application::saveTrainingSettings() const {
    try {
        const std::filesystem::path path = trainingSettingsPath();
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        std::ofstream output(path, std::ios::trunc);
        if (!output.is_open()) {
            LOG_WARN("Could not save training settings to {}", path.string());
            return;
        }

        std::string gpuSelector = persisted_training_gpu_selector_;
        if (training_device_) {
            const auto& gpu = training_device_->getSelectedPhysicalDeviceInfo();
            gpuSelector = gpu.uuid.empty() ? std::to_string(gpu.vulkanIndex) : gpu.uuid;
        }

        output << std::setprecision(std::numeric_limits<float>::max_digits10);
        output << "version 1\n";
        output << "dataset " << std::quoted(inputBufferString(training_dataset_path_)) << '\n';
        output << "output_dir " << std::quoted(inputBufferString(training_output_dir_)) << '\n';
        output << "output_name " << std::quoted(inputBufferString(training_output_name_)) << '\n';
        output << "gpu_selector " << std::quoted(gpuSelector) << '\n';
        output << "render_profile " << static_cast<int>(gaussian_render_profile_) << '\n';
        output << "gaussian_sh_bands " << gaussian_sh_bands_ << '\n';
        output << "camera_mode "
               << static_cast<int>(supersplat_camera_controller_.mode()) << '\n';
        output << "camera_fov "
               << supersplat_camera_controller_.fovDegrees() << '\n';
        output << "camera_damping "
               << supersplat_camera_controller_.dampingSeconds() << '\n';
        output << "camera_orbit_sensitivity "
               << supersplat_camera_controller_.orbitSensitivity() << '\n';
        output << "camera_pan_sensitivity "
               << supersplat_camera_controller_.panSensitivity() << '\n';
        output << "camera_zoom_sensitivity "
               << supersplat_camera_controller_.zoomSensitivity() << '\n';
        output << "camera_fly_speed "
               << supersplat_camera_controller_.flySpeed() << '\n';
        output << "downscale " << training_downscale_ << '\n';
        output << "initial_gaussians " << training_initial_gaussians_ << '\n';
        output << "random_seed " << training_random_seed_ << '\n';
        output << "initial_opacity " << training_initial_opacity_ << '\n';
        output << "scene_radius_scale " << training_scene_radius_scale_ << '\n';
        output << "training_mode " << training_mode_ << '\n';
        output << "pixel_to_2dgs_mode " << training_pixel_to_2dgs_mode_ << '\n';
        output << "pixel_to_2dgs_min_subgroup_utilization " << training_pixel_to_2dgs_min_subgroup_utilization_ << '\n';
        output << "forward_composite_mode " << training_forward_composite_mode_ << '\n';
        output << "total_iterations " << training_total_iterations_ << '\n';
        output << "steps_per_frame " << training_steps_per_frame_ << '\n';
        output << "validation_interval " << training_validation_interval_ << '\n';
        output << "benchmark_frame " << training_benchmark_frame_ << '\n';
        output << "benchmark_warmup " << training_benchmark_warmup_steps_ << '\n';
        output << "benchmark_measured " << training_benchmark_measured_steps_ << '\n';
        output << "pure_training " << (training_pure_mode_ ? 1 : 0) << '\n';
        output << "position_lr " << training_position_lr_ << '\n';
        output << "position_lr_final " << training_position_lr_final_ << '\n';
        output << "position_lr_delay_mult " << training_position_lr_delay_mult_ << '\n';
        output << "position_lr_delay_steps " << training_position_lr_delay_steps_ << '\n';
        output << "position_lr_max_steps " << training_position_lr_max_steps_ << '\n';
        output << "feature_lr " << training_feature_lr_ << '\n';
        output << "feature_rest_lr " << training_feature_rest_lr_ << '\n';
        output << "opacity_lr " << training_opacity_lr_ << '\n';
        output << "scale_lr " << training_scale_lr_ << '\n';
        output << "rotation_lr " << training_rotation_lr_ << '\n';
        output << "adam_beta1 " << training_adam_beta1_ << '\n';
        output << "adam_beta2 " << training_adam_beta2_ << '\n';
        output << "adam_epsilon " << training_adam_epsilon_ << '\n';
        output << "grad_clip " << training_grad_clip_ << '\n';
        output << "loss_dssim_weight " << training_loss_dssim_weight_ << '\n';
        output << "max_sh_degree " << training_max_sh_degree_ << '\n';
        output << "sh_degree_interval " << training_sh_degree_interval_ << '\n';
        output << "densification_enabled " << (training_densification_enabled_ ? 1 : 0) << '\n';
        output << "densify_from_iteration " << training_densify_from_iteration_ << '\n';
        output << "densify_until_iteration " << training_densify_until_iteration_ << '\n';
        output << "densification_interval " << training_densification_interval_ << '\n';
        output << "opacity_reset_interval " << training_opacity_reset_interval_ << '\n';
        output << "max_gaussians " << training_max_gaussians_ << '\n';
        output << "split_children " << training_split_children_ << '\n';
        output << "densify_grad_threshold " << training_densify_grad_threshold_ << '\n';
        output << "min_opacity " << training_min_opacity_ << '\n';
        output << "percent_dense " << training_percent_dense_ << '\n';
        output << "screen_prune_size " << training_screen_size_prune_threshold_ << '\n';
        output << "world_prune_size " << training_world_size_prune_threshold_ << '\n';
    } catch (const std::exception& error) {
        LOG_WARN("Could not save training settings: {}", error.what());
    }
}

void Application::startTrainingTimer() {
    if (training_timer_running_) {
        return;
    }
    training_timer_started_at_ = std::chrono::steady_clock::now();
    training_timer_running_ = true;
}

void Application::stopTrainingTimer() {
    stopTrainingTimerAt(std::chrono::steady_clock::now());
}

void Application::stopTrainingTimerAt(
    std::chrono::steady_clock::time_point endTime) {
    if (!training_timer_running_) {
        return;
    }
    if (endTime > training_timer_started_at_) {
        training_elapsed_seconds_ +=
            std::chrono::duration<double>(
                endTime - training_timer_started_at_).count();
    }
    training_timer_running_ = false;
}

void Application::resetTrainingTimer() {
    training_elapsed_seconds_ = 0.0;
    training_timer_started_at_ = {};
    training_timer_running_ = false;
}

double Application::trainingElapsedSeconds() const {
    if (!training_timer_running_) {
        return training_elapsed_seconds_;
    }
    return training_elapsed_seconds_ +
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - training_timer_started_at_).count();
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
