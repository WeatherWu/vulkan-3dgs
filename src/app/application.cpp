#include "app/application.hpp"

#include "app/window.hpp"
#include "context/context.hpp"
#include "graphics/gaussian_model.hpp"
#include "graphics/gaussian_renderer.hpp"
#include "render/renderer.hpp"
#include "utils/logger.hpp"
#include "viewer/viewer_panel.hpp"

#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <exception>
#include <filesystem>
#include <thread>

namespace vulkan3DGS {
namespace {

bool hasExtension(const std::filesystem::path& path, const std::string& extension) {
    std::string actual = path.extension().string();
    std::transform(actual.begin(), actual.end(), actual.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return actual == extension;
}

} // namespace

Application::Application(const std::string& title, int width, int height, RenderMode mode,
                         std::optional<std::string> gpuSelector)
    : currentMode_(mode) {
#ifdef _DEBUG
    Logger::get_instance().set_level(LogLevel::DEBUG_VULKAN_3DGS);
#endif
    LOG_INFO("Initializing Vulkan+3DGS Application");

    viewerController_.loadSettings();
    Context::initializeVulkanLoader();
    if (!glfwInit()) {
        throw std::runtime_error("GLFW initialization failed");
    }

    window_ = std::make_unique<Window>(title, width, height);
    vulkanContext_ = &Context::Instance();
    vulkanContext_->initialize(window_->get_handle());

    renderer_ = createRenderer(currentMode_);
    renderer_->initialize(window_->get_handle());
    configureRenderer();
    trainingController_.initializeDevice(std::move(gpuSelector));

    window_->set_resize_callback([this](int framebufferWidth, int framebufferHeight) {
        if (framebufferWidth <= 0 || framebufferHeight <= 0) return;
        if (renderer_) {
            renderer_->onResize(static_cast<uint32_t>(framebufferWidth),
                                static_cast<uint32_t>(framebufferHeight));
        }
        if (viewerController_.hasCamera()) {
            viewerController_.updateCamera(0.0f, framebufferWidth, framebufferHeight);
        }
    });
    window_->set_drop_callback(
        [this](const std::vector<std::string>& paths) { handleDroppedFiles(paths); });
    window_->set_scroll_callback(
        [this](double xOffset, double yOffset) { handleScroll(xOffset, yOffset); });
    initialize();
}

Application::~Application() {
    cleanup();
    glfwTerminate();
}

void Application::run() {
    LOG_INFO("Starting application main loop");
    while (!window_->should_close() && running_)
        tick();
}

void Application::tick() {
    window_->poll_events();
    trainingController_.tick();

    const double currentTime = glfwGetTime();
    float deltaTime = 0.0f;
    if (hasLastTickTime_) {
        deltaTime = std::clamp(static_cast<float>(currentTime - lastTickTime_), 0.0f, 0.1f);
    } else {
        hasLastTickTime_ = true;
    }
    lastTickTime_ = currentTime;
    update(deltaTime);

    bool renderUi = true;
    if (trainingController_.isActive() && trainingController_.state().pureActive) {
        constexpr double pureUiIntervalSeconds = 0.1;
        renderUi = currentTime - pureLastUiRenderTime_ >= pureUiIntervalSeconds;
    }
    if (renderUi) {
        pureLastUiRenderTime_ = currentTime;
        render();
    } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void Application::switchRenderMode(RenderMode mode) {
    if (currentMode_ == mode) return;
    LOG_INFO("Switching render mode from {} to {}", static_cast<int>(currentMode_),
             static_cast<int>(mode));
    if (renderer_) {
        renderer_->cleanup();
        renderer_.reset();
    }
    currentMode_ = mode;
    renderer_ = createRenderer(currentMode_);
    if (!renderer_) {
        LOG_ERROR("Failed to create renderer for mode: {}", static_cast<int>(mode));
        return;
    }
    renderer_->initialize(window_->get_handle());
    configureRenderer();
}

std::unique_ptr<Renderer> Application::createRenderer(RenderMode mode) {
    if (mode == RenderMode::GaussianGraphics) {
        return std::make_unique<GaussianRenderer>();
    }
    LOG_ERROR("Unknown render mode: {}", static_cast<int>(mode));
    return nullptr;
}

void Application::configureRenderer() {
    if (auto* gaussianRenderer = dynamic_cast<GaussianRenderer*>(renderer_.get())) {
        gaussianRenderer->setPresentModePreference(viewerController_.presentMode());
        gaussianRenderer->setRenderProfile(viewerController_.renderProfile());
        gaussianRenderer->setSHBands(viewerController_.shBands());
    }
    renderer_->setImGuiDrawCallback([this]() { drawImGuiControls(); });
}

void Application::initialize() {}

void Application::update(float deltaTime) {
    updateViewer(deltaTime);
}

void Application::updateViewer(float deltaTime) {
    if (!window_ || !viewerController_.hasCamera()) return;
    GLFWwindow* handle = window_->get_handle();
    const ImGuiIO* io = ImGui::GetCurrentContext() ? &ImGui::GetIO() : nullptr;

    ViewerInputState input{};
    glfwGetCursorPos(handle, &input.mouseX, &input.mouseY);
    input.orbitButton = glfwGetMouseButton(handle, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    input.panButton = glfwGetMouseButton(handle, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    input.captureMouse = io && io->WantCaptureMouse;
    input.captureKeyboard = io && io->WantCaptureKeyboard;
    input.toggleMode = glfwGetKey(handle, GLFW_KEY_V) == GLFW_PRESS;
    input.focus = glfwGetKey(handle, GLFW_KEY_F) == GLFW_PRESS;
    const bool shift = glfwGetKey(handle, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                       glfwGetKey(handle, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
    const bool alt = glfwGetKey(handle, GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
                     glfwGetKey(handle, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;
    input.resetFocus = input.focus && shift;
    input.moveRight = glfwGetKey(handle, GLFW_KEY_D) == GLFW_PRESS;
    input.moveLeft = glfwGetKey(handle, GLFW_KEY_A) == GLFW_PRESS;
    input.moveUp = glfwGetKey(handle, GLFW_KEY_E) == GLFW_PRESS;
    input.moveDown = glfwGetKey(handle, GLFW_KEY_Q) == GLFW_PRESS;
    input.moveForward = glfwGetKey(handle, GLFW_KEY_W) == GLFW_PRESS;
    input.moveBackward = glfwGetKey(handle, GLFW_KEY_S) == GLFW_PRESS;
    input.speedMultiplier = shift ? 10.0f : 1.0f;
    if (alt) input.speedMultiplier *= 0.1f;
    glfwGetFramebufferSize(handle, &input.framebufferWidth, &input.framebufferHeight);

    viewerController_.updateInput(input, deltaTime);
    viewerController_.updateCamera(deltaTime, input.framebufferWidth, input.framebufferHeight);
}

void Application::render() {
    if (!renderer_) return;
    auto* gaussianRenderer = dynamic_cast<GaussianRenderer*>(renderer_.get());
    if (gaussianRenderer) {
        gaussianRenderer->setPresentModePreference(viewerController_.presentMode());
        gaussianRenderer->setRenderProfile(viewerController_.renderProfile());
        gaussianRenderer->setSHBands(viewerController_.shBands());
        const ViewerRenderData data = viewerController_.renderData();
        if (data.model) {
            gaussianRenderer->setRenderData(data.model, data.view, data.projection, data.camera,
                                            data.modelMatrix);
        }
    }
    renderer_->render();
}

void Application::setModel(const GaussianModel* model) {
    viewerController_.setExternalModel(model);
    int width = 1;
    int height = 1;
    if (window_) {
        glfwGetFramebufferSize(window_->get_handle(), &width, &height);
    }
    viewerController_.resetCamera(static_cast<float>(std::max(width, 1)),
                                  static_cast<float>(std::max(height, 1)));
}

bool Application::loadModelFromFile(const std::string& filename) {
    if (!viewerController_.loadModel(filename)) {
        LOG_ERROR("Failed to load model: {}", filename);
        return false;
    }
    int width = 1;
    int height = 1;
    glfwGetFramebufferSize(window_->get_handle(), &width, &height);
    viewerController_.resetCamera(static_cast<float>(std::max(width, 1)),
                                  static_cast<float>(std::max(height, 1)));
    LOG_INFO("Loaded model from dropped file: {}", filename);
    return true;
}

void Application::setCamera(const glm::mat4& view, const glm::mat4& projection) {
    viewerController_.setCameraMatrices(view, projection);
}

void Application::setTrueCamera(const Camera& camera) {
    viewerController_.setCamera(camera);
}

void Application::drawImGuiControls() {
    int width = 1;
    int height = 1;
    glfwGetFramebufferSize(window_->get_handle(), &width, &height);
    ViewerPanel::draw(viewerController_, ImGui::GetIO().Framerate, width, height);
    trainingPanel_.draw(trainingController_);
}

void Application::handleScroll(double xOffset, double yOffset) {
    (void)xOffset;
    if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse) return;
    viewerController_.dolly(static_cast<float>(yOffset));
}

void Application::handleDroppedFiles(const std::vector<std::string>& paths) {
    if (paths.empty()) return;
    if (paths.size() > 1) {
        LOG_WARN("Multiple files dropped; loading the first one only");
    }
    const std::filesystem::path path(paths.front());
    if (std::filesystem::is_directory(path)) {
        trainingController_.setDatasetPath(path, true);
    } else if (hasExtension(path, ".ply")) {
        loadModelFromFile(paths.front());
    } else {
        trainingController_.reportError(
            "Unsupported dropped file. Drop a .ply model or a dataset folder.");
    }
}

void Application::cleanup() {
    if (cleanedUp_) return;
    cleanedUp_ = true;
    trainingController_.shutdown();
    try {
        viewerController_.saveSettings();
    } catch (const std::exception& error) {
        LOG_WARN("Could not save viewer settings: {}", error.what());
    }
    if (vulkanContext_ && vulkanContext_->isInitialized()) {
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vulkanContext_->Device());
    }
    if (renderer_) renderer_->cleanup();
}

} // namespace vulkan3DGS
