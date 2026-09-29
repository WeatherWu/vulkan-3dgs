#include "gaussian_renderer.hpp"

#include "context/context.hpp"
#include "gaussian_model.hpp"
#include "renderpass.hpp"
#include "utils/logger.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <cstdio>
#include <filesystem>
#include <stdexcept>

namespace vulkan3DGS {

namespace {

void loadImGuiFonts() {
    ImGuiIO& io = ImGui::GetIO();
    constexpr const char* fontCandidates[] = {
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    };
    for (const char* fontPath : fontCandidates) {
        if (std::filesystem::exists(fontPath)) {
            io.Fonts->AddFontFromFileTTF(fontPath, 16.0f, nullptr,
                                         io.Fonts->GetGlyphRangesChineseFull());
            LOG_INFO("Loaded ImGui CJK font: {}", fontPath);
            return;
        }
    }
    io.Fonts->AddFontDefault();
    LOG_WARN("No CJK font found for ImGui; Chinese text may not render correctly");
}

} // namespace

GaussianRenderer::GaussianRenderer() = default;

GaussianRenderer::~GaussianRenderer() noexcept {
    try {
        cleanup();
    } catch (...) {
        std::fputs("GaussianRenderer cleanup failed during destruction\n", stderr);
    }
}

void GaussianRenderer::initialize(GLFWwindow* window) {
    LOG_INFO("Starting GaussianRenderer initialization");
    window_ = window;

    frameRuntime_.initialize(window, presentModePreference_, RenderPass::DepthFormat,
                             [this](vk::Format format, vk::Extent2D extent) {
                                 pipelineSet_.initialize(format, extent);
                                 return pipelineSet_.renderPass();
                             });
    sorter_.initialize();
    resources_.initialize(frameRuntime_.frameCount(), pipelineSet_.descriptorSetLayout(),
                          sorter_.descriptorSetLayout());
    resources_.updateDescriptors(sorter_);
    uniforms_.resize(frameRuntime_.frameCount());
    initializeImGui(window);
    LOG_INFO("Gaussian Renderer initialized successfully");
}

void GaussianRenderer::cleanup() {
    const vk::Device device = getDevice();
    if (device) {
        device.waitIdle();
    }
    shutdownImGui();
    resources_.cleanup();
    sorter_.cleanup();
    frameRuntime_.cleanup();
    pipelineSet_.cleanup();
    uniforms_.clear();
    currentModel_ = nullptr;
    window_ = nullptr;
}

void GaussianRenderer::initializeImGui(GLFWwindow* window) {
    if (imguiInitialized_) {
        return;
    }
    auto& context = Context::Instance();
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    loadImGuiFonts();
    if (!ImGui_ImplGlfw_InitForVulkan(window, true)) {
        ImGui::DestroyContext();
        throw std::runtime_error("Failed to initialize ImGui GLFW backend");
    }

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_2;
    initInfo.Instance = context.getInstance();
    initInfo.PhysicalDevice = context.PhysicalDevice();
    initInfo.Device = getDevice();
    const auto graphicsIndex = context.getDevice().getQueueFamilyIndices().graphicsIndex;
    if (!graphicsIndex) throw std::runtime_error("Graphics queue family is unavailable");
    initInfo.QueueFamily = *graphicsIndex;
    initInfo.Queue = context.getDevice().getGraphicsQueue();
    initInfo.DescriptorPool = VK_NULL_HANDLE;
    initInfo.DescriptorPoolSize = 64;
    initInfo.PipelineInfoMain.RenderPass = pipelineSet_.renderPass();
    initInfo.MinImageCount = frameRuntime_.imageCount();
    initInfo.ImageCount = frameRuntime_.imageCount();
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    if (!ImGui_ImplVulkan_Init(&initInfo)) {
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        throw std::runtime_error("Failed to initialize ImGui Vulkan backend");
    }
    imguiInitialized_ = true;
    LOG_INFO("ImGui initialized");
}

void GaussianRenderer::shutdownImGui() {
    if (!imguiInitialized_) {
        return;
    }
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    imguiInitialized_ = false;
}

void GaussianRenderer::beginImGuiFrame() {
    if (!imguiInitialized_) {
        return;
    }
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    if (imguiDrawCallback_) {
        imguiDrawCallback_();
    }
    ImGui::Render();
}

void GaussianRenderer::render() {
    renderToImage();
    presentImage();
}

void GaussianRenderer::renderToImage() {
    const auto formatHandler = [this](vk::Format format, vk::Extent2D extent) {
        return rebuildFormatDependentResources(format, extent);
    };
    const std::optional<uint32_t> imageIndex = frameRuntime_.acquireFrame(
        pipelineSet_.renderPass(), RenderPass::DepthFormat, formatHandler);
    if (!imageIndex) {
        return;
    }

    prepareFrameData();
    beginImGuiFrame();
    Pipeline& pipeline = pipelineSet_.active(renderProfile_);
    const uint32_t frameIndex = frameRuntime_.currentFrame();
    const bool hasModel = currentModel_ && !currentModel_->isEmpty() && resources_.instanceBuffer();
    const uint32_t pointCount =
        hasModel
            ? static_cast<uint32_t>(std::distance(currentModel_->begin(), currentModel_->end()))
            : 0u;
    GraphicsRecordContext recordContext{
        frameRuntime_.commandBuffer(),
        pipelineSet_.renderPass(),
        frameRuntime_.framebuffer(*imageIndex),
        frameRuntime_.extent(),
        pipeline,
        resources_.graphicsDescriptorSet(frameIndex),
        resources_.computeDescriptorSet(frameIndex),
        sorter_,
        hasModel,
        pointCount,
        imguiInitialized_,
    };
    commandRecorder_.record(recordContext);
    frameRuntime_.submit(*imageIndex);
}

void GaussianRenderer::presentImage() {
    frameRuntime_.present(pipelineSet_.renderPass(), RenderPass::DepthFormat,
                          [this](vk::Format format, vk::Extent2D extent) {
                              return rebuildFormatDependentResources(format, extent);
                          });
}

void GaussianRenderer::renderToBuffer() {
    throw std::runtime_error(
        "GaussianRenderer does not support training renderToBuffer; use GaussianForwardRenderer");
}

void GaussianRenderer::onResize(uint32_t width, uint32_t height) {
    (void)width;
    (void)height;
    frameRuntime_.requestResize();
}

void GaussianRenderer::setPresentModePreference(PresentModePreference preference) {
    if (presentModePreference_ == preference) {
        return;
    }
    presentModePreference_ = preference;
    frameRuntime_.setPresentModePreference(preference);
}

void GaussianRenderer::setRenderData(const GaussianModel* model, const glm::mat4& view,
                                     const glm::mat4& projection, const Camera& camera,
                                     const glm::mat4& modelMatrix) {
    if (model != currentModel_) {
        const vk::Device device = getDevice();
        if (device) {
            device.waitIdle();
        }
        resources_.resetModel();
        sorter_.resetModelResources();
    }
    currentModel_ = model;
    camera_ = camera;
    for (GraphicsUniformData& uniform : uniforms_) {
        uniform.view = view;
        uniform.projection = projection;
        uniform.model = modelMatrix;
        const glm::mat3 rotation(view);
        const glm::vec3 translation(view[3][0], view[3][1], view[3][2]);
        uniform.cameraPositionTime =
            glm::vec4(-glm::transpose(rotation) * translation, static_cast<float>(glfwGetTime()));
    }
}

void GaussianRenderer::prepareFrameData() {
    sorter_.beginFrame();
    if (!currentModel_ || currentModel_->isEmpty()) {
        return;
    }

    resources_.ensureModelUploaded(*currentModel_, sorter_);
    const uint32_t frameIndex = frameRuntime_.currentFrame();
    GraphicsUniformData& uniform = uniforms_[frameIndex];
    const vk::Extent2D extent = frameRuntime_.extent();
    uniform.cameraPositionTime =
        glm::vec4(camera_.get_position(), static_cast<float>(glfwGetTime()));
    uniform.focal = glm::vec4(0.5f * static_cast<float>(extent.width) * uniform.projection[0][0],
                              0.5f * static_cast<float>(extent.height) * uniform.projection[1][1],
                              static_cast<float>(extent.width), static_cast<float>(extent.height));
    uniform.renderSettings =
        glm::uvec4(renderProfile_ == GaussianRenderProfile::SuperSplatCompatible ? 1u : 0u,
                   std::min(shBands_, 3u), 0u, 0u);
    resources_.updateUniform(uniform, extent);

    const uint32_t pointCount =
        static_cast<uint32_t>(std::distance(currentModel_->begin(), currentModel_->end()));
    const bool buffersChanged =
        sorter_.prepare(pointCount, currentModel_, uniform.view, uniform.projection, uniform.model,
                        frameRuntime_.inFlightFences());
    if (buffersChanged) {
        resources_.updateDescriptors(sorter_);
    }
}

vk::RenderPass GaussianRenderer::rebuildFormatDependentResources(vk::Format imageFormat,
                                                                 vk::Extent2D extent) {
    const bool rebuildImGui = imguiInitialized_;
    if (rebuildImGui) {
        shutdownImGui();
    }
    resources_.releaseDescriptors();
    pipelineSet_.recreate(imageFormat, extent);
    resources_.rebuildDescriptors(frameRuntime_.frameCount(), pipelineSet_.descriptorSetLayout(),
                                  sorter_.descriptorSetLayout(), sorter_);
    if (rebuildImGui) {
        initializeImGui(window_);
    }
    return pipelineSet_.renderPass();
}

} // namespace vulkan3DGS
