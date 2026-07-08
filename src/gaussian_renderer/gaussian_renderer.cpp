#include "gaussian_renderer.hpp"
#include "context/context.hpp"
#include "utils/logger.hpp"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <glm/gtc/packing.hpp>

namespace vulkan3DGS {

namespace {

bool matrixChanged(const glm::mat4& lhs, const glm::mat4& rhs, float epsilon) {
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            if (std::abs(lhs[col][row] - rhs[col][row]) > epsilon) {
                return true;
            }
        }
    }
    return false;
}

void loadImGuiFonts() {
    ImGuiIO& io = ImGui::GetIO();
    const char* fontCandidates[] = {
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    };

    for (const char* fontPath : fontCandidates) {
        if (std::filesystem::exists(fontPath)) {
            io.Fonts->AddFontFromFileTTF(fontPath, 16.0f, nullptr, io.Fonts->GetGlyphRangesChineseFull());
            LOG_INFO("Loaded ImGui CJK font: {}", fontPath);
            return;
        }
    }

    io.Fonts->AddFontDefault();
    LOG_WARN("No CJK font found for ImGui; Chinese text may not render correctly");
}

} // namespace

GaussianRenderer::GaussianRenderer() = default;

GaussianRenderer::~GaussianRenderer() {
    cleanup();
}

void GaussianRenderer::initialize(GLFWwindow* window) {
    LOG_INFO("Starting GaussianRenderer initialization");
    
    auto& context = Context::Instance();
    auto device = getDevice();
    auto surface = context.getSurface();
    
    // 初始化GPU排序缓存状态
    gpu_sort_completed_ = false;
    last_sorted_point_count_ = 0;
    last_sorted_model_ = nullptr;
    last_camera_position_ = glm::vec3(0.0f);
    last_view_matrix_ = glm::mat4(1.0f);
    last_projection_matrix_ = glm::mat4(1.0f);
    last_model_matrix_ = glm::mat4(1.0f);
    
    // 1. 获取窗口尺寸
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    
    // 2. 创建 Swapchain（只创建图像和ImageView）
    swapchain_ = std::make_unique<Swapchain>(surface);
    swapchain_->setPresentModePreference(presentModePreference_);
    swapchain_->createSwapchain(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
    swapchainImageCount_ = swapchain_->getImageCount();
    frameResourceCount_ = swapchainImageCount_;
    ubo_.resize(frameResourceCount_);
    
    // 3. 创建 RenderPass
    renderPass_ = std::make_unique<RenderPass>();
    renderPass_->initialize(swapchain_->getImageFormat());
    
    // 4. 创建 Framebuffers（需要RenderPass）
    swapchain_->createFramebuffers(device, renderPass_->getRenderPass(), RenderPass::DepthFormat);
    
    // 5. 创建 Pipeline
    pipeline_ = std::make_unique<Pipeline>();
    pipeline_->initialize(device, renderPass_->getRenderPass(), swapchain_->getExtent());
    
    // 6. 创建 Compute Pipeline
    createComputePipeline();
    
    // 7. 创建高斯特有资源
    createBuffers();
    
    createSyncObjects();
    initializeImGui(window);
    
    LOG_INFO("Gaussian Renderer initialized successfully");
}

void GaussianRenderer::cleanup() {
    auto device = getDevice();
    
    // 等待设备空闲
    if (device) {
        device.waitIdle();
    }

    shutdownImGui();
    
    // 1. 清理同步对象（Fences和Semaphores）
    for (vk::Semaphore semaphore : imageAvailableSemaphores_) {
        if (semaphore) {
            device.destroySemaphore(semaphore);
        }
    }
    for (vk::Fence fence : inFlightFences_) {
        if (fence) {
            device.destroyFence(fence);
        }
    }
    for (vk::Semaphore semaphore : renderFinishedSemaphores_) {
        if (semaphore) {
            device.destroySemaphore(semaphore);
        }
    }
    
    // 清空同步对象容器
    imageAvailableSemaphores_.clear();
    renderFinishedSemaphores_.clear();
    inFlightFences_.clear();
    ubo_.clear();
    frameResourceCount_ = 0;
    
    // 2. 清理Descriptor Pool（会自动销毁所有Descriptor Sets）
    destroyDescriptorPool();
    
    // 3. 清理缓冲区
    instanceBuffer_.cleanup();
    uniformBuffer_.cleanup();
    screenInfoBuffer_.cleanup();
    gpuIndexBuffer_.cleanup();
    gpuKeyBuffer_.cleanup();
    radixSortStorageBuffer_.cleanup();
    drawIndirectBuffer_.cleanup();
    sortBufferCapacity_ = 0;
    
    // 4. 清理Pipelines
    pipeline_.reset();
    radixKeygenPipeline_.reset();
    if (radixSorter_) {
        vrdxDestroySorter(radixSorter_);
        radixSorter_ = VK_NULL_HANDLE;
    }
    
    // 5. 清理命令池
    commandPool_.cleanup();
    
    
    // 6. 清理RenderPass和Swapchain
    renderPass_.reset();
    swapchain_.reset();
    
}

void GaussianRenderer::createComputePipeline() {
    auto device = getDevice();
    
    auto makeComputeBinding = [](uint32_t binding, vk::DescriptorType type) {
        vk::DescriptorSetLayoutBinding layoutBinding{};
        layoutBinding.setBinding(binding)
                     .setDescriptorType(type)
                     .setDescriptorCount(1)
                     .setStageFlags(vk::ShaderStageFlagBits::eCompute);
        return layoutBinding;
    };

    ComputePipelineConfig keygenConfig{};
    keygenConfig.descriptorBindings = {
        makeComputeBinding(0, vk::DescriptorType::eStorageBuffer), // indicesOut
        makeComputeBinding(1, vk::DescriptorType::eStorageBuffer), // keysOut
        makeComputeBinding(6, vk::DescriptorType::eStorageBuffer), // instances
        makeComputeBinding(7, vk::DescriptorType::eUniformBuffer), // ubo
        makeComputeBinding(8, vk::DescriptorType::eStorageBuffer), // drawArgs
    };
    keygenConfig.pushConstantSize = sizeof(uint32_t) * 4;

    radixKeygenPipeline_ = std::make_unique<ComputePipeline>();
    radixKeygenPipeline_->initialize(device, "shaders/radix_keygen.comp.spv", keygenConfig);

    auto& context = Context::Instance();
    VrdxSorterCreateInfo sorterInfo{};
    sorterInfo.physicalDevice = context.PhysicalDevice();
    sorterInfo.device = device;
    sorterInfo.pipelineCache = VK_NULL_HANDLE;
    vrdxCreateSorter(&sorterInfo, &radixSorter_);
    
    // 创建 Descriptor Sets
    createDescriptorSets();
    
}

void GaussianRenderer::initializeImGui(GLFWwindow* window) {
    if (imguiInitialized_) {
        return;
    }

    auto& context = Context::Instance();
    auto device = getDevice();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    loadImGuiFonts();

    if (!ImGui_ImplGlfw_InitForVulkan(window, true)) {
        LOG_ERROR("Failed to initialize ImGui GLFW backend");
        throw std::runtime_error("Failed to initialize ImGui GLFW backend");
    }

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_2;
    initInfo.Instance = context.getInstance();
    initInfo.PhysicalDevice = context.PhysicalDevice();
    initInfo.Device = device;
    initInfo.QueueFamily = context.getDevice().getQueueFamilyIndices().graphicsIndex.value();
    initInfo.Queue = context.getDevice().getGraphicsQueue();
    initInfo.DescriptorPool = VK_NULL_HANDLE;
    initInfo.DescriptorPoolSize = 64;
    initInfo.RenderPass = renderPass_->getRenderPass();
    initInfo.MinImageCount = swapchain_->getImageCount();
    initInfo.ImageCount = swapchain_->getImageCount();
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    if (!ImGui_ImplVulkan_Init(&initInfo)) {
        LOG_ERROR("Failed to initialize ImGui Vulkan backend");
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        throw std::runtime_error("Failed to initialize ImGui Vulkan backend");
    }

    ImGui_ImplVulkan_CreateFontsTexture();
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
    auto startTime = std::chrono::high_resolution_clock::now();
    (void)startTime;
    
    auto& context = Context::Instance();
    auto device = getDevice();
    record_sort_this_frame_ = false;
    sort_point_count_this_frame_ = 0;
    
    // 1. 等待上一帧完成
    vk::Result waitResult = device.waitForFences(1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);
    if (waitResult != vk::Result::eSuccess) {
        LOG_ERROR("waitForFences failed: %s", to_string(waitResult));
        throw std::runtime_error("Failed to wait for fence!");
    }
    
    // 2. 获取下一张交换链图像
    uint32_t imageIndex;
    vk::Result result = device.acquireNextImageKHR(swapchain_->getSwapchain(), UINT64_MAX, 
                                                   imageAvailableSemaphores_[currentFrame_], nullptr, &imageIndex);
    if (result != vk::Result::eSuccess){
        if (result == vk::Result::eErrorOutOfDateKHR) {
            LOG_WARN("Swapchain out of date, recreating");
            recreateSwapchain(swapchain_->getExtent().width, swapchain_->getExtent().height);
            return;
        } else if (result != vk::Result::eSuboptimalKHR) {
            LOG_ERROR("acquireNextImageKHR failed: %s", to_string(result));
            throw std::runtime_error("Failed to acquire swap chain image!");
        } else {
            LOG_INFO("Suboptimal swapchain detected, consider recreating");
        }
    }
    prepareFrameData(swapchain_->getExtent());

    beginImGuiFrame();
    
    // 6. 记录渲染命令
    recordCommandBuffer(imageIndex);
    
    // 7. 提交图形命令
    vk::SubmitInfo submitInfo{};
    vk::Semaphore waitSemaphores[] = { imageAvailableSemaphores_[currentFrame_] };
    vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
    submitInfo.setWaitSemaphoreCount(1)
              .setPWaitSemaphores(waitSemaphores)
              .setPWaitDstStageMask(waitStages);
    
    vk::CommandBuffer commandBuffers[] = { commandBuffers_[currentFrame_] };
    submitInfo.setCommandBufferCount(1)
              .setPCommandBuffers(commandBuffers);
    
    vk::Semaphore signalSemaphores[] = { renderFinishedSemaphores_[imageIndex] };
    submitInfo.setSignalSemaphoreCount(1)
              .setPSignalSemaphores(signalSemaphores);
    
    device.resetFences(inFlightFences_[currentFrame_]);
    
    context.getDevice().getGraphicsQueue().submit(submitInfo, inFlightFences_[currentFrame_]);

    acquiredImageIndex_ = imageIndex;
    imageReadyForPresent_ = true;
    presentWaitSemaphoreConsumed_ = false;
}

void GaussianRenderer::presentImage() {
    if (!imageReadyForPresent_) {
        return;
    }

    auto& context = Context::Instance();

    // 呈现图像
    vk::PresentInfoKHR presentInfo{};
    vk::Semaphore signalSemaphores[] = { renderFinishedSemaphores_[acquiredImageIndex_] };
    presentInfo.setWaitSemaphoreCount(presentWaitSemaphoreConsumed_ ? 0u : 1u)
               .setPWaitSemaphores(presentWaitSemaphoreConsumed_ ? nullptr : signalSemaphores);
    
    vk::SwapchainKHR swapchains[] = { swapchain_->getSwapchain() };
    presentInfo.setSwapchainCount(1)
               .setPSwapchains(swapchains);
    presentInfo.setPImageIndices(&acquiredImageIndex_);
    
    vk::Result presentResult = context.getDevice().getPresentQueue().presentKHR(presentInfo);
    if (presentResult == vk::Result::eErrorOutOfDateKHR) {
        LOG_WARN("Swapchain out of date during present, recreating");
        recreateSwapchain(swapchain_->getExtent().width, swapchain_->getExtent().height);
    } else if (presentResult == vk::Result::eSuboptimalKHR) {
        LOG_INFO("Suboptimal swapchain detected during present");
    } else if (presentResult != vk::Result::eSuccess) {
        LOG_ERROR("Failed to present image: {}", vk::to_string(presentResult));
    }
    
    imageReadyForPresent_ = false;
    presentWaitSemaphoreConsumed_ = false;
    currentFrame_ = (currentFrame_ + 1) % frameResourceCount_;
}

void GaussianRenderer::renderToBuffer() {
    throw std::runtime_error("GaussianRenderer does not support training renderToBuffer; use GaussianForwardRenderer");
}

void GaussianRenderer::onResize(uint32_t width, uint32_t height) {
    auto currentExtent = swapchain_->getExtent();
    if (currentExtent.width == width && currentExtent.height == height) {
        return;
    }
    recreateSwapchain(width, height);
}

void GaussianRenderer::setPresentModePreference(PresentModePreference preference) {
    if (presentModePreference_ == preference) {
        return;
    }

    presentModePreference_ = preference;
    if (!swapchain_) {
        return;
    }

    swapchain_->setPresentModePreference(preference);
    auto extent = swapchain_->getExtent();
    recreateSwapchain(extent.width, extent.height);
}

void GaussianRenderer::recreateSwapchain(uint32_t width, uint32_t height) {
    LOG_INFO("Recreating swapchain: {}x{}", width, height);
    
    auto device = getDevice();
    device.waitIdle();

    vk::Format oldImageFormat = swapchain_->getImageFormat();
    
    swapchain_->recreateSwapchain(width, height);
    recreateRenderFinishedSemaphores();

    if (oldImageFormat != swapchain_->getImageFormat()) {
        // Graphics descriptor sets were allocated from the old pipeline's set
        // layout. Destroy them before destroying/recreating that layout.
        destroyDescriptorPool();

        renderPass_->cleanup();
        renderPass_->initialize(swapchain_->getImageFormat());

        pipeline_->cleanup();
        pipeline_->initialize(device, renderPass_->getRenderPass(), swapchain_->getExtent());

        createDescriptorSets();
        updateDescriptorSets();
    }

    swapchain_->createFramebuffers(device, renderPass_->getRenderPass(), RenderPass::DepthFormat);
}

void GaussianRenderer::setRenderData(const GaussianModel* model, const glm::mat4& view, const glm::mat4& projection, const vulkan3DGS::Camera& camera, const glm::mat4& modelMatrix) {
    if (model != current_model_) {
        auto device = getDevice();
        if (device) {
            device.waitIdle();
        }

        instanceBuffer_.cleanup();
        gpuIndexBuffer_.cleanup();
        gpuKeyBuffer_.cleanup();
        radixSortStorageBuffer_.cleanup();
        drawIndirectBuffer_.cleanup();
        sortBufferCapacity_ = 0;
        gpu_sort_completed_ = false;
        last_sorted_point_count_ = 0;
        last_sorted_model_ = nullptr;
        last_model_matrix_ = glm::mat4(1.0f);
    }

    current_model_ = model;
    camera_ = camera;
    for (size_t i = 0u; i < ubo_.size(); ++i) {
        ubo_[i].view = view;
        ubo_[i].projection = projection;
        ubo_[i].model = modelMatrix;
        
        // 从视图矩阵正确提取相机世界空间位置
        // 视图矩阵 V = [R | t]，其中 t = -R * cameraPos
        // 所以 cameraPos = -R^T * t
        glm::mat3 rotation = glm::mat3(view);
        glm::vec3 translation = glm::vec3(view[3][0], view[3][1], view[3][2]);
        ubo_[i].cameraPositionTime = glm::vec4(
            -glm::transpose(rotation) * translation,
            static_cast<float>(glfwGetTime())
        );
    }
}

void GaussianRenderer::createBuffers() {
    auto& context = Context::Instance();
    auto device = getDevice();
    auto physicalDevice = context.PhysicalDevice();
    // Keep Gaussian resource uploads on the graphics queue. vkgs uses explicit
    // transfer semaphores/barriers; this renderer currently does not, so using
    // the same queue avoids cross-queue visibility issues for SSBO contents.
    auto transferQueue = context.getDevice().getGraphicsQueue();
    uint32_t transferQueueFamilyIndex = context.getDevice().getQueueFamilyIndices().graphicsIndex.value();
    
    // 1. 创建主 Uniform Buffer（每个帧一个）
    if (!uniformBuffer_.getBuffer()) {
        UniformBufferObject initialUBO{};
        uniformBuffer_.create(device, physicalDevice, transferQueue, transferQueueFamilyIndex,
                             &initialUBO, sizeof(UniformBufferObject),
                             vk::BufferUsageFlagBits::eUniformBuffer,
                             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
    
    // 2. 创建屏幕信息 Uniform Buffer
    if (!screenInfoBuffer_.getBuffer()) {
        struct ScreenInfoUBO {
            glm::vec2 screenSize;
            float padding[2]; // 填充到16字节对齐
        } initialScreenInfo{};
        screenInfoBuffer_.create(device, physicalDevice, transferQueue, transferQueueFamilyIndex,
                                &initialScreenInfo, sizeof(ScreenInfoUBO),
                                vk::BufferUsageFlagBits::eUniformBuffer,
                                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

void GaussianRenderer::createSyncObjects() {
    auto device = getDevice();
    
    // 获取Swapchain图像数量。presentation wait semaphore 必须按 imageIndex 复用。
    swapchainImageCount_ = swapchain_->getImageCount();
    frameResourceCount_ = swapchainImageCount_;
    
    imageAvailableSemaphores_.resize(frameResourceCount_);
    renderFinishedSemaphores_.resize(swapchainImageCount_);
    inFlightFences_.resize(frameResourceCount_);
    commandBuffers_.resize(frameResourceCount_);
    ubo_.resize(frameResourceCount_);
    
    vk::SemaphoreCreateInfo semaphoreInfo{};
    vk::FenceCreateInfo fenceInfo{};
    fenceInfo.setFlags(vk::FenceCreateFlagBits::eSignaled); // 初始状态为已信号
    
    for (size_t i = 0; i < frameResourceCount_; ++i) {
        imageAvailableSemaphores_[i] = device.createSemaphore(semaphoreInfo);
        inFlightFences_[i] = device.createFence(fenceInfo);
    }
    for (uint32_t i = 0; i < swapchainImageCount_; ++i) {
        renderFinishedSemaphores_[i] = device.createSemaphore(semaphoreInfo);
    }
    
    // 创建命令池
    auto& context = Context::Instance();
    uint32_t graphicsQueueFamily = context.getDevice().getQueueFamilyIndices().graphicsIndex.value();
    commandPool_.create(device, graphicsQueueFamily, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    
    // 分配命令缓冲区
    for (size_t i = 0; i < frameResourceCount_; ++i) {
        commandBuffers_[i] = commandPool_.allocateCommandBuffer();
    }
}

void GaussianRenderer::recreateRenderFinishedSemaphores() {
    auto device = getDevice();
    uint32_t newImageCount = swapchain_->getImageCount();
    if (swapchainImageCount_ == newImageCount &&
        renderFinishedSemaphores_.size() == newImageCount) {
        return;
    }

    for (vk::Semaphore semaphore : renderFinishedSemaphores_) {
        if (semaphore) {
            device.destroySemaphore(semaphore);
        }
    }

    swapchainImageCount_ = newImageCount;
    renderFinishedSemaphores_.clear();
    renderFinishedSemaphores_.resize(swapchainImageCount_);

    vk::SemaphoreCreateInfo semaphoreInfo{};
    for (uint32_t i = 0; i < swapchainImageCount_; ++i) {
        renderFinishedSemaphores_[i] = device.createSemaphore(semaphoreInfo);
    }
}

void GaussianRenderer::createDescriptorSets() {
    auto device = getDevice();

    if (frameResourceCount_ == 0 && swapchain_) {
        swapchainImageCount_ = swapchain_->getImageCount();
        frameResourceCount_ = swapchainImageCount_;
        ubo_.resize(frameResourceCount_);
    }
    if (frameResourceCount_ == 0) {
        throw std::runtime_error("Cannot create descriptor sets before frame resources are initialized");
    }
    
    destroyDescriptorPool();
    
    // 创建 Descriptor Pool（支持 Uniform Buffer 和 Storage Buffer）
    std::array<vk::DescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].setType(vk::DescriptorType::eUniformBuffer)
                .setDescriptorCount(frameResourceCount_ * 4);
    poolSizes[1].setType(vk::DescriptorType::eStorageBuffer)
                .setDescriptorCount(20 * frameResourceCount_);
    
    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.setPoolSizeCount(static_cast<uint32_t>(poolSizes.size()))
            .setPPoolSizes(poolSizes.data())
            .setMaxSets(frameResourceCount_ * 3);
    
    descriptorPool_ = device.createDescriptorPool(poolInfo);
    
    // 分配 Graphics Descriptor Sets（每帧一个，使用图形管线的Descriptor Set Layout）
    std::vector<vk::DescriptorSetLayout> graphicsLayouts(frameResourceCount_, pipeline_->getDescriptorSetLayout());
    
    vk::DescriptorSetAllocateInfo graphicsAllocInfo{};
    graphicsAllocInfo.setDescriptorPool(descriptorPool_)
                     .setDescriptorSetCount(static_cast<uint32_t>(graphicsLayouts.size()))
                     .setPSetLayouts(graphicsLayouts.data());
    
    descriptorSets_ = device.allocateDescriptorSets(graphicsAllocInfo);
    
    // 每帧两套 Compute Descriptor Sets：A->B 和 B->A。
    std::vector<vk::DescriptorSetLayout> computeLayouts(frameResourceCount_ * 2, radixKeygenPipeline_->getDescriptorSetLayout());
    
    vk::DescriptorSetAllocateInfo computeAllocInfo{};
    computeAllocInfo.setDescriptorPool(descriptorPool_)
                    .setDescriptorSetCount(static_cast<uint32_t>(computeLayouts.size()))
                    .setPSetLayouts(computeLayouts.data());
    
    computeDescriptorSets_ = device.allocateDescriptorSets(computeAllocInfo);
    
    LOG_DEBUG("Descriptor sets created successfully (Graphics: {}, Compute: {})", 
              descriptorSets_.size(), computeDescriptorSets_.size());
}

void GaussianRenderer::destroyDescriptorPool() {
    if (!descriptorPool_) {
        return;
    }

    auto device = getDevice();
    device.destroyDescriptorPool(descriptorPool_);
    descriptorPool_ = nullptr;
    descriptorSets_.clear();
    computeDescriptorSets_.clear();
}

void GaussianRenderer::updateDescriptorSets() {
    auto device = getDevice();
    
    // 为每个 Graphics Descriptor Set 更新 Uniform Buffer 和 SSBO 绑定
    for (size_t i = 0; i < descriptorSets_.size(); ++i) {
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
        writeDescriptorSets.reserve(4);
        
        // Binding 0: 主Uniform Buffer (View/Projection/Camera)
        vk::DescriptorBufferInfo uboInfo{};
        if (uniformBuffer_.getBuffer()) {
            uboInfo.setBuffer(uniformBuffer_.getBuffer())
                   .setOffset(0)
                   .setRange(sizeof(UniformBufferObject));
            
            vk::WriteDescriptorSet write{};
            write.setDstSet(descriptorSets_[i])
                 .setDstBinding(0)
                 .setDstArrayElement(0)
                 .setDescriptorCount(1)
                 .setDescriptorType(vk::DescriptorType::eUniformBuffer)
                 .setPBufferInfo(&uboInfo);
            writeDescriptorSets.push_back(write);
        }
        
        // Binding 1: 屏幕信息Uniform Buffer
        vk::DescriptorBufferInfo screenInfo{};
        if (screenInfoBuffer_.getBuffer()) {
            screenInfo.setBuffer(screenInfoBuffer_.getBuffer())
                      .setOffset(0)
                      .setRange(sizeof(glm::vec2) + sizeof(float) * 2); // screenSize + padding
            
            vk::WriteDescriptorSet write{};
            write.setDstSet(descriptorSets_[i])
                 .setDstBinding(1)
                 .setDstArrayElement(0)
                 .setDescriptorCount(1)
                 .setDescriptorType(vk::DescriptorType::eUniformBuffer)
                 .setPBufferInfo(&screenInfo);
            writeDescriptorSets.push_back(write);
        }
        
        // Binding 2: 高斯实例数据 SSBO
        vk::DescriptorBufferInfo instanceSSBO{};
        if (instanceBuffer_.getBuffer()) {
            instanceSSBO.setBuffer(instanceBuffer_.getBuffer())
                        .setOffset(0)
                        .setRange(VK_WHOLE_SIZE);
            
            vk::WriteDescriptorSet write{};
            write.setDstSet(descriptorSets_[i])
                 .setDstBinding(2)
                 .setDstArrayElement(0)
                 .setDescriptorCount(1)
                 .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                 .setPBufferInfo(&instanceSSBO);
            writeDescriptorSets.push_back(write);
        }

        // Binding 3: 排序后的实例索引 SSBO
        vk::DescriptorBufferInfo sortedIndexSSBO{};
        if (gpuIndexBuffer_.getBuffer()) {
            sortedIndexSSBO.setBuffer(gpuIndexBuffer_.getBuffer())
                           .setOffset(0)
                           .setRange(VK_WHOLE_SIZE);
            
            vk::WriteDescriptorSet write{};
            write.setDstSet(descriptorSets_[i])
                 .setDstBinding(3)
                 .setDstArrayElement(0)
                 .setDescriptorCount(1)
                 .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                 .setPBufferInfo(&sortedIndexSSBO);
            writeDescriptorSets.push_back(write);
        }
        
        if (!writeDescriptorSets.empty()) {
            device.updateDescriptorSets(static_cast<uint32_t>(writeDescriptorSets.size()), 
                                       writeDescriptorSets.data(), 
                                       0, nullptr);
        }
    }
    
    if (!gpuIndexBuffer_.getBuffer() ||
        !gpuKeyBuffer_.getBuffer() ||
        !drawIndirectBuffer_.getBuffer() ||
        !instanceBuffer_.getBuffer() ||
        !uniformBuffer_.getBuffer()) {
        LOG_DEBUG("Skipping radix descriptor update until sort buffers exist");
        return;
    }

    for (size_t i = 0; i < computeDescriptorSets_.size(); ++i) {
        std::array<vk::DescriptorBufferInfo, 9> bufferInfos{};
        bufferInfos[0].setBuffer(gpuIndexBuffer_.getBuffer()).setOffset(0).setRange(VK_WHOLE_SIZE);
        bufferInfos[1].setBuffer(gpuKeyBuffer_.getBuffer()).setOffset(0).setRange(VK_WHOLE_SIZE);
        bufferInfos[6].setBuffer(instanceBuffer_.getBuffer()).setOffset(0).setRange(VK_WHOLE_SIZE);
        bufferInfos[7].setBuffer(uniformBuffer_.getBuffer()).setOffset(0).setRange(sizeof(UniformBufferObject));
        bufferInfos[8].setBuffer(drawIndirectBuffer_.getBuffer()).setOffset(0).setRange(sizeof(VkDrawIndexedIndirectCommand));

        const std::array<uint32_t, 5> bindings = {0, 1, 6, 7, 8};
        std::array<vk::WriteDescriptorSet, bindings.size()> writeDescriptorSets{};

        for (size_t writeIndex = 0; writeIndex < bindings.size(); ++writeIndex) {
            uint32_t binding = bindings[writeIndex];
            writeDescriptorSets[writeIndex].setDstSet(computeDescriptorSets_[i])
                                           .setDstBinding(binding)
                                           .setDstArrayElement(0)
                                           .setDescriptorCount(1)
                                           .setDescriptorType(binding == 7 ? vk::DescriptorType::eUniformBuffer : vk::DescriptorType::eStorageBuffer)
                                           .setPBufferInfo(&bufferInfos[binding]);
        }
        
        device.updateDescriptorSets(static_cast<uint32_t>(writeDescriptorSets.size()), 
                                   writeDescriptorSets.data(), 
                                   0, nullptr);
    }
    
    LOG_DEBUG("Descriptor sets updated (Graphics: {}, Compute: {})", 
              descriptorSets_.size(), computeDescriptorSets_.size());
}

void GaussianRenderer::recordSortCommands(vk::CommandBuffer commandBuffer, uint32_t pointCount) {
    if (!current_model_ || current_model_->isEmpty()) {
        LOG_WARN("Skipping sort: no model or empty");
        return;
    }
    
    LOG_DEBUG("Sorting {} points", pointCount);
    
    uint32_t groupCount = (pointCount + 255u) / 256u;

    commandBuffer.fillBuffer(drawIndirectBuffer_.getBuffer(), 0, sizeof(VkDrawIndexedIndirectCommand), 0);

    vk::MemoryBarrier drawClearBarrier{};
    drawClearBarrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                    .setDstAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits{},
        1, &drawClearBarrier,
        0, nullptr,
        0, nullptr
    );

    struct PushConstants {
        uint32_t count;
        uint32_t shift;
        uint32_t groupCount;
        uint32_t padding;
    } pushConstants{};

    pushConstants.count = pointCount;
    pushConstants.groupCount = groupCount;

    vk::DescriptorSet keygenSet = computeDescriptorSets_[currentFrame_ * 2];
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, radixKeygenPipeline_->getPipeline());
    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        radixKeygenPipeline_->getPipelineLayout(),
        0,
        1,
        &keygenSet,
        0,
        nullptr
    );
    commandBuffer.pushConstants(
        radixKeygenPipeline_->getPipelineLayout(),
        vk::ShaderStageFlagBits::eCompute,
        0, sizeof(PushConstants), &pushConstants
    );
    commandBuffer.dispatch(groupCount, 1, 1);

    vk::MemoryBarrier sortInputBarrier{};
    sortInputBarrier.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                    .setDstAccessMask(vk::AccessFlagBits::eShaderRead |
                                      vk::AccessFlagBits::eShaderWrite |
                                      vk::AccessFlagBits::eTransferRead);
    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
        vk::DependencyFlagBits{},
        1, &sortInputBarrier,
        0, nullptr,
        0, nullptr
    );

    vrdxCmdSortKeyValueIndirect(
        commandBuffer,
        radixSorter_,
        pointCount,
        drawIndirectBuffer_.getBuffer(),
        sizeof(uint32_t),
        gpuKeyBuffer_.getBuffer(),
        0,
        gpuIndexBuffer_.getBuffer(),
        0,
        radixSortStorageBuffer_.getBuffer(),
        0,
        VK_NULL_HANDLE,
        0
    );
}

void GaussianRenderer::ensureSortBuffers(uint32_t pointCount) {
    if (pointCount == 0 || sortBufferCapacity_ == pointCount) {
        return;
    }
    
    auto device = getDevice();
    auto& context = Context::Instance();
    auto physicalDevice = context.PhysicalDevice();
    auto transferQueue = context.getDevice().getGraphicsQueue();
    uint32_t transferQueueFamilyIndex = context.getDevice().getQueueFamilyIndices().graphicsIndex.value();
    
    if (gpuIndexBuffer_.getBuffer() ||
        gpuKeyBuffer_.getBuffer() ||
        radixSortStorageBuffer_.getBuffer() ||
        drawIndirectBuffer_.getBuffer()) {
        std::vector<vk::Fence> fences(frameResourceCount_);
        for (size_t frame = 0; frame < frameResourceCount_; ++frame) {
            fences[frame] = inFlightFences_[frame];
        }
        (void)device.waitForFences(fences, VK_TRUE, UINT64_MAX);
    }

    gpuIndexBuffer_.cleanup();
    gpuKeyBuffer_.cleanup();
    radixSortStorageBuffer_.cleanup();
    drawIndirectBuffer_.cleanup();
    
    gpuIndexBuffer_.create(device, physicalDevice, transferQueue, transferQueueFamilyIndex,
                          nullptr,
                          static_cast<vk::DeviceSize>(pointCount) * sizeof(uint32_t),
                          vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                          vk::MemoryPropertyFlagBits::eDeviceLocal);
    
    gpuKeyBuffer_.create(device, physicalDevice, transferQueue, transferQueueFamilyIndex,
                         nullptr,
                         static_cast<vk::DeviceSize>(pointCount) * sizeof(uint32_t),
                         vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                         vk::MemoryPropertyFlagBits::eDeviceLocal);

    VrdxSorterStorageRequirements sorterRequirements{};
    vrdxGetSorterKeyValueStorageRequirements(radixSorter_, pointCount, &sorterRequirements);
    radixSortStorageBuffer_.create(device, physicalDevice, transferQueue, transferQueueFamilyIndex,
                                   nullptr,
                                   sorterRequirements.size,
                                   vk::BufferUsageFlags(sorterRequirements.usage),
                                   vk::MemoryPropertyFlagBits::eDeviceLocal);

    drawIndirectBuffer_.create(device, physicalDevice, transferQueue, transferQueueFamilyIndex,
                               nullptr,
                               sizeof(VkDrawIndexedIndirectCommand),
                               vk::BufferUsageFlagBits::eStorageBuffer |
                                   vk::BufferUsageFlagBits::eIndirectBuffer |
                                   vk::BufferUsageFlagBits::eTransferSrc |
                                   vk::BufferUsageFlagBits::eTransferDst,
                               vk::MemoryPropertyFlagBits::eDeviceLocal);
    
    sortBufferCapacity_ = pointCount;
    updateDescriptorSets();
}

void GaussianRenderer::updateVertexBuffer() {
    if (!current_model_ || current_model_->isEmpty()) {
        return;
    }
    
    auto device = getDevice();
    auto& context = Context::Instance();
    auto physicalDevice = context.PhysicalDevice();
    auto transferQueue = context.getDevice().getGraphicsQueue();
    uint32_t transferQueueFamilyIndex = context.getDevice().getQueueFamilyIndices().graphicsIndex.value();
    
    // 构建实例数据数组（SSBO格式 - 包含完整SH系数）
    struct alignas(8) PackedSH {
        uint32_t xy;
        uint32_t z0;
    };
    static_assert(sizeof(PackedSH) == sizeof(uint32_t) * 2,
                  "PackedSH must match GLSL uvec2 layout");

    struct alignas(16) GaussianInstanceData {
        glm::vec4 position;     // xyz: position
        glm::vec4 scale;        // xyz: scale
        glm::vec4 rotation;     // xyzw: quaternion

        PackedSH sh0;           // half-packed xyz DC term
        PackedSH sh1[3];        // half-packed xyz 1st-order SH
        PackedSH sh2[5];        // half-packed xyz 2nd-order SH
        PackedSH sh3[7];        // half-packed xyz 3rd-order SH
    };
    static_assert(sizeof(GaussianInstanceData) == sizeof(glm::vec4) * 3 + sizeof(PackedSH) * 16,
                  "GaussianInstanceData must match the GLSL std430 packed layout");

    auto packSH = [](const glm::vec3& value) {
        PackedSH packed{};
        packed.xy = glm::packHalf2x16(glm::vec2(value.x, value.y));
        packed.z0 = glm::packHalf2x16(glm::vec2(value.z, 0.0f));
        return packed;
    };
    
    uint32_t pointCount = static_cast<uint32_t>(std::distance(current_model_->begin(), current_model_->end()));
    std::vector<GaussianInstanceData> instanceData(pointCount);
    
    size_t idx = 0;
    for (const auto& point : *current_model_) {
        instanceData[idx].position = glm::vec4(point.position, point.alpha);
        instanceData[idx].scale = glm::vec4(point.scale, 0.0f);
        instanceData[idx].rotation = glm::vec4(
            point.rotation.x,
            point.rotation.y,
            point.rotation.z,
            point.rotation.w
        );
        
        // Half-pack SH coefficients to reduce SSBO bandwidth.
        instanceData[idx].sh0 = packSH(point.color.sh0);
        for (int i = 0; i < 3; ++i) {
            instanceData[idx].sh1[i] = packSH(point.color.sh1[i]);
        }
        for (int i = 0; i < 5; ++i) {
            instanceData[idx].sh2[i] = packSH(point.color.sh2[i]);
        }
        for (int i = 0; i < 7; ++i) {
            instanceData[idx].sh3[i] = packSH(point.color.sh3[i]);
        }
        ++idx;
    }
    
    // 创建或更新实例缓冲区（SSBO）
    if (!instanceBuffer_.getBuffer()) {
        size_t bufferSize = instanceData.size() * sizeof(GaussianInstanceData);
        instanceBuffer_.create(device, physicalDevice, transferQueue, transferQueueFamilyIndex,
                              instanceData.data(),
                              bufferSize,
                              vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                              vk::MemoryPropertyFlagBits::eDeviceLocal);
        std::ostringstream sizeMb;
        sizeMb << std::fixed << std::setprecision(2) << bufferSize / (1024.0 * 1024.0);
        LOG_INFO("Created instance SSBO with {} instances ({} bytes, {} MB)",
                 pointCount, bufferSize, sizeMb.str());
        
        // 首次创建后需要更新Descriptor Set
        updateDescriptorSets();
    } else {
        // TODO: 实现动态更新逻辑（使用 staging buffer）
        LOG_WARN("Instance SSBO update not yet implemented, using initial data");
    }
}

void GaussianRenderer::prepareFrameData(vk::Extent2D extent) {
    record_sort_this_frame_ = false;
    sort_point_count_this_frame_ = 0;

    if (!current_model_ || current_model_->isEmpty()) {
        return;
    }

    if (!instanceBuffer_.getBuffer()) {
        updateVertexBuffer();
    }

    updateUniformBuffer(ubo_[currentFrame_].view, ubo_[currentFrame_].projection, extent);

    uint32_t currentPointCount = static_cast<uint32_t>(std::distance(current_model_->begin(), current_model_->end()));
    const float matrixEpsilon = 1e-5f;
    bool needResort = !gpu_sort_completed_ ||
                      current_model_ != last_sorted_model_ ||
                      currentPointCount != last_sorted_point_count_ ||
                      matrixChanged(ubo_[currentFrame_].view, last_view_matrix_, matrixEpsilon) ||
                      matrixChanged(ubo_[currentFrame_].projection, last_projection_matrix_, matrixEpsilon) ||
                      matrixChanged(ubo_[currentFrame_].model, last_model_matrix_, matrixEpsilon);

    if (!needResort) {
        return;
    }

    ensureSortBuffers(currentPointCount);
    record_sort_this_frame_ = true;
    sort_point_count_this_frame_ = currentPointCount;

    gpu_sort_completed_ = true;
    last_sorted_point_count_ = currentPointCount;
    last_sorted_model_ = current_model_;
    last_camera_position_ = camera_.get_position();
    last_view_matrix_ = ubo_[currentFrame_].view;
    last_projection_matrix_ = ubo_[currentFrame_].projection;
    last_model_matrix_ = ubo_[currentFrame_].model;
}

void GaussianRenderer::updateUniformBuffer(const glm::mat4& view, const glm::mat4& projection, vk::Extent2D extent) {
    auto device = getDevice();
    
    // 1. 更新主Uniform Buffer
    ubo_[currentFrame_].view = view;
    ubo_[currentFrame_].projection = projection;
    ubo_[currentFrame_].cameraPositionTime = glm::vec4(
        camera_.get_position(),
        static_cast<float>(glfwGetTime())
    );
    
    ubo_[currentFrame_].focal = glm::vec4(
        0.5f * static_cast<float>(extent.width) * projection[0][0],
        0.5f * static_cast<float>(extent.height) * projection[1][1],
        static_cast<float>(extent.width),
        static_cast<float>(extent.height)
    );
    
    std::ostringstream fx;
    std::ostringstream fy;
    fx << std::fixed << std::setprecision(2) << ubo_[currentFrame_].focal.x;
    fy << std::fixed << std::setprecision(2) << ubo_[currentFrame_].focal.y;
    LOG_DEBUG("Pixel focal lengths: fx={}, fy={}, screen={}x{}",
              fx.str(), fy.str(), extent.width, extent.height);
    
    void* data = device.mapMemory(uniformBuffer_.getMemory(), 0, sizeof(UniformBufferObject));
    std::memcpy(data, &ubo_[currentFrame_], sizeof(UniformBufferObject));
    device.unmapMemory(uniformBuffer_.getMemory());
    
    // 2. 更新屏幕信息Uniform Buffer（用于片段着色器）
    struct ScreenInfoUBO {
        glm::vec2 screenSize;
        float padding[2];
    } screenInfo{};
    screenInfo.screenSize = glm::vec2(static_cast<float>(extent.width), static_cast<float>(extent.height));
    
    data = device.mapMemory(screenInfoBuffer_.getMemory(), 0, sizeof(ScreenInfoUBO));
    std::memcpy(data, &screenInfo, sizeof(ScreenInfoUBO));
    device.unmapMemory(screenInfoBuffer_.getMemory());
}

void GaussianRenderer::recordCommandBuffer(uint32_t image_index) {
    auto commandBuffer = commandBuffers_[currentFrame_];
    
    // 开始记录命令缓冲区
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    commandBuffer.begin(beginInfo);
    recordRenderCommands(commandBuffer,
                         renderPass_->getRenderPass(),
                         swapchain_->getFramebuffer(image_index),
                         swapchain_->getExtent(),
                         *pipeline_,
                         true);
    commandBuffer.end();
}

void GaussianRenderer::recordRenderCommands(vk::CommandBuffer commandBuffer,
                                            vk::RenderPass renderPass,
                                            vk::Framebuffer framebuffer,
                                            vk::Extent2D extent,
                                            Pipeline& pipeline,
                                            bool drawImGui) {
    if (record_sort_this_frame_ && sort_point_count_this_frame_ > 0) {
        recordSortCommands(commandBuffer, sort_point_count_this_frame_);
    }

    if (gpuIndexBuffer_.getBuffer()) {
        vk::MemoryBarrier sortedIndexBarrier{};
        sortedIndexBarrier.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                          .setDstAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eIndirectCommandRead);

        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eDrawIndirect,
            vk::DependencyFlagBits{},
            1, &sortedIndexBarrier,
            0, nullptr,
            0, nullptr
        );
    }

    std::array<vk::ClearValue, 2> clearValues{};
    clearValues[0].setColor(vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}));
    clearValues[1].setDepthStencil(vk::ClearDepthStencilValue(1.0f, 0));

    vk::RenderPassBeginInfo renderPassInfo{};
    renderPassInfo.setRenderPass(renderPass)
                  .setFramebuffer(framebuffer)
                  .setRenderArea(vk::Rect2D({0, 0}, extent))
                  .setClearValueCount(static_cast<uint32_t>(clearValues.size()))
                  .setPClearValues(clearValues.data());

    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

    vk::Viewport viewport(0.0f, 0.0f,
                          static_cast<float>(extent.width),
                          static_cast<float>(extent.height),
                          0.0f, 1.0f);
    commandBuffer.setViewport(0, 1, &viewport);

    vk::Rect2D scissor({0, 0}, extent);
    commandBuffer.setScissor(0, 1, &scissor);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.getPipeline());

    vk::Buffer vertexBuffers[] = { pipeline.getQuadVertexBuffer() };
    vk::DeviceSize offsets[] = { 0 };
    commandBuffer.bindVertexBuffers(0, 1, vertexBuffers, offsets);
    commandBuffer.bindIndexBuffer(pipeline.getQuadIndexBuffer(), 0, vk::IndexType::eUint16);

    if (!descriptorSets_.empty()) {
        commandBuffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            pipeline.getPipelineLayout(),
            0,
            1,
            &descriptorSets_[currentFrame_],
            0,
            nullptr
        );
    }

    if (current_model_ && instanceBuffer_.getBuffer()) {
        if (drawIndirectBuffer_.getBuffer()) {
            commandBuffer.drawIndexedIndirect(
                drawIndirectBuffer_.getBuffer(),
                0,
                1,
                sizeof(VkDrawIndexedIndirectCommand)
            );
        } else {
            uint32_t instanceCount = static_cast<uint32_t>(std::distance(current_model_->begin(), current_model_->end()));
            commandBuffer.drawIndexed(4, instanceCount, 0, 0, 0);
        }
    }

    if (drawImGui && imguiInitialized_) {
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
    }

    commandBuffer.endRenderPass();
}

} // namespace vulkan3DGS
