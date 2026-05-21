#include "gaussian_renderer.hpp"
#include "context/context.hpp"
#include "utils/logger.hpp"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>

namespace vk_gs {

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
    swapchain_->createSwapchain(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
    
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
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (imageAvailableSemaphores_.size() > i && imageAvailableSemaphores_[i]) {
            device.destroySemaphore(imageAvailableSemaphores_[i]);
        }
        if (renderFinishedSemaphores_.size() > i && renderFinishedSemaphores_[i]) {
            device.destroySemaphore(renderFinishedSemaphores_[i]);
        }
        if (inFlightFences_.size() > i && inFlightFences_[i]) {
            device.destroyFence(inFlightFences_[i]);
        }
    }
    
    // 清空同步对象容器
    imageAvailableSemaphores_.clear();
    renderFinishedSemaphores_.clear();
    inFlightFences_.clear();
    
    // 2. 清理Descriptor Pool（会自动销毁所有Descriptor Sets）
    destroyDescriptorPool();
    
    // 3. 清理缓冲区
    instanceBuffer_.cleanup();
    uniformBuffer_.cleanup();
    screenInfoBuffer_.cleanup();
    gpuIndexBuffer_.cleanup();
    gpuKeyBuffer_.cleanup();
    gpuIndexTempBuffer_.cleanup();
    gpuKeyTempBuffer_.cleanup();
    radixHistogramBuffer_.cleanup();
    radixOffsetBuffer_.cleanup();
    sortBufferCapacity_ = 0;
    
    // 4. 清理Pipelines
    pipeline_.reset();
    radixKeygenPipeline_.reset();
    radixHistogramPipeline_.reset();
    radixPrefixPipeline_.reset();
    radixScatterPipeline_.reset();
    
    // 5. 清理命令池
    commandPool_.cleanup();
    computeCommandPool_.cleanup();
    
    // 6. 清理RenderPass和Swapchain
    renderPass_.reset();
    swapchain_.reset();
    
}

void GaussianRenderer::createComputePipeline() {
    auto device = getDevice();
    
    radixKeygenPipeline_ = std::make_unique<ComputePipeline>();
    radixKeygenPipeline_->initialize(device, "shaders/radix_keygen.comp.spv");

    radixHistogramPipeline_ = std::make_unique<ComputePipeline>();
    radixHistogramPipeline_->initialize(device, "shaders/radix_histogram.comp.spv");

    radixPrefixPipeline_ = std::make_unique<ComputePipeline>();
    radixPrefixPipeline_->initialize(device, "shaders/radix_prefix.comp.spv");

    radixScatterPipeline_ = std::make_unique<ComputePipeline>();
    radixScatterPipeline_->initialize(device, "shaders/radix_scatter.comp.spv");
    
    // 创建 Descriptor Sets
    createDescriptorSets();
    
    // Keep sorting on the graphics queue. vkgs records its rank/sort/projection
    // work in the graphics submission chain before drawing; using a separate
    // compute queue here needs semaphore-based cross-queue memory dependencies,
    // otherwise the sorted index buffer can be stale when the splat pass reads it.
    auto& context = Context::Instance();
    uint32_t computeQueueFamily = context.getDevice().getQueueFamilyIndices().graphicsIndex.value();
    computeCommandPool_.create(device, computeQueueFamily, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    computeCommandBuffer_ = computeCommandPool_.allocateCommandBuffer();
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
    initInfo.MinImageCount = MAX_FRAMES_IN_FLIGHT;
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
    auto startTime = std::chrono::high_resolution_clock::now();
    
    auto& context = Context::Instance();
    auto device = getDevice();
    
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
    if (current_model_ && !current_model_->isEmpty()) {
        // 3. 更新Instance Buffer（仅在首次或模型变化时重建）
        if (!instanceBuffer_.getBuffer()) {
            updateVertexBuffer();
        }

        // 4. 更新Uniform Buffer，GPU key generation reads it in the sort pass.
        updateUniformBuffer(ubo_[currentFrame_].view, ubo_[currentFrame_].projection);

        // 5. GPU排序。透明Gaussian需要随相机变化保持远到近顺序。
        uint32_t currentPointCount = static_cast<uint32_t>(std::distance(current_model_->begin(), current_model_->end()));
        
        const float matrixEpsilon = 1e-5f;
        bool needResort = !gpu_sort_completed_ ||
                          current_model_ != last_sorted_model_ ||
                          currentPointCount != last_sorted_point_count_ ||
                          matrixChanged(ubo_[currentFrame_].view, last_view_matrix_, matrixEpsilon) ||
                          matrixChanged(ubo_[currentFrame_].projection, last_projection_matrix_, matrixEpsilon) ||
                          matrixChanged(ubo_[currentFrame_].model, last_model_matrix_, matrixEpsilon);
        
        if (needResort) {
            sortGaussiansByDepthGPU();
            
            gpu_sort_completed_ = true;
            last_sorted_point_count_ = currentPointCount;
            last_sorted_model_ = current_model_;
            last_camera_position_ = camera_.get_position();
            last_view_matrix_ = ubo_[currentFrame_].view;
            last_projection_matrix_ = ubo_[currentFrame_].projection;
            last_model_matrix_ = ubo_[currentFrame_].model;
        }
    }

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
    
    vk::Semaphore signalSemaphores[] = { renderFinishedSemaphores_[currentFrame_] };
    submitInfo.setSignalSemaphoreCount(1)
              .setPSignalSemaphores(signalSemaphores);
    
    device.resetFences(inFlightFences_[currentFrame_]);
    
    context.getDevice().getGraphicsQueue().submit(submitInfo, inFlightFences_[currentFrame_]);
    
    // 8. 呈现图像
    vk::PresentInfoKHR presentInfo{};
    presentInfo.setWaitSemaphoreCount(1)
               .setPWaitSemaphores(signalSemaphores);
    
    vk::SwapchainKHR swapchains[] = { swapchain_->getSwapchain() };
    presentInfo.setSwapchainCount(1)
               .setPSwapchains(swapchains);
    presentInfo.setPImageIndices(&imageIndex);
    
    vk::Result presentResult = context.getDevice().getPresentQueue().presentKHR(presentInfo);
    if (presentResult != vk::Result::eSuccess) {
        LOG_ERROR("Failed to present image");
    }
    
    // 9. 切换到下一帧
    currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

void GaussianRenderer::onResize(uint32_t width, uint32_t height) {
    auto currentExtent = swapchain_->getExtent();
    if (currentExtent.width == width && currentExtent.height == height) {
        return;
    }
    recreateSwapchain(width, height);
}

void GaussianRenderer::recreateSwapchain(uint32_t width, uint32_t height) {
    LOG_INFO("Recreating swapchain: {}x{}", width, height);
    
    auto device = getDevice();
    device.waitIdle();
    
    // Graphics descriptor sets were allocated from the old pipeline's set
    // layout. Destroy them before destroying/recreating that layout.
    destroyDescriptorPool();
    
    swapchain_->recreateSwapchain(width, height);
    
    renderPass_->cleanup();
    renderPass_->initialize(swapchain_->getImageFormat());
    swapchain_->createFramebuffers(device, renderPass_->getRenderPass(), RenderPass::DepthFormat);
    
    pipeline_->cleanup();
    pipeline_->initialize(device, renderPass_->getRenderPass(), swapchain_->getExtent());
    
    createDescriptorSets();
    updateDescriptorSets();
}

void GaussianRenderer::setRenderData(const GaussianModel* model, const glm::mat4& view, const glm::mat4& projection, const vk_gs::Camera& camera, const glm::mat4& modelMatrix) {
    if (model != current_model_) {
        auto device = getDevice();
        if (device) {
            device.waitIdle();
        }

        instanceBuffer_.cleanup();
        gpuIndexBuffer_.cleanup();
        gpuKeyBuffer_.cleanup();
        gpuIndexTempBuffer_.cleanup();
        gpuKeyTempBuffer_.cleanup();
        radixHistogramBuffer_.cleanup();
        radixOffsetBuffer_.cleanup();
        sortBufferCapacity_ = 0;
        gpu_sort_completed_ = false;
        last_sorted_point_count_ = 0;
        last_sorted_model_ = nullptr;
        last_model_matrix_ = glm::mat4(1.0f);
    }

    current_model_ = model;
    camera_ = camera;
    for (uint32_t i = 0u; i < MAX_FRAMES_IN_FLIGHT; ++i) {
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
    
    // 获取Swapchain图像数量（仅用于日志）
    swapchainImageCount_ = swapchain_->getImageCount();
    
    // 所有同步对象都按MAX_FRAMES_IN_FLIGHT创建
    imageAvailableSemaphores_.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores_.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences_.resize(MAX_FRAMES_IN_FLIGHT);
    commandBuffers_.resize(MAX_FRAMES_IN_FLIGHT);
    
    vk::SemaphoreCreateInfo semaphoreInfo{};
    vk::FenceCreateInfo fenceInfo{};
    fenceInfo.setFlags(vk::FenceCreateFlagBits::eSignaled); // 初始状态为已信号
    
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        imageAvailableSemaphores_[i] = device.createSemaphore(semaphoreInfo);
        renderFinishedSemaphores_[i] = device.createSemaphore(semaphoreInfo);
        inFlightFences_[i] = device.createFence(fenceInfo);
    }
    
    // 创建命令池
    auto& context = Context::Instance();
    uint32_t graphicsQueueFamily = context.getDevice().getQueueFamilyIndices().graphicsIndex.value();
    commandPool_.create(device, graphicsQueueFamily, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    
    // 分配命令缓冲区
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        commandBuffers_[i] = commandPool_.allocateCommandBuffer();
    }
}

void GaussianRenderer::createDescriptorSets() {
    auto device = getDevice();
    
    destroyDescriptorPool();
    
    // 创建 Descriptor Pool（支持 Uniform Buffer 和 Storage Buffer）
    std::array<vk::DescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].setType(vk::DescriptorType::eUniformBuffer)
                .setDescriptorCount(MAX_FRAMES_IN_FLIGHT * 4);
    poolSizes[1].setType(vk::DescriptorType::eStorageBuffer)
                .setDescriptorCount(18 * MAX_FRAMES_IN_FLIGHT);
    
    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.setPoolSizeCount(static_cast<uint32_t>(poolSizes.size()))
            .setPPoolSizes(poolSizes.data())
            .setMaxSets(MAX_FRAMES_IN_FLIGHT * 3);
    
    descriptorPool_ = device.createDescriptorPool(poolInfo);
    
    // 分配 Graphics Descriptor Sets（每帧一个，使用图形管线的Descriptor Set Layout）
    std::vector<vk::DescriptorSetLayout> graphicsLayouts(MAX_FRAMES_IN_FLIGHT, pipeline_->getDescriptorSetLayout());
    
    vk::DescriptorSetAllocateInfo graphicsAllocInfo{};
    graphicsAllocInfo.setDescriptorPool(descriptorPool_)
                     .setDescriptorSetCount(static_cast<uint32_t>(graphicsLayouts.size()))
                     .setPSetLayouts(graphicsLayouts.data());
    
    descriptorSets_ = device.allocateDescriptorSets(graphicsAllocInfo);
    
    // 每帧两套 Compute Descriptor Sets：A->B 和 B->A。
    std::vector<vk::DescriptorSetLayout> computeLayouts(MAX_FRAMES_IN_FLIGHT * 2, radixKeygenPipeline_->getDescriptorSetLayout());
    
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
        !gpuIndexTempBuffer_.getBuffer() ||
        !gpuKeyTempBuffer_.getBuffer() ||
        !radixHistogramBuffer_.getBuffer() ||
        !radixOffsetBuffer_.getBuffer() ||
        !instanceBuffer_.getBuffer() ||
        !uniformBuffer_.getBuffer()) {
        LOG_DEBUG("Skipping radix descriptor update until sort buffers exist");
        return;
    }

    for (size_t i = 0; i < computeDescriptorSets_.size(); ++i) {
        bool reverse = (i % 2) == 1;
        std::array<vk::DescriptorBufferInfo, 8> bufferInfos{};
        bufferInfos[0].setBuffer(reverse ? gpuIndexTempBuffer_.getBuffer() : gpuIndexBuffer_.getBuffer()).setOffset(0).setRange(VK_WHOLE_SIZE);
        bufferInfos[1].setBuffer(reverse ? gpuKeyTempBuffer_.getBuffer() : gpuKeyBuffer_.getBuffer()).setOffset(0).setRange(VK_WHOLE_SIZE);
        bufferInfos[2].setBuffer(reverse ? gpuIndexBuffer_.getBuffer() : gpuIndexTempBuffer_.getBuffer()).setOffset(0).setRange(VK_WHOLE_SIZE);
        bufferInfos[3].setBuffer(reverse ? gpuKeyBuffer_.getBuffer() : gpuKeyTempBuffer_.getBuffer()).setOffset(0).setRange(VK_WHOLE_SIZE);
        bufferInfos[4].setBuffer(radixHistogramBuffer_.getBuffer()).setOffset(0).setRange(VK_WHOLE_SIZE);
        bufferInfos[5].setBuffer(radixOffsetBuffer_.getBuffer()).setOffset(0).setRange(VK_WHOLE_SIZE);
        bufferInfos[6].setBuffer(instanceBuffer_.getBuffer()).setOffset(0).setRange(VK_WHOLE_SIZE);
        bufferInfos[7].setBuffer(uniformBuffer_.getBuffer()).setOffset(0).setRange(sizeof(UniformBufferObject));

        std::array<vk::WriteDescriptorSet, 8> writeDescriptorSets{};
        
        for (uint32_t binding = 0; binding < writeDescriptorSets.size(); ++binding) {
            writeDescriptorSets[binding].setDstSet(computeDescriptorSets_[i])
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

void GaussianRenderer::sortGaussiansByDepthGPU() {
    if (!current_model_ || current_model_->isEmpty()) {
        LOG_WARN("Skipping sort: no model or empty");
        return;
    }
    
    uint32_t pointCount = static_cast<uint32_t>(std::distance(current_model_->begin(), current_model_->end()));
    LOG_DEBUG("Sorting {} points", pointCount);
    
    ensureSortBuffers(pointCount);
    
    uint32_t groupCount = (pointCount + 255u) / 256u;

    auto device = getDevice();
    vk::FenceCreateInfo fenceInfo{};
    vk::Fence sortFence = device.createFence(fenceInfo);

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    computeCommandBuffer_.reset();
    computeCommandBuffer_.begin(beginInfo);

    struct PushConstants {
        uint32_t count;
        uint32_t shift;
        uint32_t groupCount;
        uint32_t padding;
    } pushConstants{};

    pushConstants.count = pointCount;
    pushConstants.groupCount = groupCount;

    auto insertComputeBarrier = [this]() {
        vk::MemoryBarrier memoryBarrier{};
        memoryBarrier.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                     .setDstAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);

        computeCommandBuffer_.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlagBits{},
            1, &memoryBarrier,
            0, nullptr,
            0, nullptr
        );
    };

    vk::DescriptorSet keygenSet = computeDescriptorSets_[currentFrame_ * 2];
    computeCommandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, radixKeygenPipeline_->getPipeline());
    computeCommandBuffer_.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        radixKeygenPipeline_->getPipelineLayout(),
        0,
        1,
        &keygenSet,
        0,
        nullptr
    );
    computeCommandBuffer_.pushConstants(
        radixKeygenPipeline_->getPipelineLayout(),
        vk::ShaderStageFlagBits::eCompute,
        0, sizeof(PushConstants), &pushConstants
    );
    computeCommandBuffer_.dispatch(groupCount, 1, 1);
    insertComputeBarrier();

    for (uint32_t pass = 0; pass < 4; ++pass) {
        pushConstants.shift = pass * 8u;
        vk::DescriptorSet radixSet = computeDescriptorSets_[currentFrame_ * 2 + (pass % 2)];

        computeCommandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, radixHistogramPipeline_->getPipeline());
        computeCommandBuffer_.bindDescriptorSets(
            vk::PipelineBindPoint::eCompute,
            radixHistogramPipeline_->getPipelineLayout(),
            0,
            1,
            &radixSet,
            0,
            nullptr
        );
        computeCommandBuffer_.pushConstants(
            radixHistogramPipeline_->getPipelineLayout(),
            vk::ShaderStageFlagBits::eCompute,
            0, sizeof(PushConstants), &pushConstants
        );
        computeCommandBuffer_.dispatch(groupCount, 1, 1);
        insertComputeBarrier();

        computeCommandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, radixPrefixPipeline_->getPipeline());
        computeCommandBuffer_.bindDescriptorSets(
            vk::PipelineBindPoint::eCompute,
            radixPrefixPipeline_->getPipelineLayout(),
            0,
            1,
            &radixSet,
            0,
            nullptr
        );
        computeCommandBuffer_.pushConstants(
            radixPrefixPipeline_->getPipelineLayout(),
            vk::ShaderStageFlagBits::eCompute,
            0, sizeof(PushConstants), &pushConstants
        );
        computeCommandBuffer_.dispatch(1, 1, 1);
        insertComputeBarrier();

        computeCommandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, radixScatterPipeline_->getPipeline());
        computeCommandBuffer_.bindDescriptorSets(
            vk::PipelineBindPoint::eCompute,
            radixScatterPipeline_->getPipelineLayout(),
            0,
            1,
            &radixSet,
            0,
            nullptr
        );
        computeCommandBuffer_.pushConstants(
            radixScatterPipeline_->getPipelineLayout(),
            vk::ShaderStageFlagBits::eCompute,
            0, sizeof(PushConstants), &pushConstants
        );
        computeCommandBuffer_.dispatch(groupCount, 1, 1);
        insertComputeBarrier();
    }
    
    computeCommandBuffer_.end();
    
    auto& context = Context::Instance();
    
    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBufferCount(1)
              .setPCommandBuffers(&computeCommandBuffer_);
    
    // 重置Fence
    device.resetFences(sortFence);
    
    auto sortStart = std::chrono::high_resolution_clock::now();
    context.getDevice().getGraphicsQueue().submit(submitInfo, sortFence);
    
    (void)device.waitForFences(sortFence, VK_TRUE, UINT64_MAX);
    auto sortEnd = std::chrono::high_resolution_clock::now();
    auto sortMs = std::chrono::duration_cast<std::chrono::milliseconds>(sortEnd - sortStart).count();
    LOG_DEBUG("GPU radix sorting completed: {} points, {} groups, {} ms", pointCount, groupCount, sortMs);
    
    device.destroyFence(sortFence);
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
    uint32_t groupCount = (pointCount + 255u) / 256u;
    
    if (gpuIndexBuffer_.getBuffer() ||
        gpuKeyBuffer_.getBuffer() ||
        gpuIndexTempBuffer_.getBuffer() ||
        gpuKeyTempBuffer_.getBuffer() ||
        radixHistogramBuffer_.getBuffer() ||
        radixOffsetBuffer_.getBuffer()) {
        std::array<vk::Fence, MAX_FRAMES_IN_FLIGHT> fences{};
        for (size_t frame = 0; frame < MAX_FRAMES_IN_FLIGHT; ++frame) {
            fences[frame] = inFlightFences_[frame];
        }
        (void)device.waitForFences(fences, VK_TRUE, UINT64_MAX);
    }

    gpuIndexBuffer_.cleanup();
    gpuKeyBuffer_.cleanup();
    gpuIndexTempBuffer_.cleanup();
    gpuKeyTempBuffer_.cleanup();
    radixHistogramBuffer_.cleanup();
    radixOffsetBuffer_.cleanup();
    
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

    vk::DeviceSize valueBufferSize = static_cast<vk::DeviceSize>(pointCount) * sizeof(uint32_t);
    gpuIndexTempBuffer_.create(device, physicalDevice, transferQueue, transferQueueFamilyIndex,
                               nullptr,
                               valueBufferSize,
                               vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                               vk::MemoryPropertyFlagBits::eDeviceLocal);

    gpuKeyTempBuffer_.create(device, physicalDevice, transferQueue, transferQueueFamilyIndex,
                             nullptr,
                             valueBufferSize,
                             vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                             vk::MemoryPropertyFlagBits::eDeviceLocal);

    vk::DeviceSize radixTableSize = static_cast<vk::DeviceSize>(groupCount) * 256u * sizeof(uint32_t);
    radixHistogramBuffer_.create(device, physicalDevice, transferQueue, transferQueueFamilyIndex,
                                 nullptr,
                                 radixTableSize,
                                 vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal);

    radixOffsetBuffer_.create(device, physicalDevice, transferQueue, transferQueueFamilyIndex,
                              nullptr,
                              radixTableSize,
                              vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
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
    struct alignas(16) GaussianInstanceData {
        glm::vec4 position;     // xyz: position
        glm::vec4 scale;        // xyz: scale
        glm::vec4 rotation;     // xyzw: quaternion
        
        // 球谐函数系数（完整3阶SH）
        glm::vec4 sh0_alpha;    // xyz: DC项, w: alpha
        glm::vec4 sh1[3];       // xyz: 1阶SH - 3个基函数
        glm::vec4 sh2[5];       // xyz: 2阶SH - 5个基函数
        glm::vec4 sh3[7];       // xyz: 3阶SH - 7个基函数
    };
    static_assert(sizeof(GaussianInstanceData) == sizeof(glm::vec4) * 19,
                  "GaussianInstanceData must match the GLSL std430 vec4 layout");
    
    uint32_t pointCount = static_cast<uint32_t>(std::distance(current_model_->begin(), current_model_->end()));
    std::vector<GaussianInstanceData> instanceData(pointCount);
    
    size_t idx = 0;
    for (const auto& point : *current_model_) {
        instanceData[idx].position = glm::vec4(point.position, 0.0f);
        instanceData[idx].scale = glm::vec4(point.scale, 0.0f);
        instanceData[idx].rotation = glm::vec4(
            point.rotation.x,
            point.rotation.y,
            point.rotation.z,
            point.rotation.w
        );
        
        // 复制完整的SH系数
        instanceData[idx].sh0_alpha = glm::vec4(point.color.sh0, point.alpha);
        for (int i = 0; i < 3; ++i) {
            instanceData[idx].sh1[i] = glm::vec4(point.color.sh1[i], 0.0f);
        }
        for (int i = 0; i < 5; ++i) {
            instanceData[idx].sh2[i] = glm::vec4(point.color.sh2[i], 0.0f);
        }
        for (int i = 0; i < 7; ++i) {
            instanceData[idx].sh3[i] = glm::vec4(point.color.sh3[i], 0.0f);
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
        LOG_INFO("Created instance SSBO with {} instances ({} bytes, {:.2f} MB)", 
                 pointCount, bufferSize, bufferSize / (1024.0f * 1024.0f));
        
        // 首次创建后需要更新Descriptor Set
        updateDescriptorSets();
    } else {
        // TODO: 实现动态更新逻辑（使用 staging buffer）
        LOG_WARN("Instance SSBO update not yet implemented, using initial data");
    }
}

void GaussianRenderer::updateUniformBuffer(const glm::mat4& view, const glm::mat4& projection) {
    auto device = getDevice();
    
    // 1. 更新主Uniform Buffer
    ubo_[currentFrame_].view = view;
    ubo_[currentFrame_].projection = projection;
    ubo_[currentFrame_].cameraPositionTime = glm::vec4(
        camera_.get_position(),
        static_cast<float>(glfwGetTime())
    );
    
    auto extent = swapchain_->getExtent();
    ubo_[currentFrame_].focal = glm::vec4(
        0.5f * static_cast<float>(extent.width) * projection[0][0],
        0.5f * static_cast<float>(extent.height) * projection[1][1],
        static_cast<float>(extent.width),
        static_cast<float>(extent.height)
    );
    
    LOG_DEBUG("Pixel focal lengths: fx={:.2f}, fy={:.2f}, screen={}x{}", 
              ubo_[currentFrame_].focal.x, ubo_[currentFrame_].focal.y, extent.width, extent.height);
    
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
    auto device = getDevice();
    auto commandBuffer = commandBuffers_[currentFrame_];
    
    // 开始记录命令缓冲区
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    commandBuffer.begin(beginInfo);{
        if (gpuIndexBuffer_.getBuffer()) {
            vk::MemoryBarrier sortedIndexBarrier{};
            sortedIndexBarrier.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                              .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

            commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eVertexShader,
                vk::DependencyFlagBits{},
                1, &sortedIndexBarrier,
                0, nullptr,
                0, nullptr
            );
        }
        
        // 开始渲染通道
        std::array<vk::ClearValue, 2> clearValues{};
        clearValues[0].setColor(vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}));
        clearValues[1].setDepthStencil(vk::ClearDepthStencilValue(1.0f, 0));
        
        vk::RenderPassBeginInfo renderPassInfo{};
        renderPassInfo.setRenderPass(renderPass_->getRenderPass())
                    .setFramebuffer(swapchain_->getFramebuffer(image_index))
                    .setRenderArea(vk::Rect2D({0, 0}, swapchain_->getExtent()))
                    .setClearValueCount(static_cast<uint32_t>(clearValues.size()))
                    .setPClearValues(clearValues.data());
        
        commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);{
            
            // 设置动态视口和裁剪矩形
            vk::Viewport viewport(0.0f, 0.0f, 
                                 static_cast<float>(swapchain_->getExtent().width),
                                 static_cast<float>(swapchain_->getExtent().height),
                                 0.0f, 1.0f);
            commandBuffer.setViewport(0, 1, &viewport);
            
            vk::Rect2D scissor({0, 0}, swapchain_->getExtent());
            commandBuffer.setScissor(0, 1, &scissor);
            
            // 绑定图形管线
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline_->getPipeline());
            
            // 绑定顶点缓冲区（只绑定四边形顶点，实例数据通过SSBO访问）
            vk::Buffer vertexBuffers[] = { pipeline_->getQuadVertexBuffer() };
            vk::DeviceSize offsets[] = { 0 };
            commandBuffer.bindVertexBuffers(0, 1, vertexBuffers, offsets);
            
            // 绑定索引缓冲区
            commandBuffer.bindIndexBuffer(pipeline_->getQuadIndexBuffer(), 0, vk::IndexType::eUint16);
            
            // 绑定 Uniform Buffer Descriptor Set
            if (!descriptorSets_.empty()) {
                commandBuffer.bindDescriptorSets(
                    vk::PipelineBindPoint::eGraphics,
                    pipeline_->getPipelineLayout(),
                    0,
                    1,
                    &descriptorSets_[currentFrame_],
                    0,
                    nullptr
                );
            }
            
            // 使用索引绘制四边形（每个高斯点一个实例）
            if (current_model_ && instanceBuffer_.getBuffer()) {
                uint32_t instanceCount = static_cast<uint32_t>(std::distance(current_model_->begin(), current_model_->end()));
                
                // drawIndexed: 索引数, 实例数, 起始索引, 顶点偏移, 起始实例
                commandBuffer.drawIndexed(4, instanceCount, 0, 0, 0);
            }

            if (imguiInitialized_) {
                ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
            }
        }commandBuffer.endRenderPass();
        
    // 结束记录
    }commandBuffer.end();
}

} // namespace vk_gs
