#include "gaussian_renderer.hpp"
#include "context/context.hpp"
#include "utils/logger.hpp"
#include <algorithm>
#include <cstring>
#include <cmath>

namespace vk_gs {

GaussianRenderer::GaussianRenderer() = default;

GaussianRenderer::~GaussianRenderer() {
    cleanup();
}

void GaussianRenderer::initialize(GLFWwindow* window) {
    LOG_INFO("Starting GaussianRenderer initialization");
    
    auto& context = Context::Instance();
    auto device = getDevice();
    auto surface = context.getSurface();
    
    LOG_INFO("Got Vulkan device and surface");
    
    // 初始化GPU排序缓存状态
    gpu_sort_completed_ = false;
    last_sorted_point_count_ = 0;
    last_camera_position_ = glm::vec3(0.0f);
    
    // 1. 获取窗口尺寸
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    LOG_INFO("Window size: {}x{}", width, height);
    
    // 2. 创建 Swapchain（只创建图像和ImageView）
    swapchain_ = std::make_unique<Swapchain>(surface);
    swapchain_->createSwapchain(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
    LOG_INFO("Swapchain created with {} images", swapchain_->getImageCount());
    
    // 3. 创建 RenderPass
    renderPass_ = std::make_unique<RenderPass>();
    renderPass_->initialize(swapchain_->getImageFormat());
    LOG_INFO("RenderPass created");
    
    // 4. 创建 Framebuffers（需要RenderPass）
    swapchain_->createFramebuffers(device, renderPass_->getRenderPass());
    LOG_INFO("Framebuffers created");
    
    // 5. 创建 Pipeline
    pipeline_ = std::make_unique<Pipeline>();
    pipeline_->initialize(device, renderPass_->getRenderPass(), swapchain_->getExtent());
    LOG_INFO("Graphics Pipeline created");
    
    // 6. 创建 Compute Pipeline
    createComputePipeline();
    LOG_INFO("Compute Pipeline created");
    
    // 7. 创建高斯特有资源
    createBuffers();
    LOG_INFO("Buffers created");
    
    createSyncObjects();
    LOG_INFO("Sync objects created");
    
    LOG_INFO("Gaussian Renderer initialized successfully");
}

void GaussianRenderer::cleanup() {
    LOG_INFO("Cleaning up Gaussian-specific resources");
    
    auto device = getDevice();
    
    // 等待设备空闲
    if (device) {
        device.waitIdle();
    }
    
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
    
    // 2. 清理缓冲区（释放Descriptor Set引用的资源）
    instanceBuffer_.cleanup();
    uniformBuffer_.cleanup();
    screenInfoBuffer_.cleanup();
    gpuIndexBuffer_.cleanup();
    gpuDistanceBuffer_.cleanup();
    
    // 3. 清理Pipelines
    pipeline_.reset();
    computePipeline_.reset();
    
    // 4. 清理Descriptor Pool（会自动销毁所有Descriptor Sets）
    if (descriptorPool_) {
        device.destroyDescriptorPool(descriptorPool_);
        descriptorPool_ = nullptr;
    }
    
    // 5. 清理命令池
    commandPool_.cleanup();
    computeCommandPool_.cleanup();
    
    // 6. 清理RenderPass和Swapchain
    renderPass_.reset();
    swapchain_.reset();
    
    LOG_INFO("Gaussian Renderer cleaned up");
}

void GaussianRenderer::createComputePipeline() {
    LOG_INFO("Creating compute pipeline for GPU sorting");
    
    auto device = getDevice();
    
    // 创建 Compute Pipeline（使用独立封装的类）
    computePipeline_ = std::make_unique<ComputePipeline>();
    computePipeline_->initialize(device, "shaders/gaussian_compute_shader.comp.spv");
    
    // 创建 Descriptor Sets
    createDescriptorSets();
    
    // 分配 Compute Command Buffer（优先使用专用计算队列）
    auto& context = Context::Instance();
    uint32_t computeQueueFamily = context.getDevice().getQueueFamilyIndices().computeIndex.value_or(
        context.getDevice().getQueueFamilyIndices().graphicsIndex.value());
    computeCommandPool_.create(device, computeQueueFamily, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    computeCommandBuffer_ = computeCommandPool_.allocateCommandBuffer();
    
    LOG_INFO("Compute pipeline created successfully");
}

void GaussianRenderer::render() {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    auto& context = Context::Instance();
    auto device = getDevice();
    
    // 1. 等待上一帧完成
    vk::Result waitResult = device.waitForFences(1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);
    (void)waitResult;
    
    // 2. 获取下一张交换链图像
    uint32_t imageIndex;
    try {
        vk::Result result = device.acquireNextImageKHR(swapchain_->getSwapchain(), UINT64_MAX, 
                                                        imageAvailableSemaphores_[currentFrame_], nullptr, &imageIndex);
        if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("Failed to acquire swap chain image!");
        }
    } catch (const vk::OutOfDateKHRError&) {
        LOG_WARN("Swapchain out of date, skipping frame");
        return;
    }
    
    // 3. GPU排序（仅首帧或点数变化时执行）
    uint32_t currentPointCount = static_cast<uint32_t>(std::distance(current_model_->begin(), current_model_->end()));
    
    bool needResort = !gpu_sort_completed_ || 
                      currentPointCount != last_sorted_point_count_;
    
    if (needResort) {
        sortGaussiansByDepthGPU();
        
        gpu_sort_completed_ = true;
        last_sorted_point_count_ = currentPointCount;
    }
    
    // 4. 更新Instance Buffer（仅在首次或模型变化时重建）
    if (!instanceBuffer_.getBuffer()) {
        updateVertexBuffer();
    }
    
    // 5. 更新Uniform Buffer
    updateUniformBuffer(ubo_.view, ubo_.projection);
    
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
    
    try {
        vk::Result presentResult = context.getDevice().getPresentQueue().presentKHR(presentInfo);
        (void)presentResult;
    } catch (...) {
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
    
    swapchain_->recreateSwapchain(width, height);
    
    renderPass_->cleanup();
    renderPass_->initialize(swapchain_->getImageFormat());
    
    pipeline_->cleanup();
    pipeline_->initialize(device, renderPass_->getRenderPass(), swapchain_->getExtent());
}

void GaussianRenderer::setRenderData(const GaussianModel* model, const glm::mat4& view, const glm::mat4& projection) {
    current_model_ = model;
    ubo_.view = view;
    ubo_.projection = projection;
    ubo_.cameraPosition = glm::vec3(view[3][0], view[3][1], view[3][2]);
    ubo_.time = static_cast<float>(glfwGetTime());
}

void GaussianRenderer::createBuffers() {
    LOG_INFO("Creating Gaussian buffers with Jacobian projection support");
    
    auto& context = Context::Instance();
    auto device = getDevice();
    auto physicalDevice = context.PhysicalDevice();
    auto transferQueue = context.getTransferQueue();
    uint32_t transferQueueFamilyIndex = context.getDevice().getQueueFamilyIndices().transferIndex.value_or(
        context.getDevice().getQueueFamilyIndices().graphicsIndex.value());
    
    // 1. 创建主 Uniform Buffer（每个帧一个）
    UniformBufferObject initialUBO{};
    uniformBuffer_.create(device, physicalDevice, transferQueue, transferQueueFamilyIndex,
                         &initialUBO, sizeof(UniformBufferObject),
                         vk::BufferUsageFlagBits::eUniformBuffer,
                         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    
    // 2. 创建屏幕信息 Uniform Buffer
    struct ScreenInfoUBO {
        glm::vec2 screenSize;
        float padding[2]; // 填充到16字节对齐
    } initialScreenInfo{};
    screenInfoBuffer_.create(device, physicalDevice, transferQueue, transferQueueFamilyIndex,
                            &initialScreenInfo, sizeof(ScreenInfoUBO),
                            vk::BufferUsageFlagBits::eUniformBuffer,
                            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    
    LOG_INFO("Gaussian Jacobian projection buffers created successfully");
}

void GaussianRenderer::createSyncObjects() {
    LOG_INFO("Creating synchronization objects (MAX_FRAMES_IN_FLIGHT={})", MAX_FRAMES_IN_FLIGHT);
    
    auto device = getDevice();
    
    // 获取Swapchain图像数量（仅用于日志）
    swapchainImageCount_ = swapchain_->getImageCount();
    LOG_INFO("Swapchain has {} images, using {} frames in flight", 
             swapchainImageCount_, MAX_FRAMES_IN_FLIGHT);
    
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
    
    LOG_INFO("Synchronization objects created successfully");
}

void GaussianRenderer::createDescriptorSets() {
    auto device = getDevice();
    
    // 创建 Descriptor Pool（支持 Uniform Buffer 和 Storage Buffer）
    std::array<vk::DescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].setType(vk::DescriptorType::eUniformBuffer)
                .setDescriptorCount(MAX_FRAMES_IN_FLIGHT * 2);  // Graphics需要2个UBO
    poolSizes[1].setType(vk::DescriptorType::eStorageBuffer)
                .setDescriptorCount(2 * MAX_FRAMES_IN_FLIGHT); // Compute需要2个SSBO
    
    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.setPoolSizeCount(static_cast<uint32_t>(poolSizes.size()))
            .setPPoolSizes(poolSizes.data())
            .setMaxSets(MAX_FRAMES_IN_FLIGHT * 2);  // 总共需要2套Descriptor Sets
    
    descriptorPool_ = device.createDescriptorPool(poolInfo);
    
    // 分配 Graphics Descriptor Sets（每帧一个，使用图形管线的Descriptor Set Layout）
    std::vector<vk::DescriptorSetLayout> graphicsLayouts(MAX_FRAMES_IN_FLIGHT, pipeline_->getDescriptorSetLayout());
    
    vk::DescriptorSetAllocateInfo graphicsAllocInfo{};
    graphicsAllocInfo.setDescriptorPool(descriptorPool_)
                     .setDescriptorSetCount(static_cast<uint32_t>(graphicsLayouts.size()))
                     .setPSetLayouts(graphicsLayouts.data());
    
    descriptorSets_ = device.allocateDescriptorSets(graphicsAllocInfo);
    
    // 分配 Compute Descriptor Sets（每帧一个，使用计算管线的Descriptor Set Layout）
    std::vector<vk::DescriptorSetLayout> computeLayouts(MAX_FRAMES_IN_FLIGHT, computePipeline_->getDescriptorSetLayout());
    
    vk::DescriptorSetAllocateInfo computeAllocInfo{};
    computeAllocInfo.setDescriptorPool(descriptorPool_)
                    .setDescriptorSetCount(static_cast<uint32_t>(computeLayouts.size()))
                    .setPSetLayouts(computeLayouts.data());
    
    computeDescriptorSets_ = device.allocateDescriptorSets(computeAllocInfo);
    
    LOG_INFO("Descriptor sets created successfully (Graphics: {}, Compute: {})", 
             descriptorSets_.size(), computeDescriptorSets_.size());
}

void GaussianRenderer::updateDescriptorSets() {
    auto device = getDevice();
    
    // 为每个 Graphics Descriptor Set 更新 Uniform Buffer 绑定
    for (size_t i = 0; i < descriptorSets_.size(); ++i) {
        std::array<vk::WriteDescriptorSet, 2> writeDescriptorSets{};
        
        // Binding 0: 主Uniform Buffer (View/Projection/Camera)
        vk::DescriptorBufferInfo uboInfo{};
        uboInfo.setBuffer(uniformBuffer_.getBuffer())
               .setOffset(0)
               .setRange(sizeof(UniformBufferObject));
        
        writeDescriptorSets[0].setDstSet(descriptorSets_[i])
                              .setDstBinding(0)
                              .setDstArrayElement(0)
                              .setDescriptorCount(1)
                              .setDescriptorType(vk::DescriptorType::eUniformBuffer)
                              .setPBufferInfo(&uboInfo);
        
        // Binding 1: 屏幕信息Uniform Buffer
        vk::DescriptorBufferInfo screenInfo{};
        screenInfo.setBuffer(screenInfoBuffer_.getBuffer())
                  .setOffset(0)
                  .setRange(sizeof(glm::vec2) + sizeof(float) * 2); // screenSize + padding
        
        writeDescriptorSets[1].setDstSet(descriptorSets_[i])
                              .setDstBinding(1)
                              .setDstArrayElement(0)
                              .setDescriptorCount(1)
                              .setDescriptorType(vk::DescriptorType::eUniformBuffer)
                              .setPBufferInfo(&screenInfo);
        
        device.updateDescriptorSets(static_cast<uint32_t>(writeDescriptorSets.size()), 
                                   writeDescriptorSets.data(), 
                                   0, nullptr);
    }
    
    // 为每个 Compute Descriptor Set 更新 Storage Buffer 绑定
    for (size_t i = 0; i < computeDescriptorSets_.size(); ++i) {
        std::array<vk::WriteDescriptorSet, 2> writeDescriptorSets{};
        
        // Binding 0: 索引缓冲区 (Storage Buffer)
        vk::DescriptorBufferInfo indexInfo{};
        indexInfo.setBuffer(gpuIndexBuffer_.getBuffer())
                 .setOffset(0)
                 .setRange(VK_WHOLE_SIZE);
        
        writeDescriptorSets[0].setDstSet(computeDescriptorSets_[i])
                              .setDstBinding(0)
                              .setDstArrayElement(0)
                              .setDescriptorCount(1)
                              .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                              .setPBufferInfo(&indexInfo);
        
        // Binding 1: 距离缓冲区 (Storage Buffer)
        vk::DescriptorBufferInfo distanceInfo{};
        distanceInfo.setBuffer(gpuDistanceBuffer_.getBuffer())
                    .setOffset(0)
                    .setRange(VK_WHOLE_SIZE);
        
        writeDescriptorSets[1].setDstSet(computeDescriptorSets_[i])
                              .setDstBinding(1)
                              .setDstArrayElement(0)
                              .setDescriptorCount(1)
                              .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                              .setPBufferInfo(&distanceInfo);
        
        device.updateDescriptorSets(static_cast<uint32_t>(writeDescriptorSets.size()), 
                                   writeDescriptorSets.data(), 
                                   0, nullptr);
    }
    
    LOG_INFO("Descriptor sets updated (Graphics UBOs: {}, Compute SSBOs: {})", 
             descriptorSets_.size(), computeDescriptorSets_.size());
}

void GaussianRenderer::sortGaussiansByDepthGPU() {
    LOG_INFO("Starting GPU sorting");
    
    if (!current_model_ || current_model_->isEmpty()) {
        LOG_WARN("Skipping sort: no model or empty");
        return;
    }
    
    uint32_t pointCount = static_cast<uint32_t>(std::distance(current_model_->begin(), current_model_->end()));
    LOG_DEBUG("Sorting {} points", pointCount);
    
    // 1. 计算每个点到相机的距离
    computeDistances(ubo_.cameraPosition);
    
    // 2. 执行 Bitonic Sort（需要元素数量是 2 的幂次）
    uint32_t paddedCount = 1;
    while (paddedCount < pointCount) {
        paddedCount *= 2;
    }
    
    // 创建复用的Fence
    auto device = getDevice();
    vk::FenceCreateInfo fenceInfo{};
    vk::Fence sortFence = device.createFence(fenceInfo);
    
    // 执行多轮排序
    uint32_t totalDispatches = 0;
    
    // 优化：批量记录所有dispatch到单个Command Buffer，减少同步开销
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    computeCommandBuffer_.begin(beginInfo);
    
    // 绑定Compute Pipeline（只需绑定一次）
    computeCommandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline_->getPipeline());
    
    // 绑定Descriptor Set（只需绑定一次）
    if (!computeDescriptorSets_.empty()) {
        computeCommandBuffer_.bindDescriptorSets(
            vk::PipelineBindPoint::eCompute,
            computePipeline_->getPipelineLayout(),
            0,
            1,
            &computeDescriptorSets_[currentFrame_],
            0,
            nullptr
        );
    }
    
    for (uint32_t stage = 1; stage <= paddedCount; stage *= 2) {
        for (uint32_t substage = stage / 2; substage > 0; substage /= 2) {
            // 设置Push Constants
            struct PushConstants {
                uint32_t count;
                uint32_t stage;
                uint32_t substage;
            } pushConstants;
            
            pushConstants.count = pointCount;
            pushConstants.stage = stage;
            pushConstants.substage = substage;
            
            computeCommandBuffer_.pushConstants(
                computePipeline_->getPipelineLayout(),
                vk::ShaderStageFlagBits::eCompute,
                0, sizeof(PushConstants), &pushConstants
            );
            
            // 分派计算任务
            uint32_t workgroupCount = (pointCount + 255) / 256;
            computeCommandBuffer_.dispatch(workgroupCount, 1, 1);
            
            // 添加内存屏障，确保当前dispatch完成后再执行下一个
            vk::MemoryBarrier memoryBarrier{};
            memoryBarrier.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                        .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
            
            computeCommandBuffer_.pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlagBits{},
                1, &memoryBarrier,
                0, nullptr,
                0, nullptr
            );
            
            totalDispatches++;
        }
    }
    
    // 结束记录
    computeCommandBuffer_.end();
    
    // 一次性提交所有dispatch（大幅减少同步开销）
    auto& context = Context::Instance();
    
    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBufferCount(1)
              .setPCommandBuffers(&computeCommandBuffer_);
    
    // 重置Fence
    device.resetFences(sortFence);
    
    LOG_INFO("Submitting {} dispatches in single batch", totalDispatches);
    auto sortStart = std::chrono::high_resolution_clock::now();
    context.getDevice().getComputeQueue().submit(submitInfo, sortFence);
    
    // 只等待一次
    (void)device.waitForFences(sortFence, VK_TRUE, UINT64_MAX);
    auto sortEnd = std::chrono::high_resolution_clock::now();
    auto sortMs = std::chrono::duration_cast<std::chrono::milliseconds>(sortEnd - sortStart).count();
    LOG_INFO("All dispatches completed (took {} ms)", sortMs);
    
    // 销毁复用的Fence
    device.destroyFence(sortFence);
    
    LOG_INFO("GPU sorting completed ({} dispatches)", totalDispatches);
}

void GaussianRenderer::computeDistances(const glm::vec3& cameraPosition) {
    if (!current_model_) {
        return;
    }
    
    auto device = getDevice();
    auto& context = Context::Instance();
    auto physicalDevice = context.PhysicalDevice();
    auto transferQueue = context.getTransferQueue();
    uint32_t transferQueueFamilyIndex = context.getDevice().getQueueFamilyIndices().transferIndex.value_or(
        context.getDevice().getQueueFamilyIndices().graphicsIndex.value());
    
    // 初始化索引和距离缓冲区（如果尚未创建）
    uint32_t pointCount = static_cast<uint32_t>(std::distance(current_model_->begin(), current_model_->end()));
    
    // 计算padding后的大小（必须是2的幂次）
    uint32_t paddedCount = 1;
    while (paddedCount < pointCount) {
        paddedCount *= 2;
    }
    
    if (!gpuIndexBuffer_.getBuffer()) {
        // 创建初始索引数组（使用paddedCount大小）
        std::vector<uint32_t> initialIndices(paddedCount);
        for (uint32_t i = 0; i < pointCount; ++i) {
            initialIndices[i] = i;
        }
        // 填充剩余部分为无效值
        for (uint32_t i = pointCount; i < paddedCount; ++i) {
            initialIndices[i] = 0xFFFFFFFF;
        }
        
        gpuIndexBuffer_.create(device, physicalDevice, transferQueue, transferQueueFamilyIndex,
                              initialIndices.data(),
                              initialIndices.size() * sizeof(uint32_t),
                              vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                              vk::MemoryPropertyFlagBits::eDeviceLocal);
    }
    
    // 计算并创建距离数组（只在首次创建时）
    if (!gpuDistanceBuffer_.getBuffer()) {
        std::vector<float> distances(paddedCount);
        uint32_t idx = 0;
        for (const auto& point : *current_model_) {
            distances[idx++] = glm::length(point.position - cameraPosition);
        }
        // 填充剩余部分为最大值（确保排序到末尾）
        for (uint32_t i = pointCount; i < paddedCount; ++i) {
            distances[i] = std::numeric_limits<float>::max();
        }
        
        gpuDistanceBuffer_.create(device, physicalDevice, transferQueue, transferQueueFamilyIndex,
                                 distances.data(),
                                 distances.size() * sizeof(float),
                                 vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal);
        
        // 更新 Descriptor Set（只在首次创建时）
        updateDescriptorSets();
    }
}

void GaussianRenderer::updateVertexBuffer() {
    if (!current_model_ || current_model_->isEmpty()) {
        return;
    }
    
    auto device = getDevice();
    auto& context = Context::Instance();
    auto physicalDevice = context.PhysicalDevice();
    auto transferQueue = context.getTransferQueue();
    uint32_t transferQueueFamilyIndex = context.getDevice().getQueueFamilyIndices().transferIndex.value_or(
        context.getDevice().getQueueFamilyIndices().graphicsIndex.value());
    
    // 构建实例数据数组
    std::vector<float> instanceData;
    uint32_t pointCount = static_cast<uint32_t>(std::distance(current_model_->begin(), current_model_->end()));
    instanceData.reserve(pointCount * 14); // 每个实例14个float
    
    for (const auto& point : *current_model_) {
        // position (3)
        instanceData.push_back(point.position.x);
        instanceData.push_back(point.position.y);
        instanceData.push_back(point.position.z);
        
        // scale (3)
        instanceData.push_back(point.scale.x);
        instanceData.push_back(point.scale.y);
        instanceData.push_back(point.scale.z);
        
        // rotation quaternion (4): x, y, z, w
        instanceData.push_back(point.rotation.x);
        instanceData.push_back(point.rotation.y);
        instanceData.push_back(point.rotation.z);
        instanceData.push_back(point.rotation.w);
        
        // color from SH0 (3)
        instanceData.push_back(point.color.sh0.x);
        instanceData.push_back(point.color.sh0.y);
        instanceData.push_back(point.color.sh0.z);
        
        // alpha (1)
        instanceData.push_back(point.alpha);
    }
    
    // 创建或更新实例缓冲区
    if (!instanceBuffer_.getBuffer()) {
        instanceBuffer_.create(device, physicalDevice, transferQueue, transferQueueFamilyIndex,
                              instanceData.data(),
                              instanceData.size() * sizeof(float),
                              vk::BufferUsageFlagBits::eVertexBuffer,
                              vk::MemoryPropertyFlagBits::eDeviceLocal);
        LOG_INFO("Created instance buffer with {} instances", pointCount);
    } else {
        // TODO: 实现动态更新逻辑（使用 staging buffer）
        LOG_WARN("Instance buffer update not yet implemented, using initial data");
    }
}

void GaussianRenderer::updateUniformBuffer(const glm::mat4& view, const glm::mat4& projection) {
    auto device = getDevice();
    
    // 1. 更新主Uniform Buffer
    ubo_.view = view;
    ubo_.projection = projection;
    ubo_.time = static_cast<float>(glfwGetTime());
    
    // 更新屏幕分辨率
    auto extent = swapchain_->getExtent();
    ubo_.screenSize = glm::vec2(static_cast<float>(extent.width), static_cast<float>(extent.height));
    
    void* data = device.mapMemory(uniformBuffer_.getMemory(), 0, sizeof(UniformBufferObject));
    std::memcpy(data, &ubo_, sizeof(UniformBufferObject));
    device.unmapMemory(uniformBuffer_.getMemory());
    
    // 2. 更新屏幕信息Uniform Buffer
    struct ScreenInfoUBO {
        glm::vec2 screenSize;
        float padding[2];
    } screenInfo{};
    screenInfo.screenSize = ubo_.screenSize;
    
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
        
        // 开始渲染通道
        vk::ClearValue clearColor{ vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}) };
        
        vk::RenderPassBeginInfo renderPassInfo{};
        renderPassInfo.setRenderPass(renderPass_->getRenderPass())
                    .setFramebuffer(swapchain_->getFramebuffer(image_index))
                    .setRenderArea(vk::Rect2D({0, 0}, swapchain_->getExtent()))
                    .setClearValueCount(1)
                    .setPClearValues(&clearColor);
        
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
            
            // 绑定顶点缓冲区（Binding 0: 四边形顶点，Binding 1: 实例数据）
            std::array<vk::Buffer, 2> vertexBuffers = {
                pipeline_->getQuadVertexBuffer(),
                instanceBuffer_.getBuffer()
            };
            std::array<vk::DeviceSize, 2> offsets = {0, 0};
            commandBuffer.bindVertexBuffers(0, 2, vertexBuffers.data(), offsets.data());
            
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
        }commandBuffer.endRenderPass();
        
    // 结束记录
    }commandBuffer.end();
}

} // namespace vk_gs
