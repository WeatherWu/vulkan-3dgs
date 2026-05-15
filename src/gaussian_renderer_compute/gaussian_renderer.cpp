#include "gaussian_renderer.hpp"
#include "context/context.hpp"
#include "utils/logger.hpp"
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
    
    LOG_INFO("Got Vulkan device and surface");
    
    // 初始化GPU排序缓存状态
    gpu_sort_completed_ = false;
    last_sorted_point_count_ = 0;
    last_sorted_model_ = nullptr;
    last_camera_position_ = glm::vec3(0.0f);
    last_view_matrix_ = glm::mat4(1.0f);
    last_projection_matrix_ = glm::mat4(1.0f);
    
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
    swapchain_->createFramebuffers(device, renderPass_->getRenderPass(), RenderPass::DepthFormat);
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
    
    // Keep sorting on the graphics queue. vkgs records its rank/sort/projection
    // work in the graphics submission chain before drawing; using a separate
    // compute queue here needs semaphore-based cross-queue memory dependencies,
    // otherwise the sorted index buffer can be stale when the splat pass reads it.
    auto& context = Context::Instance();
    uint32_t computeQueueFamily = context.getDevice().getQueueFamilyIndices().graphicsIndex.value();
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
    LOG_INFO("Acquired swap chain image successfully");
    
    // 3. GPU排序。透明Gaussian需要随相机变化保持远到近顺序。
    uint32_t currentPointCount = static_cast<uint32_t>(std::distance(current_model_->begin(), current_model_->end()));
    
    const float matrixEpsilon = 1e-5f;
    bool needResort = !gpu_sort_completed_ ||
                      current_model_ != last_sorted_model_ ||
                      currentPointCount != last_sorted_point_count_ ||
                      matrixChanged(ubo_[currentFrame_].view, last_view_matrix_, matrixEpsilon) ||
                      matrixChanged(ubo_[currentFrame_].projection, last_projection_matrix_, matrixEpsilon);
    
    if (needResort) {
        sortGaussiansByDepthGPU();
        
        gpu_sort_completed_ = true;
        last_sorted_point_count_ = currentPointCount;
        last_sorted_model_ = current_model_;
        last_camera_position_ = camera_.get_position();
        last_view_matrix_ = ubo_[currentFrame_].view;
        last_projection_matrix_ = ubo_[currentFrame_].projection;
    }
    LOG_INFO("GPU sorting  successfully");
    
    // 4. 更新Instance Buffer（仅在首次或模型变化时重建）
    if (!instanceBuffer_.getBuffer()) {
        updateVertexBuffer();
    }
    LOG_INFO("Vertex buffer updated successfully");
    
    // 5. 更新Uniform Buffer（每帧更新）
    updateUniformBuffer(ubo_[currentFrame_].view, ubo_[currentFrame_].projection);
    LOG_INFO("Uniform buffer updated successfully");
    
    // 6. 记录渲染命令
    recordCommandBuffer(imageIndex);
    LOG_INFO("Command buffer recorded successfully");
    
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
    LOG_INFO("Command buffer submitted successfully");
    
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
    LOG_INFO("Image presented successfully");
    
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
    swapchain_->createFramebuffers(device, renderPass_->getRenderPass(), RenderPass::DepthFormat);
    
    pipeline_->cleanup();
    pipeline_->initialize(device, renderPass_->getRenderPass(), swapchain_->getExtent());
}

void GaussianRenderer::setRenderData(const GaussianModel* model, const glm::mat4& view, const glm::mat4& projection, const vk_gs::Camera& camera) {
    if (model != current_model_) {
        gpu_sort_completed_ = false;
    }

    current_model_ = model;
    camera_ = camera;
    for (uint32_t i = 0u; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        ubo_[i].view = view;
        ubo_[i].projection = projection;
        
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
    LOG_INFO("Creating Gaussian buffers with Jacobian projection support");
    
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
                .setDescriptorCount(4 * MAX_FRAMES_IN_FLIGHT); // Compute需要2个SSBO + Graphics需要2个SSBO
    
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
    
    LOG_INFO("Descriptor sets updated (Graphics UBOs+SSBO: {}, Compute SSBOs: {})", 
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
    computeDistances(glm::vec3(ubo_[currentFrame_].cameraPositionTime));
    
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
    computeCommandBuffer_.reset();
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
    
    for (uint32_t stage = 2; stage <= paddedCount; stage *= 2) {
        for (uint32_t substage = stage / 2; substage > 0; substage /= 2) {
            // 设置Push Constants
            struct PushConstants {
                uint32_t count;
                uint32_t stage;
                uint32_t substage;
            } pushConstants;
            
            pushConstants.count = paddedCount;
            pushConstants.stage = stage;
            pushConstants.substage = substage;
            
            computeCommandBuffer_.pushConstants(
                computePipeline_->getPipelineLayout(),
                vk::ShaderStageFlagBits::eCompute,
                0, sizeof(PushConstants), &pushConstants
            );
            
            // 分派计算任务
            uint32_t workgroupCount = (paddedCount + 255) / 256;
            computeCommandBuffer_.dispatch(workgroupCount, 1, 1);
            
            // 添加内存屏障，确保当前dispatch完成后再执行下一个
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
    context.getDevice().getGraphicsQueue().submit(submitInfo, sortFence);
    
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
    auto transferQueue = context.getDevice().getGraphicsQueue();
    uint32_t transferQueueFamilyIndex = context.getDevice().getQueueFamilyIndices().graphicsIndex.value();
    
    // 初始化索引和距离缓冲区（如果尚未创建）
    uint32_t pointCount = static_cast<uint32_t>(std::distance(current_model_->begin(), current_model_->end()));
    
    // 计算padding后的大小（必须是2的幂次）
    uint32_t paddedCount = 1;
    while (paddedCount < pointCount) {
        paddedCount *= 2;
    }
    
    // 每次重排前重置索引和深度数据，避免沿用上一帧已排序的距离。
    gpuIndexBuffer_.cleanup();
    gpuDistanceBuffer_.cleanup();

    std::vector<uint32_t> initialIndices(paddedCount);
    for (uint32_t i = 0; i < pointCount; ++i) {
        initialIndices[i] = i;
    }
    for (uint32_t i = pointCount; i < paddedCount; ++i) {
        initialIndices[i] = 0xFFFFFFFF;
    }
    
    gpuIndexBuffer_.create(device, physicalDevice, transferQueue, transferQueueFamilyIndex,
                          initialIndices.data(),
                          initialIndices.size() * sizeof(uint32_t),
                          vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                          vk::MemoryPropertyFlagBits::eDeviceLocal);
    
    std::vector<float> distances(paddedCount);
    uint32_t idx = 0;
    for (const auto& point : *current_model_) {
        glm::vec4 clip = ubo_[currentFrame_].projection * ubo_[currentFrame_].view * glm::vec4(point.position, 1.0f);
        if (clip.w <= 1e-6f) {
            distances[idx++] = -std::numeric_limits<float>::infinity();
            continue;
        }

        glm::vec3 ndc = glm::vec3(clip) / clip.w;
        if (std::abs(ndc.x) > 1.0f || std::abs(ndc.y) > 1.0f || ndc.z < 0.0f || ndc.z > 1.0f) {
            distances[idx++] = -std::numeric_limits<float>::infinity();
            continue;
        }

        // The local bitonic sorter orders keys descending. Vulkan NDC depth is
        // larger for farther points, so this yields the required back-to-front
        // blending order. vkgs uses 1-depth with its radix-sort order; copying
        // that key here would invert the local draw order.
        distances[idx++] = ndc.z;
    }
    for (uint32_t i = pointCount; i < paddedCount; ++i) {
        distances[i] = -std::numeric_limits<float>::infinity();
    }
    
    gpuDistanceBuffer_.create(device, physicalDevice, transferQueue, transferQueueFamilyIndex,
                             distances.data(),
                             distances.size() * sizeof(float),
                             vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                             vk::MemoryPropertyFlagBits::eDeviceLocal);
    
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
        }commandBuffer.endRenderPass();
        
    // 结束记录
    }commandBuffer.end();
}

} // namespace vk_gs
