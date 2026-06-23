#include "gaussian_training.hpp"

#include "gaussian_renderer/gaussian_renderer.hpp"
#include "utils/logger.hpp"
#include "utils/camera.hpp"

#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>

namespace vk_gs {

GaussianTraining::GaussianTraining() = default;

GaussianTraining::~GaussianTraining() {
    cleanup();
}

void GaussianTraining::initialize(vk::Device device,
                                  vk::PhysicalDevice physicalDevice,
                                  vk::Queue transferQueue,
                                  uint32_t transferQueueFamilyIndex) {
    if (initialized_) {
        return;
    }

    buffers_.initialize(device, physicalDevice, transferQueue, transferQueueFamilyIndex);
    pipelines_.initialize(device);
    initialized_ = true;
}

void GaussianTraining::cleanup() {
    if (backward_) {
        backward_->cleanup();
        backward_.reset();
    }

    if (ownedForward_) {
        ownedForward_->cleanup();
        ownedForward_.reset();
    }
    forward_ = nullptr;
    rendererInitialized_ = false;

    destroyTrainingCommandResources();
    pipelines_.cleanup();
    buffers_.cleanup();
    initialized_ = false;
}

void GaussianTraining::resize(uint32_t gaussianCount, TrainingExtent extent) {
    buffers_.resize(gaussianCount, extent);
}

void GaussianTraining::initializeRenderer(GLFWwindow* window,
                                          vk::Device device,
                                          vk::PhysicalDevice physicalDevice,
                                          vk::Queue computeQueue,
                                          uint32_t computeQueueFamilyIndex,
                                          uint32_t gaussianCount,
                                          TrainingExtent extent) {
    if (rendererInitialized_) {
        return;
    }

    ownedForward_ = std::make_unique<GaussianRenderer>();
    ownedForward_->initialize(window);
    forward_ = ownedForward_.get();

    backward_ = std::make_unique<GaussianBackwardRenderer>();
    backward_->initialize(device,
                          physicalDevice,
                          computeQueue,
                          computeQueueFamilyIndex,
                          gaussianCount,
                          extent);

    createTrainingCommandResources(device, computeQueue, computeQueueFamilyIndex);
    rendererInitialized_ = true;
}

void GaussianTraining::initializeBackward(vk::Device device,
                                          vk::PhysicalDevice physicalDevice,
                                          vk::Queue computeQueue,
                                          uint32_t computeQueueFamilyIndex,
                                          uint32_t gaussianCount,
                                          TrainingExtent extent) {
    if (rendererInitialized_) {
        return;
    }

    backward_ = std::make_unique<GaussianBackwardRenderer>();
    backward_->initialize(device,
                          physicalDevice,
                          computeQueue,
                          computeQueueFamilyIndex,
                          gaussianCount,
                          extent);

    createTrainingCommandResources(device, computeQueue, computeQueueFamilyIndex);
    rendererInitialized_ = true;
}

void GaussianTraining::initializeTraining(GLFWwindow* window,
                                          vk::Device device,
                                          vk::PhysicalDevice physicalDevice,
                                          vk::Queue transferQueue,
                                          uint32_t transferQueueFamilyIndex,
                                          vk::Queue computeQueue,
                                          uint32_t computeQueueFamilyIndex,
                                          uint32_t gaussianCount,
                                          TrainingExtent extent) {
    initialize(device,
               physicalDevice,
               transferQueue,
               transferQueueFamilyIndex);
    resize(gaussianCount, extent);
    initializeRenderer(window,
                       device,
                       physicalDevice,
                       computeQueue,
                       computeQueueFamilyIndex,
                       gaussianCount,
                       extent);
    createTrainingCommandResources(device, computeQueue, computeQueueFamilyIndex);
}

void GaussianTraining::trainStep() {
    if (!initialized_) {
        LOG_WARN("GaussianTraining::trainStep called before initialization");
        return;
    }

    if (!rendererInitialized_) {
        LOG_WARN("GaussianTraining::trainStep called before renderer initialization");
        return;
    }

    if (!trainingCommandBuffer_) {
        LOG_WARN("GaussianTraining::trainStep called without training command buffer");
        return;
    }

    if (hasDataset()) {
        uploadCurrentTrainingFrame();
        syncForwardRendererToCurrentFrame();
    }

    forward_->renderToBuffer();

    auto* gaussianRenderer = dynamic_cast<GaussianRenderer*>(forward_);
    auto* gaussianBackward = dynamic_cast<GaussianBackwardRenderer*>(backward_.get());
    if (!gaussianRenderer || !gaussianBackward) {
        throw std::runtime_error("GaussianTraining currently requires GaussianRenderer and GaussianBackwardRenderer");
    }

    auto forwardBufferInfo = gaussianRenderer->renderBufferInfo();
    auto pushConstants = createPushConstants();

    trainingCommandPool_.reset();
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    trainingCommandBuffer_.begin(beginInfo);

    copyForwardRenderToTrainingBuffer(trainingCommandBuffer_, forwardBufferInfo);
    gaussianBackward->setTrainingBuffers(buffers_, trainingCommandBuffer_, pushConstants);
    gaussianBackward->backward();
    gaussianBackward->gradientDescent();

    trainingCommandBuffer_.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBufferCount(1)
              .setPCommandBuffers(&trainingCommandBuffer_);
    computeQueue_.submit(submitInfo);
    computeQueue_.waitIdle();

    if (hasDataset()) {
        currentDatasetFrameIndex_ = (currentDatasetFrameIndex_ + 1) % dataset_.size();
    }
}

void GaussianTraining::loadMipNeRF360Dataset(const std::filesystem::path& sceneRoot,
                                             uint32_t preferredDownscale) {
    dataset_ = TrainingDatasetLoader::loadMipNeRF360Scene(sceneRoot, preferredDownscale);
    currentDatasetFrameIndex_ = 0;

    const auto& firstFrame = dataset_.frames.front();
    LOG_INFO("GaussianTraining loaded dataset {} ({} frames, {}x{}, downscale {})",
             dataset_.sceneRoot.string(),
             dataset_.size(),
             firstFrame.width,
             firstFrame.height,
             dataset_.imageDownscale);
}

void GaussianTraining::setTrainingFrameIndex(size_t frameIndex) {
    if (dataset_.empty()) {
        throw std::runtime_error("Cannot set training frame index without a loaded dataset");
    }
    if (frameIndex >= dataset_.size()) {
        throw std::runtime_error("Training frame index is out of range");
    }
    currentDatasetFrameIndex_ = frameIndex;
}

void GaussianTraining::setForwardModel(const GaussianModel* model) {
    forwardModel_ = model;
}

void GaussianTraining::setForwardRenderer(Renderer& renderer) {
    if (ownedForward_) {
        ownedForward_->cleanup();
        ownedForward_.reset();
    }
    forward_ = &renderer;
}

TrainingPushConstants GaussianTraining::createPushConstants() const {
    TrainingPushConstants pushConstants{};
    pushConstants.gaussianCount = buffers_.gaussianCapacity();
    auto extent = buffers_.extent();
    pushConstants.width = extent.width;
    pushConstants.height = extent.height;
    pushConstants.pixelCount = extent.width * extent.height;
    return pushConstants;
}

void GaussianTraining::uploadCurrentTrainingFrame() {
    if (dataset_.empty()) {
        return;
    }

    const auto& frame = dataset_.frames[currentDatasetFrameIndex_];
    auto extent = buffers_.extent();
    if (frame.width != extent.width || frame.height != extent.height) {
        throw std::runtime_error("Dataset frame size does not match GaussianTraining extent. Resize training buffers to " +
                                 std::to_string(frame.width) + "x" + std::to_string(frame.height) +
                                 " before trainStep.");
    }

    TrainingImage targetImage = TrainingDatasetLoader::loadImage(frame);
    buffers_.uploadTargetColor(targetImage.pixels.data(), targetImage.width, targetImage.height);
    buffers_.uploadCamera(createTrainingCamera(frame));
}

TrainingForwardCamera GaussianTraining::createTrainingCamera(const TrainingCameraFrame& frame) const {
    TrainingForwardCamera camera{};
    camera.view = frame.worldToCamera;
    camera.model = glm::mat4(1.0f);
    camera.projection = createProjectionMatrix(frame);

    const float width = static_cast<float>(frame.width);
    const float height = static_cast<float>(frame.height);
    camera.viewport = glm::vec4(0.0f, 0.0f, width, height);
    camera.cameraPosition = glm::vec4(frame.position, 1.0f);
    return camera;
}

void GaussianTraining::syncForwardRendererToCurrentFrame() {
    if (!forwardModel_) {
        LOG_WARN("GaussianTraining has a dataset but no forward model; render target may not match training data");
        return;
    }

    auto* gaussianRenderer = dynamic_cast<GaussianRenderer*>(forward_);
    if (!gaussianRenderer) {
        return;
    }

    const auto& frame = dataset_.frames[currentDatasetFrameIndex_];
    Camera camera;
    camera.set_position(frame.position);

    glm::mat3 cameraToWorldRotation(frame.cameraToWorld);
    glm::vec3 forward = cameraToWorldRotation * glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 up = cameraToWorldRotation * glm::vec3(0.0f, 1.0f, 0.0f);
    forward = glm::length(forward) > 1e-6f ? glm::normalize(forward) : glm::vec3(0.0f, 0.0f, -1.0f);
    up = glm::length(up) > 1e-6f ? glm::normalize(up) : glm::vec3(0.0f, 1.0f, 0.0f);
    camera.look_at(frame.position + forward, up);

    gaussianRenderer->setRenderData(forwardModel_,
                                    frame.worldToCamera,
                                    createProjectionMatrix(frame),
                                    camera,
                                    glm::mat4(1.0f));
}

glm::mat4 GaussianTraining::createProjectionMatrix(const TrainingCameraFrame& frame) const {
    const float width = static_cast<float>(frame.width);
    const float height = static_cast<float>(frame.height);
    const float nearPlane = 0.01f;
    const float farPlane = 1000.0f;

    glm::mat4 projection(0.0f);
    projection[0][0] = 2.0f * frame.fx / width;
    projection[1][1] = -2.0f * frame.fy / height;
    projection[2][0] = 1.0f - 2.0f * frame.cx / width;
    projection[2][1] = 2.0f * frame.cy / height - 1.0f;
    projection[2][2] = farPlane / (farPlane - nearPlane);
    projection[2][3] = 1.0f;
    projection[3][2] = -(farPlane * nearPlane) / (farPlane - nearPlane);
    return projection;
}

void GaussianTraining::createTrainingCommandResources(vk::Device device,
                                                      vk::Queue computeQueue,
                                                      uint32_t computeQueueFamilyIndex) {
    if (trainingCommandBuffer_) {
        return;
    }

    computeQueue_ = computeQueue;
    computeQueueFamilyIndex_ = computeQueueFamilyIndex;
    trainingCommandPool_.create(device,
                                computeQueueFamilyIndex_,
                                vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    trainingCommandBuffer_ = trainingCommandPool_.allocateCommandBuffer();
}

void GaussianTraining::destroyTrainingCommandResources() {
    if (trainingCommandBuffer_) {
        trainingCommandPool_.freeCommandBuffer(trainingCommandBuffer_);
        trainingCommandBuffer_ = nullptr;
    }

    trainingCommandPool_.cleanup();
    computeQueue_ = nullptr;
    computeQueueFamilyIndex_ = 0;
}

void GaussianTraining::copyForwardRenderToTrainingBuffer(vk::CommandBuffer commandBuffer,
                                                         const vk::DescriptorBufferInfo& sourceInfo) {
    auto destinationInfo = buffers_.renderedColorInfo();
    if (!sourceInfo.buffer || !destinationInfo.buffer) {
        LOG_WARN("GaussianTraining::copyForwardRenderToTrainingBuffer skipped because buffers are not ready");
        return;
    }

    vk::DeviceSize copySize = std::min(sourceInfo.range, destinationInfo.range);
    if (copySize == 0 || copySize == VK_WHOLE_SIZE) {
        auto extent = buffers_.extent();
        copySize = static_cast<vk::DeviceSize>(extent.width) *
                   static_cast<vk::DeviceSize>(extent.height) *
                   sizeof(glm::vec4);
    }

    vk::BufferMemoryBarrier sourceReady{};
    sourceReady.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eShaderWrite)
               .setDstAccessMask(vk::AccessFlagBits::eTransferRead)
               .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
               .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
               .setBuffer(sourceInfo.buffer)
               .setOffset(sourceInfo.offset)
               .setSize(sourceInfo.range);

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eComputeShader,
                                  vk::PipelineStageFlagBits::eTransfer,
                                  vk::DependencyFlagBits{},
                                  0,
                                  nullptr,
                                  1,
                                  &sourceReady,
                                  0,
                                  nullptr);

    vk::BufferCopy copyRegion{};
    copyRegion.setSrcOffset(sourceInfo.offset)
              .setDstOffset(destinationInfo.offset)
              .setSize(copySize);
    commandBuffer.copyBuffer(sourceInfo.buffer,
                             destinationInfo.buffer,
                             1,
                             &copyRegion);

    vk::BufferMemoryBarrier destinationReady{};
    destinationReady.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                    .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
                    .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .setBuffer(destinationInfo.buffer)
                    .setOffset(destinationInfo.offset)
                    .setSize(destinationInfo.range);

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                  vk::PipelineStageFlagBits::eComputeShader,
                                  vk::DependencyFlagBits{},
                                  0,
                                  nullptr,
                                  1,
                                  &destinationReady,
                                  0,
                                  nullptr);
}

} // namespace vk_gs
