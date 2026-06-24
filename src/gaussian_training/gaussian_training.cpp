#include "gaussian_training.hpp"

#include "gaussian_renderer/gaussian_model.hpp"
#include "utils/logger.hpp"

#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>
#include <vector>

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
    initialized_ = true;
}

void GaussianTraining::cleanup() {
    if (backward_) {
        backward_->cleanup();
        backward_.reset();
    }

    if (forward_) {
        forward_->cleanup();
        forward_.reset();
    }
    rendererInitialized_ = false;

    destroyTrainingCommandResources();
    buffers_.cleanup();
    initialized_ = false;
}

void GaussianTraining::resize(uint32_t gaussianCount, TrainingExtent extent) {
    buffers_.resize(gaussianCount, extent);
    uploadForwardModelToBuffers();
}

void GaussianTraining::initializeTrainingRenderers(vk::Device device,
                                                   vk::PhysicalDevice physicalDevice,
                                                   vk::Queue computeQueue,
                                                   uint32_t computeQueueFamilyIndex,
                                                   uint32_t gaussianCount,
                                                   TrainingExtent extent) {
    if (rendererInitialized_) {
        return;
    }

    forward_ = std::make_unique<GaussianForwardRenderer>();
    forward_->initialize(device,
                         physicalDevice,
                         computeQueue,
                         computeQueueFamilyIndex,
                         gaussianCount,
                         extent);

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
    (void)window;
    initializeTrainingRenderers(device,
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
    }

    auto* gaussianForward = dynamic_cast<GaussianForwardRenderer*>(forward_.get());
    auto* gaussianBackward = dynamic_cast<GaussianBackwardRenderer*>(backward_.get());
    if (!gaussianForward || !gaussianBackward) {
        throw std::runtime_error("GaussianTraining currently requires GaussianForwardRenderer and GaussianBackwardRenderer");
    }

    auto pushConstants = createPushConstants();

    trainingCommandPool_.reset();
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    trainingCommandBuffer_.begin(beginInfo);

    gaussianForward->setTrainingBuffers(buffers_, trainingCommandBuffer_, pushConstants);
    gaussianForward->forward();
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
    uploadForwardModelToBuffers();
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

void GaussianTraining::uploadForwardModelToBuffers() {
    if (!forwardModel_ || forwardModel_->isEmpty() || buffers_.gaussianCapacity() == 0) {
        return;
    }

    const uint32_t gaussianCount = static_cast<uint32_t>(std::distance(forwardModel_->begin(), forwardModel_->end()));
    if (gaussianCount > buffers_.gaussianCapacity()) {
        throw std::runtime_error("Forward model has more gaussians than training buffer capacity");
    }

    std::vector<GaussianTrainParam> params(gaussianCount);
    uint32_t index = 0;
    for (const auto& point : *forwardModel_) {
        GaussianTrainParam param{};
        param.positionOpacity = glm::vec4(point.position, point.alpha);
        param.scale = glm::vec4(point.scale, 0.0f);
        param.rotation = glm::vec4(point.rotation.x,
                                   point.rotation.y,
                                   point.rotation.z,
                                   point.rotation.w);
        param.sh[0] = glm::vec4(point.color.sh0, 0.0f);
        for (int i = 0; i < 3; ++i) {
            param.sh[1 + i] = glm::vec4(point.color.sh1[i], 0.0f);
        }
        for (int i = 0; i < 5; ++i) {
            param.sh[4 + i] = glm::vec4(point.color.sh2[i], 0.0f);
        }
        for (int i = 0; i < 7; ++i) {
            param.sh[9 + i] = glm::vec4(point.color.sh3[i], 0.0f);
        }
        params[index++] = param;
    }

    buffers_.uploadGaussianParams(params.data(), gaussianCount);
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

} // namespace vk_gs
