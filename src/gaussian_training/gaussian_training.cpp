#include "gaussian_training.hpp"

#include "utils/logger.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace vk_gs {

namespace {

float logit(float value) {
    constexpr float epsilon = 1e-6f;
    const float clamped = std::clamp(value, epsilon, 1.0f - epsilon);
    return std::log(clamped / (1.0f - clamped));
}

glm::vec3 colorToSH0(const glm::vec3& color) {
    constexpr float shC0 = 0.28209479177387814f;
    return (color - glm::vec3(0.5f)) / shC0;
}

} // namespace

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

    if (densification_) {
        densification_->cleanup();
        densification_.reset();
    }

    if (forward_) {
        forward_->cleanup();
        forward_.reset();
    }
    rendererInitialized_ = false;

    destroyTrainingCommandResources();
    buffers_.cleanup();
    trainableGaussianCount_ = 0;
    usedRandomInitialization_ = false;
    initialized_ = false;
}

void GaussianTraining::resize(uint32_t gaussianCount, TrainingExtent extent) {
    buffers_.resize(gaussianCount, extent);
    trainableGaussianCount_ = gaussianCount;
    sceneExtent_ = std::max(sceneExtent_, 1.0f);
}

void GaussianTraining::initializeTrainingRenderers(vk::Device device,
                                                   vk::PhysicalDevice physicalDevice,
                                                   vk::Queue computeQueue,
                                                   uint32_t computeQueueFamilyIndex,
                                                   uint32_t gaussianCount,
                                                   TrainingExtent extent) {
    if (rendererInitialized_) {
        if (forward_) {
            forward_->cleanup();
            forward_.reset();
        }
        if (backward_) {
            backward_->cleanup();
            backward_.reset();
        }
        if (densification_) {
            densification_->cleanup();
            densification_.reset();
        }
        rendererInitialized_ = false;
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

    densification_ = std::make_unique<GaussianDensificationRenderer>();
    densification_->initialize(device,
                               physicalDevice,
                               computeQueue,
                               computeQueueFamilyIndex);

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

    if (!hasTrainableModel()) {
        throw std::runtime_error("GaussianTraining::trainStep called without initialized training gaussians");
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
    gaussianForward->prepareTileItems();

    trainingCommandBuffer_.end();

    vk::SubmitInfo prepareSubmitInfo{};
    prepareSubmitInfo.setCommandBufferCount(1)
                     .setPCommandBuffers(&trainingCommandBuffer_);
    computeQueue_.submit(prepareSubmitInfo);
    computeQueue_.waitIdle();

    const uint32_t requiredTileItems = buffers_.requiredTileItemCount();
    if (requiredTileItems > buffers_.tileItemCapacity()) {
        const uint64_t doubledCapacity = static_cast<uint64_t>(buffers_.tileItemCapacity()) * 2ull;
        const uint32_t grownCapacity = static_cast<uint32_t>(
            std::min<uint64_t>(std::max<uint64_t>(requiredTileItems, doubledCapacity),
                               std::numeric_limits<uint32_t>::max()));
        LOG_INFO("Resizing training tile item buffer from {} to {} entries",
                 buffers_.tileItemCapacity(), grownCapacity);
        buffers_.resizeTileItems(grownCapacity);
    }

    const bool runDensificationThisStep = shouldRunDensification();
    const bool pruneByScreenSize = (trainingIteration_ + 1u) > densificationConfig_.opacityResetInterval;
    if (runDensificationThisStep) {
        const uint32_t possibleGrowth = trainableGaussianCount_ *
                                        std::max(densificationConfig_.splitChildren, 2u);
        const uint32_t requestedCapacity = std::min(densificationConfig_.maxGaussianCount,
                                                    std::max(trainableGaussianCount_, possibleGrowth));
        buffers_.ensureDensificationCapacity(std::max(requestedCapacity, trainableGaussianCount_));
    }

    trainingCommandPool_.reset();
    trainingCommandBuffer_.begin(beginInfo);

    gaussianForward->setTrainingBuffers(buffers_, trainingCommandBuffer_, pushConstants);
    gaussianForward->renderPreparedTiles();
    gaussianBackward->setTrainingBuffers(buffers_, trainingCommandBuffer_, pushConstants);
    gaussianBackward->backward();
    gaussianBackward->gradientDescent();

    if (runDensificationThisStep) {
        densification_->setTrainingBuffers(buffers_,
                                           trainingCommandBuffer_,
                                           createDensificationPushConstants(pruneByScreenSize));
        densification_->densifyAndPrune();
    }

    trainingCommandBuffer_.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBufferCount(1)
              .setPCommandBuffers(&trainingCommandBuffer_);
    computeQueue_.submit(submitInfo);
    computeQueue_.waitIdle();

    if (runDensificationThisStep) {
        const uint32_t newGaussianCount = std::min(buffers_.densifiedGaussianCount(),
                                                   densificationConfig_.maxGaussianCount);
        if (newGaussianCount > 0 && newGaussianCount != trainableGaussianCount_) {
            LOG_INFO("Training densify/prune changed gaussian count from {} to {} at iteration {}",
                     trainableGaussianCount_,
                     newGaussianCount,
                     trainingIteration_ + 1u);
        }
        if (newGaussianCount > 0) {
            buffers_.adoptDensifiedGaussians(newGaussianCount);
            trainableGaussianCount_ = newGaussianCount;
        }
    }

    ++trainingIteration_;
    if (hasDataset()) {
        currentDatasetFrameIndex_ = (currentDatasetFrameIndex_ + 1) % dataset_.size();
    }
}

void GaussianTraining::loadMipNeRF360Dataset(const std::filesystem::path& sceneRoot,
                                             uint32_t preferredDownscale) {
    dataset_ = TrainingDatasetLoader::loadMipNeRF360Scene(sceneRoot, preferredDownscale);
    currentDatasetFrameIndex_ = 0;

    const auto& firstFrame = dataset_.frames.front();
    LOG_INFO("GaussianTraining loaded dataset {} ({} frames, {} sparse points, {}x{}, downscale {})",
             dataset_.sceneRoot.string(),
             dataset_.size(),
             dataset_.sparsePoints.size(),
             firstFrame.width,
             firstFrame.height,
             dataset_.imageDownscale);
}

void GaussianTraining::initializeModelFromDataset(const TrainingInitializationConfig& config) {
    if (!initialized_) {
        throw std::runtime_error("GaussianTraining must be initialized before dataset model initialization");
    }
    if (dataset_.empty()) {
        throw std::runtime_error("Load a training dataset before initializing training gaussians");
    }
    usedRandomInitialization_ = dataset_.sparsePoints.empty();
    std::vector<GaussianTrainParam> params = usedRandomInitialization_
        ? createRandomInitialGaussians(config)
        : createSparsePointInitialGaussians();
    if (params.empty()) {
        throw std::runtime_error("No Gaussian initialization data was available");
    }

    const auto& firstFrame = dataset_.frames.front();
    resize(static_cast<uint32_t>(params.size()), TrainingExtent{firstFrame.width, firstFrame.height});
    buffers_.uploadGaussianParams(params.data(), static_cast<uint32_t>(params.size()));
    trainableGaussianCount_ = static_cast<uint32_t>(params.size());
    trainingIteration_ = 0;
    sceneExtent_ = estimateSceneExtent();

    if (usedRandomInitialization_) {
        LOG_INFO("Initialized training model from {} random fallback gaussians", trainableGaussianCount_);
    } else {
        LOG_INFO("Initialized training model from {} COLMAP sparse points", trainableGaussianCount_);
    }
}

bool GaussianTraining::exportToPLY(const std::filesystem::path& path) {
    if (!hasTrainableModel()) {
        throw std::runtime_error("No trained Gaussian model is available to export");
    }

    std::filesystem::path outputPath = path;
    if (outputPath.extension().empty()) {
        outputPath += ".ply";
    }
    if (outputPath.has_parent_path()) {
        std::filesystem::create_directories(outputPath.parent_path());
    }

    std::vector<GaussianTrainParam> params = buffers_.downloadGaussianParams(trainableGaussianCount_);

    std::ofstream file(outputPath, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Failed to create training PLY file: {}", outputPath.string());
        return false;
    }

    file << "ply\n";
    file << "format binary_little_endian 1.0\n";
    file << "element vertex " << params.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property float f_dc_0\n";
    file << "property float f_dc_1\n";
    file << "property float f_dc_2\n";
    for (int i = 0; i < 45; ++i) {
        file << "property float f_rest_" << i << "\n";
    }
    file << "property float opacity\n";
    file << "property float scale_0\n";
    file << "property float scale_1\n";
    file << "property float scale_2\n";
    file << "property float rot_0\n";
    file << "property float rot_1\n";
    file << "property float rot_2\n";
    file << "property float rot_3\n";
    file << "end_header\n";

    constexpr float epsilon = 1e-8f;
    for (const auto& gaussian : params) {
        const glm::vec3 position(gaussian.positionOpacity);
        file.write(reinterpret_cast<const char*>(&position.x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&position.y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&position.z), sizeof(float));

        const float fDc[3] = {
            gaussian.sh[0].x,
            gaussian.sh[0].y,
            gaussian.sh[0].z,
        };
        file.write(reinterpret_cast<const char*>(fDc), sizeof(float) * 3);

        float fRest[45]{};
        int index = 0;
        for (int channel = 0; channel < 3; ++channel) {
            for (int basis = 1; basis < 16; ++basis) {
                fRest[index++] = gaussian.sh[basis][channel];
            }
        }
        file.write(reinterpret_cast<const char*>(fRest), sizeof(float) * 45);

        const float opacity = gaussian.positionOpacity.w;
        file.write(reinterpret_cast<const char*>(&opacity), sizeof(float));

        const float logScale[3] = {
            gaussian.scale.x,
            gaussian.scale.y,
            gaussian.scale.z,
        };
        file.write(reinterpret_cast<const char*>(logScale), sizeof(float) * 3);

        const glm::vec4 rotation = glm::normalize(gaussian.rotation);
        file.write(reinterpret_cast<const char*>(&rotation.w), sizeof(float));
        file.write(reinterpret_cast<const char*>(&rotation.x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&rotation.y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&rotation.z), sizeof(float));
    }

    if (!file.good()) {
        LOG_ERROR("Failed while writing training PLY file: {}", outputPath.string());
        return false;
    }

    LOG_INFO("Exported trained Gaussian model to {} ({} points)", outputPath.string(), params.size());
    return true;
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

TrainingPushConstants GaussianTraining::createPushConstants() const {
    TrainingPushConstants pushConstants{};
    pushConstants.gaussianCount = trainableGaussianCount_;
    auto extent = buffers_.extent();
    pushConstants.width = extent.width;
    pushConstants.height = extent.height;
    pushConstants.pixelCount = extent.width * extent.height;
    pushConstants.optimizerLearningRate = optimizerConfig_.learningRate;
    pushConstants.optimizerBeta1 = optimizerConfig_.beta1;
    pushConstants.optimizerBeta2 = optimizerConfig_.beta2;
    pushConstants.optimizerEpsilon = optimizerConfig_.epsilon;
    pushConstants.optimizerGradClip = optimizerConfig_.gradClip;
    return pushConstants;
}

TrainingDensificationPushConstants GaussianTraining::createDensificationPushConstants(bool pruneByScreenSize) const {
    TrainingDensificationPushConstants pushConstants{};
    pushConstants.gaussianCount = trainableGaussianCount_;
    pushConstants.maxGaussianCount = std::min(buffers_.densificationCapacity(),
                                              densificationConfig_.maxGaussianCount);
    pushConstants.splitChildren = std::max(densificationConfig_.splitChildren, 2u);
    pushConstants.pruneByScreenSize = pruneByScreenSize ? 1u : 0u;
    pushConstants.sceneExtent = std::max(sceneExtent_, 1e-6f);
    pushConstants.gradThreshold = densificationConfig_.densifyGradThreshold;
    pushConstants.minOpacity = densificationConfig_.minOpacity;
    pushConstants.percentDense = densificationConfig_.percentDense;
    pushConstants.screenSizePruneThreshold = densificationConfig_.screenSizePruneThreshold;
    pushConstants.randomSeed = trainingIteration_ + 1u;
    const uint32_t opacityResetInterval = std::max(densificationConfig_.opacityResetInterval, 1u);
    pushConstants.resetOpacity = ((trainingIteration_ + 1u) % opacityResetInterval == 0) ? 1u : 0u;
    return pushConstants;
}

bool GaussianTraining::shouldRunDensification() const {
    if (!densificationConfig_.enabled || !densification_) {
        return false;
    }
    if (trainableGaussianCount_ == 0) {
        return false;
    }
    const uint32_t nextIteration = trainingIteration_ + 1u;
    if (nextIteration < densificationConfig_.densifyFromIteration ||
        nextIteration > densificationConfig_.densifyUntilIteration) {
        return false;
    }
    const uint32_t interval = std::max(densificationConfig_.densificationInterval, 1u);
    return nextIteration % interval == 0;
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

std::vector<GaussianTrainParam> GaussianTraining::createSparsePointInitialGaussians() const {
    std::vector<GaussianTrainParam> params;
    params.reserve(dataset_.sparsePoints.size());

    glm::vec3 minPosition(std::numeric_limits<float>::max());
    glm::vec3 maxPosition(std::numeric_limits<float>::lowest());
    for (const auto& point : dataset_.sparsePoints) {
        minPosition = glm::min(minPosition, point.position);
        maxPosition = glm::max(maxPosition, point.position);
    }

    const float sceneRadius = std::max(glm::length(maxPosition - minPosition) * 0.5f, 1.0f);
    const float pointCount = std::max(static_cast<float>(dataset_.sparsePoints.size()), 1.0f);
    const float initialScale = std::clamp(sceneRadius / std::cbrt(pointCount), 1e-4f, sceneRadius * 0.05f);
    constexpr float initialOpacity = 0.1f;
    const float rawOpacity = logit(initialOpacity);
    const float rawScale = std::log(std::max(initialScale, 1e-6f));

    for (const auto& point : dataset_.sparsePoints) {
        GaussianTrainParam param{};
        param.positionOpacity = glm::vec4(point.position, rawOpacity);
        param.scale = glm::vec4(glm::vec3(rawScale), 0.0f);
        param.rotation = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        param.sh[0] = glm::vec4(colorToSH0(point.color), 0.0f);
        params.push_back(param);
    }

    return params;
}

std::vector<GaussianTrainParam> GaussianTraining::createRandomInitialGaussians(
    const TrainingInitializationConfig& config) const {
    if (!config.allowRandomFallback) {
        throw std::runtime_error("Dataset has no COLMAP sparse points3D.bin for Gaussian initialization");
    }

    const uint32_t gaussianCount = std::max(config.randomGaussianCount, 1u);
    std::vector<GaussianTrainParam> params;
    params.reserve(gaussianCount);

    glm::vec3 minCamera(std::numeric_limits<float>::max());
    glm::vec3 maxCamera(std::numeric_limits<float>::lowest());
    glm::vec3 center(0.0f);
    for (const auto& frame : dataset_.frames) {
        minCamera = glm::min(minCamera, frame.position);
        maxCamera = glm::max(maxCamera, frame.position);
        center += frame.position;
    }

    const float frameCount = std::max(static_cast<float>(dataset_.frames.size()), 1.0f);
    center /= frameCount;

    const float cameraRadius = std::max(glm::length(maxCamera - minCamera) * 0.5f, 1.0f);
    const float sceneRadius = std::max(cameraRadius * std::max(config.sceneRadiusScale, 0.01f), 1.0f);
    const float initialScale = std::clamp(sceneRadius / std::cbrt(static_cast<float>(gaussianCount)),
                                          1e-4f,
                                          sceneRadius * 0.05f);
    const float rawOpacity = logit(config.initialOpacity);
    const float rawScale = std::log(std::max(initialScale, 1e-6f));

    std::mt19937 rng(config.randomSeed);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::uniform_real_distribution<float> colorJitter(-0.05f, 0.05f);

    for (uint32_t i = 0; i < gaussianCount; ++i) {
        GaussianTrainParam param{};
        glm::vec3 offset(unit(rng), unit(rng), unit(rng));
        param.positionOpacity = glm::vec4(center + offset * sceneRadius, rawOpacity);
        param.scale = glm::vec4(glm::vec3(rawScale), 0.0f);
        param.rotation = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        const glm::vec3 color = glm::clamp(glm::vec3(0.5f + colorJitter(rng),
                                                     0.5f + colorJitter(rng),
                                                     0.5f + colorJitter(rng)),
                                           glm::vec3(0.0f),
                                           glm::vec3(1.0f));
        param.sh[0] = glm::vec4(colorToSH0(color), 0.0f);
        params.push_back(param);
    }

    return params;
}

TrainingForwardCamera GaussianTraining::createTrainingCamera(const TrainingCameraFrame& frame) const {
    TrainingForwardCamera camera{};
    camera.view = frame.worldToCamera;
    camera.model = glm::mat4(1.0f);
    camera.projection = createProjectionMatrix(frame);

    const float width = static_cast<float>(frame.width);
    const float height = static_cast<float>(frame.height);
    camera.viewport = glm::vec4(0.0f, 0.0f, width, height);
    camera.focalTan = glm::vec4(frame.fx,
                                frame.fy,
                                width / std::max(2.0f * frame.fx, 1e-6f),
                                height / std::max(2.0f * frame.fy, 1e-6f));
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

float GaussianTraining::estimateSceneExtent() const {
    if (!dataset_.sparsePoints.empty()) {
        glm::vec3 minPosition(std::numeric_limits<float>::max());
        glm::vec3 maxPosition(std::numeric_limits<float>::lowest());
        for (const auto& point : dataset_.sparsePoints) {
            minPosition = glm::min(minPosition, point.position);
            maxPosition = glm::max(maxPosition, point.position);
        }
        return std::max(glm::length(maxPosition - minPosition), 1.0f);
    }

    if (!dataset_.frames.empty()) {
        glm::vec3 minCamera(std::numeric_limits<float>::max());
        glm::vec3 maxCamera(std::numeric_limits<float>::lowest());
        for (const auto& frame : dataset_.frames) {
            minCamera = glm::min(minCamera, frame.position);
            maxCamera = glm::max(maxCamera, frame.position);
        }
        return std::max(glm::length(maxCamera - minCamera), 1.0f);
    }

    return 1.0f;
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
