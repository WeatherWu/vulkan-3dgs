#include "gaussian_training.hpp"

#include "utils/logger.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vulkan3DGS {

namespace {

float positionLearningRateAtStep(const TrainingOptimizerConfig& config,
                                 float step,
                                 float spatialScale) {
    const float lrInit = std::max(config.positionLearningRate * spatialScale, 0.0f);
    const float lrFinal = std::max(config.positionLearningRateFinal * spatialScale, 0.0f);
    if (lrInit == 0.0f && lrFinal == 0.0f) {
        return 0.0f;
    }

    const float maxSteps = std::max(config.positionLearningRateMaxSteps, 1.0f);
    const float t = std::clamp(step / maxSteps, 0.0f, 1.0f);
    float learningRate = std::exp(std::lerp(std::log(std::max(lrInit, 1e-12f)),
                                             std::log(std::max(lrFinal, 1e-12f)),
                                             t));
    if (config.positionLearningRateDelaySteps > 0.0f) {
        const float delayMult = std::clamp(config.positionLearningRateDelayMult, 0.0f, 1.0f);
        const float delayT = std::clamp(step / config.positionLearningRateDelaySteps, 0.0f, 1.0f);
        const float delayRate = delayMult +
            (1.0f - delayMult) * std::sin(0.5f * 3.14159265358979323846f * delayT);
        learningRate *= delayRate;
    }
    return learningRate;
}

} // namespace

GaussianTraining::GaussianTraining() = default;

GaussianTraining::~GaussianTraining() {
    cleanup();
}

void GaussianTraining::initialize(vk::Device device,
                                  vk::PhysicalDevice physicalDevice,
                                  vk::Queue transferQueue,
                                  uint32_t transferQueueFamilyIndex,
                                  uint32_t computeQueueFamilyIndex) {
    if (initialized_) {
        return;
    }

    device_ = device;
    physicalDevice_ = physicalDevice;
    transferQueue_ = transferQueue;
    transferQueueFamilyIndex_ = transferQueueFamilyIndex;
    computeQueueFamilyIndex_ = computeQueueFamilyIndex;
    buffers_.initialize(device,
                        physicalDevice,
                        transferQueue,
                        transferQueueFamilyIndex,
                        computeQueueFamilyIndex);
    initialized_ = true;
}

void GaussianTraining::cleanup() {
    imageRuntime_.cleanup(buffers_);
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

    destroyTrainingProfilingResources();
    validationService_.cleanup();
    stepExecutor_.cleanup();
    buffers_.cleanup();
    trainableGaussianCount_ = 0;
    usedRandomInitialization_ = false;
    frameScheduler_.clear();
    lastDensificationStats_ = {};
    lastDensificationStatsIteration_ = 0;
    profilingService_.resetStats();
    validationService_.resetStats();
    fixedBenchmarkStats_ = {};
    fixedBenchmarkCompletedSteps_ = 0;
    device_ = nullptr;
    physicalDevice_ = nullptr;
    transferQueue_ = nullptr;
    transferQueueFamilyIndex_ = 0;
    computeQueueFamilyIndex_ = 0;
    initialized_ = false;
}

void GaussianTraining::resize(uint32_t gaussianCount, TrainingExtent extent) {
    buffers_.resize(gaussianCount, extent);
    trainableGaussianCount_ = gaussianCount;
    optimizerStep_ = 0;
    sceneExtent_ = std::max(sceneExtent_, 1.0f);
}

void GaussianTraining::initializeTrainingRenderers(vk::Device device,
                                                   vk::PhysicalDevice physicalDevice,
                                                   vk::Queue computeQueue,
                                                   uint32_t computeQueueFamilyIndex,
                                                    uint32_t gaussianCount,
                                                    TrainingExtent extent) {
    destroyTrainingProfilingResources();
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
    if (auto* gaussianForward = dynamic_cast<GaussianForwardRenderer*>(forward_.get())) {
        gaussianForward->setCompositeMode(forwardCompositeMode_);
    }

    backward_ = std::make_unique<GaussianBackwardRenderer>();
    backward_->initialize(device,
                          physicalDevice,
                          computeQueue,
                          computeQueueFamilyIndex,
                          gaussianCount,
                          extent);
    if (auto* gaussianBackward = dynamic_cast<GaussianBackwardRenderer*>(backward_.get())) {
        gaussianBackward->setPixelTo2DGSMode(pixelTo2DGSMode_);
    }

    densification_ = std::make_unique<GaussianDensificationRenderer>();
    densification_->initialize(device,
                               physicalDevice,
                               computeQueue,
                               computeQueueFamilyIndex);

    computeQueueFamilyIndex_ = computeQueueFamilyIndex;
    validationService_.initialize(device, physicalDevice);
    stepExecutor_.initialize(device, computeQueue, computeQueueFamilyIndex);
    createTrainingProfilingResources(device, physicalDevice, computeQueueFamilyIndex);
    rendererInitialized_ = true;
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

    if (!stepExecutor_.initialized()) {
        LOG_WARN("GaussianTraining::trainStep called without initialized step executor");
        return;
    }

    if (!hasTrainableModel()) {
        throw std::runtime_error("GaussianTraining::trainStep called without initialized training gaussians");
    }

    const bool fixedBenchmarkThisStep = fixedBenchmarkStats_.active;
    if (!fixedBenchmarkThisStep && isTrainingComplete()) {
        return;
    }

    using Clock = std::chrono::steady_clock;
    const auto totalStart = Clock::now();
    resetProfilingLastSamples();

    const bool runDensificationThisStep = !fixedBenchmarkThisStep && shouldRunDensification();
    if (runDensificationThisStep) {
        imageRuntime_.reserveForDensification(trainingIteration_, buffers_);
    } else if (!fixedBenchmarkThisStep && (trainingIteration_ % 16u) == 0u) {
        imageRuntime_.refreshBudget(trainingIteration_, buffers_);
    }

    if (hasDataset()) {
        const auto frameUploadStart = Clock::now();
        if (!fixedBenchmarkThisStep) {
            (void)frameScheduler_.selectForIteration();
            prefetchUpcomingTrainingFrames();
        }
        uploadCurrentTrainingFrame();
        recordCpuProfilingSample(TrainingCpuProfileStage::FrameUpload,
                                 std::chrono::duration<float, std::milli>(Clock::now() - frameUploadStart).count());
    }

    auto* gaussianForward = dynamic_cast<GaussianForwardRenderer*>(forward_.get());
    auto* gaussianBackward = dynamic_cast<GaussianBackwardRenderer*>(backward_.get());
    if (!gaussianForward || !gaussianBackward) {
        throw std::runtime_error("GaussianTraining currently requires GaussianForwardRenderer and GaussianBackwardRenderer");
    }

    auto pushConstants = createPushConstants();
    const uint32_t currentIteration = trainingIteration_ + 1u;
    const bool beforeFinalIteration =
        frameScheduler_.totalIterations() == 0u ||
        currentIteration < frameScheduler_.totalIterations();
    pushConstants.optimizerEnabled = !fixedBenchmarkThisStep &&
                                     !runDensificationThisStep &&
                                     beforeFinalIteration ? 1u : 0u;
    if (fixedBenchmarkThisStep) {
        const uint32_t totalBenchmarkSteps = fixedBenchmarkStats_.warmupSteps +
                                             fixedBenchmarkStats_.measuredSteps;
        const bool finalBenchmarkStep = fixedBenchmarkCompletedSteps_ + 1u >= totalBenchmarkSteps;
        pushConstants.validationEnabled = finalBenchmarkStep ? 1u : 0u;
        pushConstants.validationIteration = currentIteration;
    }

    const bool pruneByScreenSize =
        currentIteration > densificationConfig_.opacityResetInterval;
    const TrainingStepExecutionResult execution = stepExecutor_.execute({
        .buffers = buffers_,
        .forward = *gaussianForward,
        .backward = *gaussianBackward,
        .densification = *densification_,
        .validation = validationService_,
        .profiling = profilingService_,
        .pushConstants = pushConstants,
        .densificationPushConstants =
            createDensificationPushConstants(pruneByScreenSize),
        .uploadSemaphore = imageRuntime_.uploadSemaphore(),
        .uploadWaitValue = imageRuntime_.pendingUploadValue(),
        .trainableGaussianCount = trainableGaussianCount_,
        .maxGaussianCount = densificationConfig_.maxGaussianCount,
        .splitChildren = densificationConfig_.splitChildren,
        .runDensification = runDensificationThisStep,
        .pruneByScreenSize = pruneByScreenSize,
    });
    imageRuntime_.clearPendingUpload();
    if (execution.gpuProfilingDisabled) {
        detachProfilingQueryPool();
    }
    if (execution.optimizerUpdated) {
        ++optimizerStep_;
    }
    if (execution.densificationRan) {
        lastDensificationStats_ = execution.densification;
        lastDensificationStatsIteration_ = currentIteration;
        LOG_INFO(
            "Training densify/prune stats at iteration {}: output={} kept={} cloned={} split={} pruned={} pruneHits(opacity={}, screen={}, world={})",
            currentIteration, execution.densification.outputCount,
            execution.densification.keptSources,
            execution.densification.cloneSources,
            execution.densification.splitSources,
            execution.densification.prunedSources,
            execution.densification.pruneOpacityHits,
            execution.densification.pruneScreenHits,
            execution.densification.pruneWorldHits);
        if (execution.gaussianCount != trainableGaussianCount_) {
            LOG_INFO(
                "Training densify/prune changed gaussian count from {} to {} at iteration {}",
                trainableGaussianCount_, execution.gaussianCount,
                currentIteration);
        }
        trainableGaussianCount_ = execution.gaussianCount;
    }

    recordCpuProfilingSample(TrainingCpuProfileStage::Total,
                             std::chrono::duration<float, std::milli>(Clock::now() - totalStart).count());

    if (fixedBenchmarkThisStep) {
        ++fixedBenchmarkCompletedSteps_;
        fixedBenchmarkStats_.completedWarmupSteps = std::min(
            fixedBenchmarkCompletedSteps_, fixedBenchmarkStats_.warmupSteps);
        fixedBenchmarkStats_.completedMeasuredSteps = fixedBenchmarkCompletedSteps_ >
                                                      fixedBenchmarkStats_.warmupSteps
            ? std::min(fixedBenchmarkCompletedSteps_ - fixedBenchmarkStats_.warmupSteps,
                       fixedBenchmarkStats_.measuredSteps)
            : 0u;
        if (fixedBenchmarkStats_.warmupSteps > 0u &&
            fixedBenchmarkCompletedSteps_ == fixedBenchmarkStats_.warmupSteps) {
            resetProfilingStats();
            resetCandidateProfileStats();
        }
        const uint32_t totalBenchmarkSteps = fixedBenchmarkStats_.warmupSteps +
                                             fixedBenchmarkStats_.measuredSteps;
        if (fixedBenchmarkCompletedSteps_ >= totalBenchmarkSteps) {
            fixedBenchmarkStats_.active = false;
            fixedBenchmarkStats_.complete = true;
            LOG_INFO("Fixed workload benchmark completed: frame {}, warmup {}, measured {}",
                     fixedBenchmarkStats_.frameIndex,
                     fixedBenchmarkStats_.warmupSteps,
                     fixedBenchmarkStats_.measuredSteps);
        }
        return;
    }

    ++trainingIteration_;
    if (hasDataset()) frameScheduler_.advanceAfterIteration();
}

void GaussianTraining::loadMipNeRF360Dataset(const std::filesystem::path& sceneRoot,
                                             uint32_t preferredDownscale) {
    dataset_ = TrainingDatasetLoader::loadMipNeRF360Scene(sceneRoot, preferredDownscale);
    frameScheduler_.setFrameCount(dataset_.size());

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
    TrainingModelInitialization initialization =
        TrainingModelIO::initialize(dataset_, config);
    usedRandomInitialization_ = initialization.usedRandomFallback;
    std::vector<GaussianTrainParam> params =
        std::move(initialization.parameters);
    if (params.empty()) {
        throw std::runtime_error("No Gaussian initialization data was available");
    }

    const auto& firstFrame = dataset_.frames.front();
    resize(static_cast<uint32_t>(params.size()), TrainingExtent{firstFrame.width, firstFrame.height});
    buffers_.uploadGaussianParams(params.data(), static_cast<uint32_t>(params.size()));
    trainableGaussianCount_ = static_cast<uint32_t>(params.size());
    trainingIteration_ = 0;
    optimizerStep_ = 0;
    frameScheduler_.restartSequence();
    imageRuntime_.initialize(device_, physicalDevice_, transferQueue_,
                             transferQueueFamilyIndex_,
                             computeQueueFamilyIndex_, dataset_, buffers_);
    lastDensificationStats_ = {};
    lastDensificationStatsIteration_ = 0;
    validationService_.resetStats();
    resetCandidateProfileStats();
    resetProfilingStats();
    sceneExtent_ = initialization.sceneExtent;

    if (usedRandomInitialization_) {
        LOG_INFO("Initialized training model from {} random fallback gaussians", trainableGaussianCount_);
    } else {
        LOG_INFO("Initialized training model from {} COLMAP sparse points", trainableGaussianCount_);
    }
    LOG_INFO("Training scene extent set to {} using camera-center normalization", sceneExtent_);
}

bool GaussianTraining::exportToPLY(const std::filesystem::path& path) {
    if (!hasTrainableModel()) {
        throw std::runtime_error("No trained Gaussian model is available to export");
    }

    const std::vector<GaussianTrainParam> parameters =
        buffers_.downloadGaussianParams(trainableGaussianCount_);
    return TrainingModelIO::writePly(path, parameters);
}

void GaussianTraining::setTrainingFrameIndex(size_t frameIndex) {
    if (dataset_.empty()) {
        throw std::runtime_error("Cannot set training frame index without a loaded dataset");
    }
    if (frameIndex >= dataset_.size()) {
        throw std::runtime_error("Training frame index is out of range");
    }
    frameScheduler_.setCurrentFrame(frameIndex);
}

void GaussianTraining::startFixedWorkloadBenchmark(const TrainingFixedBenchmarkConfig& config) {
    if (!initialized_ || !rendererInitialized_) {
        throw std::runtime_error("Initialize training before starting a fixed workload benchmark");
    }
    if (!hasTrainableModel()) {
        throw std::runtime_error("A trainable Gaussian model is required for a fixed workload benchmark");
    }
    if (dataset_.empty()) {
        throw std::runtime_error("A loaded dataset is required for a fixed workload benchmark");
    }
    if (config.frameIndex >= dataset_.size()) {
        throw std::runtime_error("Fixed workload benchmark frame index is out of range");
    }

    validationService_.collect(
        buffers_.extent(), buffers_.gaussianCapacity());
    frameScheduler_.setCurrentFrame(config.frameIndex);
    fixedBenchmarkStats_ = {};
    fixedBenchmarkStats_.active = true;
    fixedBenchmarkStats_.frameIndex = config.frameIndex;
    fixedBenchmarkStats_.warmupSteps = config.warmupSteps;
    fixedBenchmarkStats_.measuredSteps = std::max(config.measuredSteps, 1u);
    fixedBenchmarkCompletedSteps_ = 0u;
    resetProfilingStats();
    resetCandidateProfileStats();
    LOG_INFO("Started fixed workload benchmark: frame {}, warmup {}, measured {}",
             fixedBenchmarkStats_.frameIndex,
             fixedBenchmarkStats_.warmupSteps,
             fixedBenchmarkStats_.measuredSteps);
}

void GaussianTraining::stopFixedWorkloadBenchmark() {
    fixedBenchmarkStats_.active = false;
    fixedBenchmarkStats_.complete = false;
}

void GaussianTraining::setScheduleConfig(const TrainingScheduleConfig& config) {
    frameScheduler_.configure(config);
}

bool GaussianTraining::isTrainingComplete() const {
    return frameScheduler_.isComplete(trainingIteration_);
}

TrainingPushConstants GaussianTraining::createPushConstants() const {
    TrainingPushConstants pushConstants{};
    pushConstants.gaussianCount = trainableGaussianCount_;
    auto extent = buffers_.extent();
    pushConstants.width = extent.width;
    pushConstants.height = extent.height;
    pushConstants.pixelCount = extent.width * extent.height;
    pushConstants.trainingIteration = trainingIteration_;
    const uint32_t shInterval = std::max(optimizerConfig_.shDegreeInterval, 1u);
    pushConstants.maxSHDegree = std::min(optimizerConfig_.maxSHDegree, 3u);
    pushConstants.activeSHDegree = std::min((trainingIteration_ + 1u) / shInterval, pushConstants.maxSHDegree);
    const float spatialLearningRateScale = std::max(sceneExtent_, 1e-6f);
    pushConstants.positionLearningRate = positionLearningRateAtStep(
        optimizerConfig_, static_cast<float>(trainingIteration_ + 1u), spatialLearningRateScale);
    pushConstants.featureLearningRate = optimizerConfig_.featureLearningRate;
    pushConstants.featureRestLearningRate = optimizerConfig_.featureRestLearningRate;
    pushConstants.opacityLearningRate = optimizerConfig_.opacityLearningRate;
    pushConstants.scaleLearningRate = optimizerConfig_.scaleLearningRate;
    pushConstants.rotationLearningRate = optimizerConfig_.rotationLearningRate;
    pushConstants.optimizerBeta1 = std::clamp(optimizerConfig_.beta1, 0.0f, 0.999999f);
    pushConstants.optimizerBeta2 = std::clamp(optimizerConfig_.beta2, 0.0f, 0.999999f);
    pushConstants.optimizerOneMinusBeta1 = 1.0f - pushConstants.optimizerBeta1;
    pushConstants.optimizerOneMinusBeta2 = 1.0f - pushConstants.optimizerBeta2;
    const float optimizerStep = static_cast<float>(std::max(optimizerStep_ + 1u, 1u));
    pushConstants.optimizerInvFirstMomentCorrection = 1.0f / std::max(
        1.0f - std::pow(pushConstants.optimizerBeta1, optimizerStep), 1e-8f);
    pushConstants.optimizerInvSecondMomentCorrection = 1.0f / std::max(
        1.0f - std::pow(pushConstants.optimizerBeta2, optimizerStep), 1e-8f);
    pushConstants.optimizerEpsilon = optimizerConfig_.epsilon;
    pushConstants.optimizerGradClip = optimizerConfig_.gradClip;
    pushConstants.lossDssimWeight = std::clamp(optimizerConfig_.lossDssimWeight, 0.0f, 1.0f);
    pushConstants.optimizerStep = optimizerStep_ + 1u;
    const uint32_t nextIteration = trainingIteration_ + 1u;
    pushConstants.validationEnabled =
        (nextIteration <= 5u ||
         (validationInterval_ > 0u && nextIteration % validationInterval_ == 0u))
            ? 1u
            : 0u;
    pushConstants.validationIteration = nextIteration;
    pushConstants.pixelTo2DGSMinSubgroupUtilization =
        std::clamp(pixelTo2DGSMinSubgroupUtilization_, 0.0f, 1.0f);
    return pushConstants;
}

void GaussianTraining::setPixelTo2DGSMode(TrainingPixelTo2DGSMode mode) {
    pixelTo2DGSMode_ = mode;
    if (auto* gaussianBackward = dynamic_cast<GaussianBackwardRenderer*>(backward_.get())) {
        gaussianBackward->setPixelTo2DGSMode(mode);
    }
}

void GaussianTraining::setForwardCompositeMode(TrainingForwardCompositeMode mode) {
    forwardCompositeMode_ = mode;
    if (auto* gaussianForward = dynamic_cast<GaussianForwardRenderer*>(forward_.get())) {
        gaussianForward->setCompositeMode(mode);
    }
}

bool GaussianTraining::subgroupPixelTo2DGSSupported() const {
    const auto* gaussianBackward = dynamic_cast<const GaussianBackwardRenderer*>(backward_.get());
    return gaussianBackward && gaussianBackward->subgroupPixelTo2DGSSupported();
}

bool GaussianTraining::tileGaussianPixelTo2DGSSupported() const {
    const auto* gaussianBackward = dynamic_cast<const GaussianBackwardRenderer*>(backward_.get());
    return gaussianBackward && gaussianBackward->tileGaussianPixelTo2DGSSupported();
}

bool GaussianTraining::vkSplatPerSplatSupported() const {
    const auto* gaussianBackward = dynamic_cast<const GaussianBackwardRenderer*>(backward_.get());
    return gaussianBackward && gaussianBackward->vkSplatPerSplatSupported();
}

bool GaussianTraining::vkSplatTensorSupported() const {
    const auto* gaussianBackward = dynamic_cast<const GaussianBackwardRenderer*>(backward_.get());
    return gaussianBackward && gaussianBackward->vkSplatTensorSupported();
}

TrainingPixelTo2DGSMode GaussianTraining::activePixelTo2DGSMode() const {
    const auto* gaussianBackward = dynamic_cast<const GaussianBackwardRenderer*>(backward_.get());
    return gaussianBackward
        ? gaussianBackward->activePixelTo2DGSMode()
        : TrainingPixelTo2DGSMode::Direct;
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
    pushConstants.worldSizePruneThreshold = densificationConfig_.worldSizePruneThreshold;
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
    if (nextIteration <= densificationConfig_.densifyFromIteration ||
        nextIteration >= densificationConfig_.densifyUntilIteration) {
        return false;
    }
    const uint32_t interval = std::max(densificationConfig_.densificationInterval, 1u);
    return nextIteration % interval == 0;
}

void GaussianTraining::prefetchUpcomingTrainingFrames() {
    const std::vector<size_t> frameIndices =
        frameScheduler_.upcomingFrames(2u);
    imageRuntime_.prefetch(frameIndices);
}

void GaussianTraining::uploadCurrentTrainingFrame() {
    if (dataset_.empty()) return;

    const size_t frameIndex = frameScheduler_.currentFrame();
    const TrainingImageUploadResult result = imageRuntime_.uploadFrame(
        frameIndex, dataset_.frames[frameIndex],
        createTrainingCamera(dataset_.frames[frameIndex]), buffers_);
    recordCpuProfilingSample(TrainingCpuProfileStage::ImageRequest,
                             result.imageRequestMilliseconds);
    recordCpuProfilingSample(TrainingCpuProfileStage::TargetUpload,
                             result.targetUploadMilliseconds);
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
    projection[2][0] = 2.0f * frame.cx / width - 1.0f;
    projection[2][1] = 1.0f - 2.0f * frame.cy / height;
    projection[2][2] = farPlane / (farPlane - nearPlane);
    projection[2][3] = 1.0f;
    projection[3][2] = -(farPlane * nearPlane) / (farPlane - nearPlane);
    return projection;
}

void GaussianTraining::createTrainingProfilingResources(vk::Device device,
                                                        vk::PhysicalDevice physicalDevice,
                                                        uint32_t computeQueueFamilyIndex) {
    profilingService_.initialize(
        device, physicalDevice, computeQueueFamilyIndex);
    const vk::QueryPool queryPool = profilingService_.queryPool();

    if (auto* gaussianForward = dynamic_cast<GaussianForwardRenderer*>(forward_.get())) {
        gaussianForward->setProfilingQueryPool(queryPool);
    }
    if (auto* gaussianBackward = dynamic_cast<GaussianBackwardRenderer*>(backward_.get())) {
        gaussianBackward->setProfilingQueryPool(queryPool);
    }
    densification_->setProfilingQueryPool(queryPool);
}

void GaussianTraining::destroyTrainingProfilingResources() {
    detachProfilingQueryPool();
    profilingService_.cleanup();
}

void GaussianTraining::resetProfilingLastSamples() {
    profilingService_.resetLastSamples();
}

void GaussianTraining::recordCpuProfilingSample(TrainingCpuProfileStage stage, float milliseconds) {
    profilingService_.recordCpu(stage, milliseconds);
}

void GaussianTraining::resetProfilingStats() {
    profilingService_.resetStats();
}

void GaussianTraining::resetCandidateProfileStats() {
    validationService_.resetCandidateProfileStats();
}

void GaussianTraining::detachProfilingQueryPool() {
    if (auto* gaussianForward =
            dynamic_cast<GaussianForwardRenderer*>(forward_.get())) {
        gaussianForward->setProfilingQueryPool(nullptr);
    }
    if (auto* gaussianBackward =
            dynamic_cast<GaussianBackwardRenderer*>(backward_.get())) {
        gaussianBackward->setProfilingQueryPool(nullptr);
    }
    if (densification_) {
        densification_->setProfilingQueryPool(nullptr);
    }
}

} // namespace vulkan3DGS
