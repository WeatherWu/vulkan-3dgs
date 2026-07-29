#include "gaussian_training.hpp"

#include "image/image_decoder.hpp"
#include "utils/logger.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <random>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vulkan3DGS {

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

struct KdNode {
    uint32_t pointIndex = 0;
    int axis = 0;
    int left = -1;
    int right = -1;
};

int buildKdTreeRecursive(const std::vector<TrainingSparsePoint>& points,
                         std::vector<uint32_t>& indices,
                         std::vector<KdNode>& nodes,
                         size_t begin,
                         size_t end,
                         int depth) {
    if (begin >= end) {
        return -1;
    }

    const int axis = depth % 3;
    const size_t mid = begin + (end - begin) / 2;
    std::nth_element(indices.begin() + static_cast<std::ptrdiff_t>(begin),
                     indices.begin() + static_cast<std::ptrdiff_t>(mid),
                     indices.begin() + static_cast<std::ptrdiff_t>(end),
                     [&](uint32_t lhs, uint32_t rhs) {
                         return points[lhs].position[axis] < points[rhs].position[axis];
                     });

    const int nodeIndex = static_cast<int>(nodes.size());
    nodes.push_back(KdNode{indices[mid], axis, -1, -1});
    nodes[nodeIndex].left = buildKdTreeRecursive(points, indices, nodes, begin, mid, depth + 1);
    nodes[nodeIndex].right = buildKdTreeRecursive(points, indices, nodes, mid + 1, end, depth + 1);
    return nodeIndex;
}

void queryNearestSquaredDistances(const std::vector<TrainingSparsePoint>& points,
                                  const std::vector<KdNode>& nodes,
                                  int nodeIndex,
                                  uint32_t queryIndex,
                                  std::priority_queue<float>& nearestSquaredDistances,
                                  uint32_t neighborCount) {
    if (nodeIndex < 0) {
        return;
    }

    const KdNode& node = nodes[static_cast<size_t>(nodeIndex)];
    const glm::vec3 query = points[queryIndex].position;
    const glm::vec3 candidate = points[node.pointIndex].position;
    if (node.pointIndex != queryIndex) {
        const float dist2 = glm::dot(query - candidate, query - candidate);
        if (nearestSquaredDistances.size() < neighborCount) {
            nearestSquaredDistances.push(dist2);
        } else if (dist2 < nearestSquaredDistances.top()) {
            nearestSquaredDistances.pop();
            nearestSquaredDistances.push(dist2);
        }
    }

    const float axisDelta = query[node.axis] - candidate[node.axis];
    const int nearChild = axisDelta <= 0.0f ? node.left : node.right;
    const int farChild = axisDelta <= 0.0f ? node.right : node.left;
    queryNearestSquaredDistances(points, nodes, nearChild, queryIndex, nearestSquaredDistances, neighborCount);

    const float worstDist2 = nearestSquaredDistances.empty()
        ? std::numeric_limits<float>::infinity()
        : nearestSquaredDistances.top();
    if (nearestSquaredDistances.size() < neighborCount || axisDelta * axisDelta < worstDist2) {
        queryNearestSquaredDistances(points, nodes, farChild, queryIndex, nearestSquaredDistances, neighborCount);
    }
}

std::vector<float> estimateSparsePointScaleDistances(const std::vector<TrainingSparsePoint>& points,
                                                      float fallbackScale) {
    std::vector<float> scales(points.size(), fallbackScale);
    if (points.size() < 2) {
        return scales;
    }

    std::vector<uint32_t> indices(points.size());
    for (uint32_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    std::vector<KdNode> nodes;
    nodes.reserve(points.size());
    const int root = buildKdTreeRecursive(points, indices, nodes, 0, indices.size(), 0);
    const uint32_t neighborCount = std::min<uint32_t>(3u, static_cast<uint32_t>(points.size() - 1u));

    for (uint32_t i = 0; i < points.size(); ++i) {
        std::priority_queue<float> nearestSquaredDistances;
        queryNearestSquaredDistances(points, nodes, root, i, nearestSquaredDistances, neighborCount);
        if (nearestSquaredDistances.empty()) {
            continue;
        }

        float dist2Sum = 0.0f;
        uint32_t count = 0;
        while (!nearestSquaredDistances.empty()) {
            dist2Sum += nearestSquaredDistances.top();
            nearestSquaredDistances.pop();
            ++count;
        }

        const float meanDist2 = dist2Sum / static_cast<float>(std::max(count, 1u));
        scales[i] = std::sqrt(std::max(meanDist2, 1e-7f));
    }

    return scales;
}

uint32_t findMemoryType(vk::PhysicalDevice physicalDevice,
                        uint32_t typeFilter,
                        vk::MemoryPropertyFlags properties) {
    const vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();
    for (uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index) {
        if ((typeFilter & (1u << index)) != 0u &&
            (memoryProperties.memoryTypes[index].propertyFlags & properties) == properties) {
            return index;
        }
    }
    throw std::runtime_error("Failed to find host-visible validation readback memory");
}

uint64_t alignUp(uint64_t value, uint64_t alignment) {
    if (alignment <= 1u) {
        return value;
    }
    return ((value + alignment - 1u) / alignment) * alignment;
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
    buffers_.clearTargetColorDescriptor();
    if (deviceImageCache_) {
        deviceImageCache_->cleanup();
        deviceImageCache_.reset();
    }
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
    destroyTrainingCommandResources();
    buffers_.cleanup();
    trainableGaussianCount_ = 0;
    usedRandomInitialization_ = false;
    randomFrameStack_.clear();
    imageStreamer_.reset();
    lastDensificationStats_ = {};
    lastDensificationStatsIteration_ = 0;
    profilingStats_ = {};
    device_ = nullptr;
    physicalDevice_ = nullptr;
    transferQueue_ = nullptr;
    transferQueueFamilyIndex_ = 0;
    pendingImageUploadValue_ = 0;
    deviceCacheGrowthResumeIteration_ = 0;
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

    createTrainingCommandResources(device, computeQueue, computeQueueFamilyIndex);
    createTrainingProfilingResources(device, physicalDevice, computeQueueFamilyIndex);
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
               transferQueueFamilyIndex,
               computeQueueFamilyIndex);
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

    if (!prepareCommandBuffer_ || !mainCommandBuffer_) {
        LOG_WARN("GaussianTraining::trainStep called without training command buffers");
        return;
    }

    if (!hasTrainableModel()) {
        throw std::runtime_error("GaussianTraining::trainStep called without initialized training gaussians");
    }

    if (isTrainingComplete()) {
        return;
    }

    using Clock = std::chrono::steady_clock;
    const auto totalStart = Clock::now();
    resetProfilingLastSamples();
    profilingStats_.gpuTimestampsAvailable = gpuTimestampProfilingAvailable_;

    const bool runDensificationThisStep = shouldRunDensification();
    if (runDensificationThisStep) {
        deviceCacheGrowthResumeIteration_ = trainingIteration_ + 33u;
        refreshDeviceImageCacheBudget(true);
    } else if ((trainingIteration_ % 16u) == 0u) {
        refreshDeviceImageCacheBudget(false);
    }

    if (hasDataset()) {
        const auto frameUploadStart = Clock::now();
        selectTrainingFrameForIteration();
        prefetchUpcomingTrainingFrames();
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
    const bool beforeFinalIteration = scheduleConfig_.totalIterations == 0u ||
                                      currentIteration < scheduleConfig_.totalIterations;
    pushConstants.optimizerEnabled = !runDensificationThisStep && beforeFinalIteration ? 1u : 0u;

    prepareCommandBuffer_.reset();
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    prepareCommandBuffer_.begin(beginInfo);
    if (gpuTimestampProfilingAvailable_) {
        prepareCommandBuffer_.resetQueryPool(profilingQueryPool_,
                                              0,
                                              kTrainingGpuTimestampQueryCount);
    }

    gaussianForward->setTrainingBuffers(buffers_, prepareCommandBuffer_, pushConstants);
    gaussianForward->prepareTileItems();
    buffers_.recordTileItemCountReadback(prepareCommandBuffer_);

    prepareCommandBuffer_.end();

    vk::SubmitInfo prepareSubmitInfo{};
    prepareSubmitInfo.setCommandBufferCount(1)
                     .setPCommandBuffers(&prepareCommandBuffer_);
    const auto prepareSubmitStart = Clock::now();
    LOG_DEBUG("Submitting training prepare pass for iteration {}", trainingIteration_ + 1u);
    device_.resetFences(prepareFence_);
    computeQueue_.submit(prepareSubmitInfo, prepareFence_);
    (void)device_.waitForFences(prepareFence_, VK_TRUE, UINT64_MAX);
    LOG_DEBUG("Training prepare pass completed for iteration {}", trainingIteration_ + 1u);
    recordCpuProfilingSample(TrainingCpuProfileStage::PrepareSubmit,
                             std::chrono::duration<float, std::milli>(Clock::now() - prepareSubmitStart).count());

    const auto tileCountReadbackStart = Clock::now();
    const uint32_t requiredTileItems = buffers_.requiredTileItemCount();
    pushConstants.tileItemCount = requiredTileItems;
    recordCpuProfilingSample(TrainingCpuProfileStage::TileCountReadback,
                             std::chrono::duration<float, std::milli>(Clock::now() - tileCountReadbackStart).count());
    if (requiredTileItems > buffers_.tileItemCapacity()) {
        const auto tileResizeStart = Clock::now();
        const uint64_t doubledCapacity = static_cast<uint64_t>(buffers_.tileItemCapacity()) * 2ull;
        const uint32_t grownCapacity = static_cast<uint32_t>(
            std::min<uint64_t>(std::max<uint64_t>(requiredTileItems, doubledCapacity),
                               std::numeric_limits<uint32_t>::max()));
        LOG_INFO("Resizing training tile item buffer from {} to {} entries",
                 buffers_.tileItemCapacity(), grownCapacity);
        buffers_.resizeTileItems(grownCapacity);
        recordCpuProfilingSample(TrainingCpuProfileStage::TileBufferResize,
                                 std::chrono::duration<float, std::milli>(Clock::now() - tileResizeStart).count());
    }

    const bool pruneByScreenSize = (trainingIteration_ + 1u) > densificationConfig_.opacityResetInterval;
    if (runDensificationThisStep) {
        const uint32_t possibleGrowth = trainableGaussianCount_ *
                                        std::max(densificationConfig_.splitChildren, 2u);
        const uint32_t requestedCapacity = std::min(densificationConfig_.maxGaussianCount,
                                                    std::max(trainableGaussianCount_, possibleGrowth));
        buffers_.ensureDensificationCapacity(std::max(requestedCapacity, trainableGaussianCount_));
    }

    ValidationReadbackSlot* validationReadbackSlot = nullptr;
    if (pushConstants.validationEnabled != 0u) {
        collectCompletedValidationReadbacks();
        validationReadbackSlot = acquireValidationReadbackSlot();
        if (!validationReadbackSlot) {
            LOG_WARN("Training validation readback ring is full at iteration {}", pushConstants.validationIteration);
            pushConstants.validationEnabled = 0u;
        } else {
            validationReadbackSlot->iteration = pushConstants.validationIteration;
            validationReadbackSlot->tileItemCount = requiredTileItems;
            validationReadbackSlot->gaussianCount = trainableGaussianCount_;
        }
    }

    const auto mainRecordStart = Clock::now();
    mainCommandBuffer_.reset();
    mainCommandBuffer_.begin(beginInfo);

    gaussianForward->setTrainingBuffers(buffers_, mainCommandBuffer_, pushConstants);
    LOG_DEBUG("Recording training main pass for iteration {}", trainingIteration_ + 1u);
    gaussianForward->renderPreparedTiles(requiredTileItems);
    LOG_DEBUG("Recorded training forward pass for iteration {}", trainingIteration_ + 1u);
    gaussianBackward->setTrainingBuffers(buffers_, mainCommandBuffer_, pushConstants);
    gaussianBackward->backward();

    if (runDensificationThisStep) {
        densification_->setTrainingBuffers(buffers_,
                                           mainCommandBuffer_,
                                           createDensificationPushConstants(pruneByScreenSize));
        densification_->densifyAndPrune();
        buffers_.recordDensificationStatsReadback(mainCommandBuffer_);
        if (pushConstants.validationEnabled != 0u) {
            TrainingPushConstants validationPushConstants = pushConstants;
            validationPushConstants.gaussianCount = buffers_.densificationCapacity();
            validationPushConstants.validationUsesDensifiedGaussians = 1u;
            gaussianBackward->setTrainingBuffers(buffers_,
                                                 mainCommandBuffer_,
                                                 validationPushConstants);
        }
    }

    gaussianBackward->gradientDescent();

    if (validationReadbackSlot) {
        recordValidationReadbackCopy(*validationReadbackSlot);
    }

    mainCommandBuffer_.end();
    LOG_DEBUG("Finished recording training main command buffer for iteration {}", trainingIteration_ + 1u);
    recordCpuProfilingSample(TrainingCpuProfileStage::MainRecord,
                             std::chrono::duration<float, std::milli>(Clock::now() - mainRecordStart).count());

    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBufferCount(1)
              .setPCommandBuffers(&mainCommandBuffer_);
    vk::TimelineSemaphoreSubmitInfo uploadWaitInfo{};
    vk::PipelineStageFlags uploadWaitStage = vk::PipelineStageFlagBits::eComputeShader;
    vk::Semaphore uploadSemaphore = nullptr;
    if (pendingImageUploadValue_ > 0 && deviceImageCache_) {
        uploadSemaphore = deviceImageCache_->uploadSemaphore();
        uploadWaitInfo.setWaitSemaphoreValues(pendingImageUploadValue_);
        submitInfo.setPNext(&uploadWaitInfo)
                  .setWaitSemaphores(uploadSemaphore)
                  .setWaitDstStageMask(uploadWaitStage);
    }
    const auto mainSubmitStart = Clock::now();
    LOG_DEBUG("Submitting training main pass for iteration {} with image upload timeline value {}",
              trainingIteration_ + 1u,
              pendingImageUploadValue_);
    if (validationReadbackSlot) {
        device_.resetFences(validationReadbackSlot->fence);
        computeQueue_.submit(submitInfo, validationReadbackSlot->fence);
        validationReadbackSlot->pending = true;
        (void)device_.waitForFences(validationReadbackSlot->fence, VK_TRUE, UINT64_MAX);
    } else {
        device_.resetFences(mainFence_);
        computeQueue_.submit(submitInfo, mainFence_);
        (void)device_.waitForFences(mainFence_, VK_TRUE, UINT64_MAX);
    }
    LOG_DEBUG("Training main pass completed for iteration {}", trainingIteration_ + 1u);
    pendingImageUploadValue_ = 0;
    recordCpuProfilingSample(TrainingCpuProfileStage::MainSubmit,
                             std::chrono::duration<float, std::milli>(Clock::now() - mainSubmitStart).count());
    collectGpuProfilingStats();
    if (pushConstants.optimizerEnabled != 0u) {
        ++optimizerStep_;
    }

    const uint32_t nextIteration = currentIteration;
    if (nextIteration <= 5u ||
        (validationInterval_ > 0u && nextIteration % validationInterval_ == 0u)) {
        const auto validationStart = Clock::now();
        LOG_DEBUG("Starting training validation readback for iteration {}", nextIteration);
        collectCompletedValidationReadbacks();
        LOG_DEBUG("Completed training validation readback for iteration {}", nextIteration);
        recordCpuProfilingSample(TrainingCpuProfileStage::Validation,
                                 std::chrono::duration<float, std::milli>(Clock::now() - validationStart).count());
    }

    if (runDensificationThisStep) {
        const auto densificationAdoptStart = Clock::now();
        const TrainingDensificationStats densificationStats = buffers_.densificationStats();
        lastDensificationStats_ = densificationStats;
        lastDensificationStatsIteration_ = trainingIteration_ + 1u;
        const uint32_t newGaussianCount = std::min(densificationStats.outputCount,
                                                   densificationConfig_.maxGaussianCount);
        LOG_INFO("Training densify/prune stats at iteration {}: output={} kept={} cloned={} split={} pruned={} pruneHits(opacity={}, screen={}, world={})",
                 trainingIteration_ + 1u,
                 densificationStats.outputCount,
                 densificationStats.keptSources,
                 densificationStats.cloneSources,
                 densificationStats.splitSources,
                 densificationStats.prunedSources,
                 densificationStats.pruneOpacityHits,
                 densificationStats.pruneScreenHits,
                 densificationStats.pruneWorldHits);
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
        recordCpuProfilingSample(TrainingCpuProfileStage::DensificationAdopt,
                                 std::chrono::duration<float, std::milli>(Clock::now() - densificationAdoptStart).count());
    }

    recordCpuProfilingSample(TrainingCpuProfileStage::Total,
                             std::chrono::duration<float, std::milli>(Clock::now() - totalStart).count());

    ++trainingIteration_;
    if (hasDataset() &&
        scheduleConfig_.imageSelectionMode == TrainingImageSelectionMode::Sequential) {
        currentDatasetFrameIndex_ = (currentDatasetFrameIndex_ + 1) % dataset_.size();
    }
}

void GaussianTraining::loadMipNeRF360Dataset(const std::filesystem::path& sceneRoot,
                                             uint32_t preferredDownscale) {
    dataset_ = TrainingDatasetLoader::loadMipNeRF360Scene(sceneRoot, preferredDownscale);
    currentDatasetFrameIndex_ = 0;
    randomFrameStack_.clear();

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
    optimizerStep_ = 0;
    deviceCacheGrowthResumeIteration_ = 0;
    frameRng_.seed(scheduleConfig_.randomSeed);
    randomFrameStack_.clear();
    imageStreamer_ = std::make_unique<ImageStreamer>();
    if (imageStreamer_) {
        std::vector<ImageSourceDesc> sources;
        sources.reserve(dataset_.frames.size());
        for (size_t frameIndex = 0; frameIndex < dataset_.frames.size(); ++frameIndex) {
            const TrainingCameraFrame& frame = dataset_.frames[frameIndex];
            ImageSourceDesc source{};
            source.id = static_cast<ImageId>(frameIndex);
            source.path = frame.imagePath;
            source.expectedWidth = frame.width;
            source.expectedHeight = frame.height;
            source.format = ImagePixelFormat::Rgba8Unorm;
            source.colorSpace = ImageColorSpace::LinearUnorm;
            sources.push_back(std::move(source));
        }
        imageStreamer_->refreshMemoryBudget();
        imageStreamer_->setSources(std::move(sources));
    }
    initializeDeviceImageCache();
    lastDensificationStats_ = {};
    lastDensificationStatsIteration_ = 0;
    sceneExtent_ = estimateSceneExtent();

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

        const glm::vec4 rotation = gaussian.rotation;
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

void GaussianTraining::setScheduleConfig(const TrainingScheduleConfig& config) {
    scheduleConfig_ = config;
    frameRng_.seed(scheduleConfig_.randomSeed);
    randomFrameStack_.clear();
}

bool GaussianTraining::isTrainingComplete() const {
    return scheduleConfig_.totalIterations > 0 &&
           trainingIteration_ >= scheduleConfig_.totalIterations;
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
    pushConstants.positionLearningRate = optimizerConfig_.positionLearningRate * spatialLearningRateScale;
    pushConstants.positionLearningRateFinal = optimizerConfig_.positionLearningRateFinal * spatialLearningRateScale;
    pushConstants.positionLearningRateDelayMult = optimizerConfig_.positionLearningRateDelayMult;
    pushConstants.positionLearningRateDelaySteps = optimizerConfig_.positionLearningRateDelaySteps;
    pushConstants.positionLearningRateMaxSteps = optimizerConfig_.positionLearningRateMaxSteps;
    pushConstants.featureLearningRate = optimizerConfig_.featureLearningRate;
    pushConstants.featureRestLearningRate = optimizerConfig_.featureRestLearningRate;
    pushConstants.opacityLearningRate = optimizerConfig_.opacityLearningRate;
    pushConstants.scaleLearningRate = optimizerConfig_.scaleLearningRate;
    pushConstants.rotationLearningRate = optimizerConfig_.rotationLearningRate;
    pushConstants.optimizerBeta1 = optimizerConfig_.beta1;
    pushConstants.optimizerBeta2 = optimizerConfig_.beta2;
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

    const auto imageRequestStart = std::chrono::steady_clock::now();
    if (imageStreamer_) {
        const ImageHandle handle = imageStreamer_->request(static_cast<ImageId>(currentDatasetFrameIndex_));
        const ImageRgba8& image = handle.image();
        recordCpuProfilingSample(
            TrainingCpuProfileStage::ImageRequest,
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - imageRequestStart).count());
        if (image.width != frame.width || image.height != frame.height) {
            throw std::runtime_error("Cached training image dimensions do not match frame metadata");
        }

        const auto targetUploadStart = std::chrono::steady_clock::now();
        if (deviceImageCache_ && deviceImageCache_->isInitialized()) {
            const DeviceImageBinding binding = deviceImageCache_->getOrUpload(
                static_cast<ImageId>(currentDatasetFrameIndex_), image);
            buffers_.setTargetColorDescriptor(binding.descriptor);
            pendingImageUploadValue_ = binding.readyValue;
        } else {
            buffers_.clearTargetColorDescriptor();
            pendingImageUploadValue_ = 0;
            buffers_.uploadTargetColor(image.pixels.data(), image.width, image.height);
        }
        buffers_.uploadCamera(createTrainingCamera(frame));
        recordCpuProfilingSample(
            TrainingCpuProfileStage::TargetUpload,
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - targetUploadStart).count());
    } else {
        ImageSourceDesc source{};
        source.path = frame.imagePath;
        source.expectedWidth = frame.width;
        source.expectedHeight = frame.height;
        const ImageRgba8 targetImage = ImageDecoder::decodeRgba8(source);
        recordCpuProfilingSample(
            TrainingCpuProfileStage::ImageRequest,
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - imageRequestStart).count());
        const auto targetUploadStart = std::chrono::steady_clock::now();
        buffers_.clearTargetColorDescriptor();
        pendingImageUploadValue_ = 0;
        buffers_.uploadTargetColor(targetImage.pixels.data(), targetImage.width, targetImage.height);
        buffers_.uploadCamera(createTrainingCamera(frame));
        recordCpuProfilingSample(
            TrainingCpuProfileStage::TargetUpload,
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - targetUploadStart).count());
    }
}

void GaussianTraining::initializeDeviceImageCache() {
    buffers_.clearTargetColorDescriptor();
    if (deviceImageCache_) {
        deviceImageCache_->cleanup();
        deviceImageCache_.reset();
    }
    if (!device_ || !physicalDevice_ || !transferQueue_ || dataset_.empty()) {
        return;
    }

    const TrainingCameraFrame& firstFrame = dataset_.frames.front();
    try {
        auto cache = std::make_unique<DeviceImageCache>();
        cache->initialize(device_,
                          physicalDevice_,
                          transferQueue_,
                          transferQueueFamilyIndex_,
                          computeQueueFamilyIndex_,
                          firstFrame.width,
                          firstFrame.height,
                          static_cast<uint32_t>(dataset_.size()),
                          initialDeviceImageCacheBudget());
        const DeviceImageCacheStats cacheStats = cache->stats();
        LOG_INFO("Initialized device image cache with {} slots ({:.1f} MiB, mode {})",
                 cacheStats.slotCount,
                 static_cast<double>(cacheStats.allocatedBytes) / (1024.0 * 1024.0),
                 static_cast<uint32_t>(cacheStats.mode));
        deviceImageCache_ = std::move(cache);
    } catch (const std::exception& error) {
        LOG_WARN("Device image cache unavailable; using the per-frame target buffer: {}", error.what());
    }
}

void GaussianTraining::refreshDeviceImageCacheBudget(bool reserveForDensification) {
    if (!deviceImageCache_ || !deviceImageCache_->isInitialized()) {
        return;
    }
    const bool allowGrowth = !reserveForDensification &&
                             trainingIteration_ >= deviceCacheGrowthResumeIteration_;
    if (deviceImageCache_->refreshMemoryBudget(allowGrowth, reserveForDensification)) {
        buffers_.clearTargetColorDescriptor();
        pendingImageUploadValue_ = 0;
        const DeviceImageCacheStats cacheStats = deviceImageCache_->stats();
        LOG_INFO("Resized device image cache to {} slots ({:.1f} MiB){}",
                 cacheStats.slotCount,
                 static_cast<double>(cacheStats.allocatedBytes) / (1024.0 * 1024.0),
                 reserveForDensification ? " before densification" : "");
    }
}

uint64_t GaussianTraining::initialDeviceImageCacheBudget() const {
    if (!physicalDevice_ || dataset_.empty()) {
        return 0;
    }

    const TrainingCameraFrame& firstFrame = dataset_.frames.front();
    const uint64_t imageBytes = static_cast<uint64_t>(firstFrame.width) * firstFrame.height * sizeof(uint32_t);
    const vk::PhysicalDeviceLimits limits = physicalDevice_.getProperties().limits;
    const uint64_t slotStride = alignUp(
        imageBytes,
        std::max<uint64_t>(limits.minStorageBufferOffsetAlignment, 1u));
    const uint64_t fullDatasetBytes = slotStride * dataset_.size();
    const vk::PhysicalDeviceMemoryProperties memory = physicalDevice_.getMemoryProperties();
    uint64_t largestDeviceLocalHeap = 0;
    uint32_t deviceLocalHeapIndex = 0;
    for (uint32_t typeIndex = 0; typeIndex < memory.memoryTypeCount; ++typeIndex) {
        const vk::MemoryType& type = memory.memoryTypes[typeIndex];
        if ((type.propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal) == vk::MemoryPropertyFlagBits::eDeviceLocal &&
            memory.memoryHeaps[type.heapIndex].size > largestDeviceLocalHeap) {
            largestDeviceLocalHeap = memory.memoryHeaps[type.heapIndex].size;
            deviceLocalHeapIndex = type.heapIndex;
        }
    }

    const auto extensions = physicalDevice_.enumerateDeviceExtensionProperties();
    const bool hasMemoryBudget = std::any_of(
        extensions.begin(), extensions.end(),
        [](const vk::ExtensionProperties& extension) {
            return std::strcmp(extension.extensionName.data(), vk::EXTMemoryBudgetExtensionName) == 0;
        });
    if (hasMemoryBudget) {
        vk::PhysicalDeviceMemoryBudgetPropertiesEXT budgetProperties{};
        vk::PhysicalDeviceMemoryProperties2 memoryProperties{};
        memoryProperties.pNext = &budgetProperties;
        physicalDevice_.getMemoryProperties2(&memoryProperties);
        const uint64_t heapBudget = budgetProperties.heapBudget[deviceLocalHeapIndex];
        const uint64_t heapUsage = budgetProperties.heapUsage[deviceLocalHeapIndex];
        const uint64_t freeBytes = heapBudget > heapUsage ? heapBudget - heapUsage : 0;
        constexpr uint64_t minimumReserveBytes = uint64_t{512} * 1024u * 1024u;
        const uint64_t reserveBytes = std::min(heapBudget / 2u,
                                              std::max(minimumReserveBytes,
                                                       heapBudget * 15u / 100u));
        const uint64_t usableBytes = freeBytes > reserveBytes ? freeBytes - reserveBytes : imageBytes;
        return std::min(fullDatasetBytes,
                        std::max(imageBytes, std::min(usableBytes, heapBudget / 4u)));
    }
    const uint64_t automaticBudget = largestDeviceLocalHeap / 20u;
    return std::min(fullDatasetBytes, std::max(imageBytes, automaticBudget));
}

void GaussianTraining::selectTrainingFrameForIteration() {
    if (dataset_.empty() ||
        scheduleConfig_.imageSelectionMode != TrainingImageSelectionMode::Random) {
        return;
    }

    if (randomFrameStack_.empty()) {
        randomFrameStack_.reserve(dataset_.size());
        for (size_t frameIndex = 0; frameIndex < dataset_.size(); ++frameIndex) {
            randomFrameStack_.push_back(frameIndex);
        }
        std::shuffle(randomFrameStack_.begin(), randomFrameStack_.end(), frameRng_);
    }

    currentDatasetFrameIndex_ = randomFrameStack_.back();
    randomFrameStack_.pop_back();
}

void GaussianTraining::prefetchUpcomingTrainingFrames() {
    if (!imageStreamer_ || dataset_.empty()) {
        return;
    }

    std::array<ImageId, 2> upcoming{};
    size_t count = 0;
    if (scheduleConfig_.imageSelectionMode == TrainingImageSelectionMode::Random) {
        for (auto it = randomFrameStack_.rbegin(); it != randomFrameStack_.rend() && count < upcoming.size(); ++it) {
            upcoming[count++] = static_cast<ImageId>(*it);
        }
    } else {
        for (size_t offset = 1; offset <= upcoming.size(); ++offset) {
            upcoming[count++] = static_cast<ImageId>((currentDatasetFrameIndex_ + offset) % dataset_.size());
        }
    }
    imageStreamer_->prefetch(std::span<const ImageId>(upcoming.data(), count));
}

void GaussianTraining::validateTrainingStep(const TrainingValidationGpuResult& result,
                                            uint32_t tileItemCount,
                                            uint32_t gaussianCount) {
    TrainingValidationStats stats{};
    stats.tileItemCount = tileItemCount;

    const auto extent = buffers_.extent();
    const uint32_t pixelCount = extent.width * extent.height;
    if (pixelCount == 0 || gaussianCount == 0) {
        stats.valid = false;
        validationStats_ = stats;
        LOG_WARN("Training validation failed: empty pixel or gaussian count");
        return;
    }

    stats.meanLoss = result.lossSum;
    stats.maxLoss = result.maxLoss;
    stats.invalidLossCount = result.invalidLossCount;
    stats.invalidRenderedPixelCount = result.invalidRenderedPixelCount;
    stats.nonFiniteGaussianCount = result.nonFiniteGaussianCount;
    stats.nonFinitePositionCount = result.nonFinitePositionCount;
    stats.nonFiniteOpacityCount = result.nonFiniteOpacityCount;
    stats.nonFiniteRawScaleCount = result.nonFiniteRawScaleCount;
    stats.nonFiniteActivatedScaleCount = result.nonFiniteActivatedScaleCount;
    stats.nonFiniteRotationCount = result.nonFiniteRotationCount;
    stats.nonFiniteSHCount = result.nonFiniteSHCount;
    stats.firstNonFiniteGaussianIndex = result.firstNonFiniteGaussianIndex;
    if (result.validRenderedPixelCount > 0u) {
        const float validRenderedCount = static_cast<float>(result.validRenderedPixelCount);
        stats.meanRenderedAlpha = result.alphaSum / validRenderedCount;
        stats.meanRenderedLuminance = result.luminanceSum / validRenderedCount;
    }
    const float validationPixelCount = static_cast<float>(pixelCount);
    stats.meanProcessedCandidatesPerPixel = result.processedCandidateSum / validationPixelCount;
    stats.meanContributorsPerPixel = result.contributorSum / validationPixelCount;
    stats.maxProcessedCandidatesPerPixel = result.maxProcessedCandidates;
    stats.renderedNonEmpty = stats.meanRenderedAlpha > 1e-5f ||
                             std::abs(stats.meanRenderedLuminance) > 1e-5f;

    stats.valid = stats.invalidLossCount == 0 &&
                  stats.invalidRenderedPixelCount == 0 &&
                  stats.nonFiniteGaussianCount == 0 &&
                  stats.renderedNonEmpty &&
                  stats.tileItemCount > 0 &&
                  gaussianCount <= buffers_.gaussianCapacity();
    validationStats_ = stats;

    if (!stats.valid) {
        LOG_WARN("Training validation issue at iteration {}: loss={} invalidLoss={} invalidPixels={} nonFiniteGaussians={} firstNonFinite={} categories(position={}, opacity={}, rawScale={}, activatedScale={}, rotation={}, sh={}) tileItems={} alpha={} luminance={}",
                 result.validationIteration,
                 stats.meanLoss,
                 stats.invalidLossCount,
                 stats.invalidRenderedPixelCount,
                 stats.nonFiniteGaussianCount,
                 stats.firstNonFiniteGaussianIndex,
                 stats.nonFinitePositionCount,
                 stats.nonFiniteOpacityCount,
                 stats.nonFiniteRawScaleCount,
                 stats.nonFiniteActivatedScaleCount,
                 stats.nonFiniteRotationCount,
                 stats.nonFiniteSHCount,
                 stats.tileItemCount,
                 stats.meanRenderedAlpha,
                 stats.meanRenderedLuminance);
    }
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
    const float fallbackScale = std::clamp(sceneRadius / std::cbrt(pointCount), 1e-4f, sceneRadius * 0.05f);
    const std::vector<float> initialScales =
        estimateSparsePointScaleDistances(dataset_.sparsePoints, fallbackScale);
    constexpr float initialOpacity = 0.1f;
    const float rawOpacity = logit(initialOpacity);

    for (size_t i = 0; i < dataset_.sparsePoints.size(); ++i) {
        const auto& point = dataset_.sparsePoints[i];
        const float rawScale = std::log(initialScales[i]);
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
    projection[2][0] = 2.0f * frame.cx / width - 1.0f;
    projection[2][1] = 1.0f - 2.0f * frame.cy / height;
    projection[2][2] = farPlane / (farPlane - nearPlane);
    projection[2][3] = 1.0f;
    projection[3][2] = -(farPlane * nearPlane) / (farPlane - nearPlane);
    return projection;
}

float GaussianTraining::estimateSceneExtent() const {
    if (!dataset_.frames.empty()) {
        glm::vec3 center(0.0f);
        for (const auto& frame : dataset_.frames) {
            center += frame.position;
        }
        center /= std::max(static_cast<float>(dataset_.frames.size()), 1.0f);

        float diagonal = 0.0f;
        for (const auto& frame : dataset_.frames) {
            diagonal = std::max(diagonal, glm::length(frame.position - center));
        }
        return std::max(diagonal * 1.1f, 1e-6f);
    }

    if (!dataset_.sparsePoints.empty()) {
        glm::vec3 minPosition(std::numeric_limits<float>::max());
        glm::vec3 maxPosition(std::numeric_limits<float>::lowest());
        for (const auto& point : dataset_.sparsePoints) {
            minPosition = glm::min(minPosition, point.position);
            maxPosition = glm::max(maxPosition, point.position);
        }
        return std::max(glm::length(maxPosition - minPosition), 1e-6f);
    }

    return 1.0f;
}

void GaussianTraining::createTrainingProfilingResources(vk::Device device,
                                                        vk::PhysicalDevice physicalDevice,
                                                        uint32_t computeQueueFamilyIndex) {
    profilingDevice_ = device;
    profilingStats_ = {};

    const std::vector<vk::QueueFamilyProperties> queueFamilies = physicalDevice.getQueueFamilyProperties();
    if (computeQueueFamilyIndex >= queueFamilies.size() ||
        queueFamilies[computeQueueFamilyIndex].timestampValidBits == 0u) {
        LOG_WARN("Training GPU timestamps are unavailable on the selected compute queue; CPU timings remain enabled");
        return;
    }

    timestampPeriodNanoseconds_ = physicalDevice.getProperties().limits.timestampPeriod;
    if (timestampPeriodNanoseconds_ <= 0.0f) {
        LOG_WARN("Training GPU timestamps report an invalid timestamp period; CPU timings remain enabled");
        return;
    }

    vk::QueryPoolCreateInfo queryPoolInfo{};
    queryPoolInfo.setQueryType(vk::QueryType::eTimestamp)
                 .setQueryCount(kTrainingGpuTimestampQueryCount);
    profilingQueryPool_ = profilingDevice_.createQueryPool(queryPoolInfo);
    gpuTimestampProfilingAvailable_ = true;
    profilingStats_.gpuTimestampsAvailable = true;

    if (auto* gaussianForward = dynamic_cast<GaussianForwardRenderer*>(forward_.get())) {
        gaussianForward->setProfilingQueryPool(profilingQueryPool_);
    }
    if (auto* gaussianBackward = dynamic_cast<GaussianBackwardRenderer*>(backward_.get())) {
        gaussianBackward->setProfilingQueryPool(profilingQueryPool_);
    }
    densification_->setProfilingQueryPool(profilingQueryPool_);
    LOG_INFO("Training profiling enabled: {} GPU timestamp stages, timestamp period {} ns",
             kTrainingGpuProfileStageCount,
             timestampPeriodNanoseconds_);
}

void GaussianTraining::destroyTrainingProfilingResources() {
    gpuTimestampProfilingAvailable_ = false;
    timestampPeriodNanoseconds_ = 0.0f;
    if (profilingQueryPool_) {
        profilingDevice_.destroyQueryPool(profilingQueryPool_);
        profilingQueryPool_ = nullptr;
    }
    profilingDevice_ = nullptr;
}

void GaussianTraining::resetProfilingLastSamples() {
    for (TrainingTiming& timing : profilingStats_.cpu) {
        timing.lastMs = 0.0f;
    }
    for (TrainingTiming& timing : profilingStats_.gpu) {
        timing.lastMs = 0.0f;
    }
}

void GaussianTraining::recordCpuProfilingSample(TrainingCpuProfileStage stage, float milliseconds) {
    TrainingTiming& timing = profilingStats_.cpu[static_cast<size_t>(stage)];
    timing.lastMs = milliseconds;
    ++timing.sampleCount;
    timing.averageMs += (milliseconds - timing.averageMs) / static_cast<float>(timing.sampleCount);
}

void GaussianTraining::recordGpuProfilingSample(TrainingGpuProfileStage stage, float milliseconds) {
    TrainingTiming& timing = profilingStats_.gpu[static_cast<size_t>(stage)];
    timing.lastMs = milliseconds;
    ++timing.sampleCount;
    timing.averageMs += (milliseconds - timing.averageMs) / static_cast<float>(timing.sampleCount);
}

void GaussianTraining::collectGpuProfilingStats() {
    if (!gpuTimestampProfilingAvailable_) {
        return;
    }

    struct TimestampQueryResult {
        uint64_t timestamp = 0;
        uint64_t available = 0;
    };
    std::array<TimestampQueryResult, kTrainingGpuTimestampQueryCount> timestamps{};
    const VkResult result = vkGetQueryPoolResults(
        static_cast<VkDevice>(profilingDevice_),
        static_cast<VkQueryPool>(profilingQueryPool_),
        0,
        kTrainingGpuTimestampQueryCount,
        sizeof(timestamps),
        timestamps.data(),
        sizeof(TimestampQueryResult),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (result != VK_SUCCESS && result != VK_NOT_READY) {
        LOG_WARN("Failed to read training GPU timestamps: {}. GPU timing has been disabled for this session.",
                 static_cast<int>(result));
        gpuTimestampProfilingAvailable_ = false;
        profilingStats_.gpuTimestampsAvailable = false;
        if (auto* gaussianForward = dynamic_cast<GaussianForwardRenderer*>(forward_.get())) {
            gaussianForward->setProfilingQueryPool(nullptr);
        }
        if (auto* gaussianBackward = dynamic_cast<GaussianBackwardRenderer*>(backward_.get())) {
            gaussianBackward->setProfilingQueryPool(nullptr);
        }
        densification_->setProfilingQueryPool(nullptr);
        return;
    }

    static constexpr std::array<const char*, kTrainingGpuProfileStageCount> stageNames = {
        "Prepare tile items",
        "Tile emit",
        "Tile sort and ranges",
        "Composite",
        "Loss",
        "Loss to pixel",
        "Pixel to 2DGS",
        "2DGS to 3DGS",
        "Optimizer",
        "Densification",
    };
    for (uint32_t stageIndex = 0; stageIndex < kTrainingGpuProfileStageCount; ++stageIndex) {
        const TrainingGpuProfileStage stage = static_cast<TrainingGpuProfileStage>(stageIndex);
        const TimestampQueryResult& beginResult = timestamps[trainingGpuTimestampQuery(stage, false)];
        const TimestampQueryResult& endResult = timestamps[trainingGpuTimestampQuery(stage, true)];
        if (beginResult.available == 0u || endResult.available == 0u) {
            continue;
        }
        const uint64_t begin = beginResult.timestamp;
        const uint64_t end = endResult.timestamp;
        const float milliseconds = static_cast<float>(end - begin) * timestampPeriodNanoseconds_ / 1.0e6f;
        recordGpuProfilingSample(stage, milliseconds);
        LOG_DEBUG("Training GPU stage {}: {} ms", stageNames[stageIndex], milliseconds);
    }
}

void GaussianTraining::createTrainingCommandResources(vk::Device device,
                                                      vk::Queue computeQueue,
                                                      uint32_t computeQueueFamilyIndex) {
    if (prepareCommandBuffer_ || mainCommandBuffer_) {
        return;
    }

    computeQueue_ = computeQueue;
    computeQueueFamilyIndex_ = computeQueueFamilyIndex;
    trainingCommandPool_.create(device,
                                computeQueueFamilyIndex_,
                                vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    prepareCommandBuffer_ = trainingCommandPool_.allocateCommandBuffer();
    mainCommandBuffer_ = trainingCommandPool_.allocateCommandBuffer();
    prepareFence_ = device_.createFence(vk::FenceCreateInfo{vk::FenceCreateFlagBits::eSignaled});
    mainFence_ = device_.createFence(vk::FenceCreateInfo{vk::FenceCreateFlagBits::eSignaled});
    createValidationReadbackResources();
}

void GaussianTraining::destroyTrainingCommandResources() {
    destroyValidationReadbackResources();
    if (prepareFence_ && device_) {
        (void)device_.waitForFences(prepareFence_, VK_TRUE, UINT64_MAX);
        device_.destroyFence(prepareFence_);
        prepareFence_ = nullptr;
    }
    if (mainFence_ && device_) {
        (void)device_.waitForFences(mainFence_, VK_TRUE, UINT64_MAX);
        device_.destroyFence(mainFence_);
        mainFence_ = nullptr;
    }
    if (prepareCommandBuffer_) {
        trainingCommandPool_.freeCommandBuffer(prepareCommandBuffer_);
        prepareCommandBuffer_ = nullptr;
    }
    if (mainCommandBuffer_) {
        trainingCommandPool_.freeCommandBuffer(mainCommandBuffer_);
        mainCommandBuffer_ = nullptr;
    }

    trainingCommandPool_.cleanup();
    computeQueue_ = nullptr;
    computeQueueFamilyIndex_ = 0;
}

void GaussianTraining::createValidationReadbackResources() {
    if (validationReadbackSlots_[0].buffer) {
        return;
    }

    try {
        for (auto& slot : validationReadbackSlots_) {
            vk::BufferCreateInfo bufferInfo{};
            bufferInfo.setSize(sizeof(TrainingValidationGpuResult))
                      .setUsage(vk::BufferUsageFlagBits::eTransferDst)
                      .setSharingMode(vk::SharingMode::eExclusive);
            slot.buffer = device_.createBuffer(bufferInfo);

            const vk::MemoryRequirements requirements = device_.getBufferMemoryRequirements(slot.buffer);
            vk::MemoryAllocateInfo allocationInfo{};
            allocationInfo.setAllocationSize(requirements.size)
                          .setMemoryTypeIndex(findMemoryType(
                              physicalDevice_,
                              requirements.memoryTypeBits,
                              vk::MemoryPropertyFlagBits::eHostVisible |
                                  vk::MemoryPropertyFlagBits::eHostCoherent));
            slot.memory = device_.allocateMemory(allocationInfo);
            device_.bindBufferMemory(slot.buffer, slot.memory, 0);
            slot.mapped = device_.mapMemory(slot.memory, 0, sizeof(TrainingValidationGpuResult));
            slot.fence = device_.createFence(vk::FenceCreateInfo{});
        }
    } catch (...) {
        destroyValidationReadbackResources();
        throw;
    }
    nextValidationReadbackSlot_ = 0;
}

void GaussianTraining::destroyValidationReadbackResources() {
    for (auto& slot : validationReadbackSlots_) {
        if (slot.pending && slot.fence && device_) {
            (void)device_.waitForFences(slot.fence, VK_TRUE, UINT64_MAX);
        }
        if (slot.fence && device_) {
            device_.destroyFence(slot.fence);
        }
        if (slot.mapped && slot.memory && device_) {
            device_.unmapMemory(slot.memory);
        }
        if (slot.buffer && device_) {
            device_.destroyBuffer(slot.buffer);
        }
        if (slot.memory && device_) {
            device_.freeMemory(slot.memory);
        }
        slot = {};
    }
    nextValidationReadbackSlot_ = 0;
}

GaussianTraining::ValidationReadbackSlot* GaussianTraining::acquireValidationReadbackSlot() {
    for (size_t offset = 0; offset < validationReadbackSlots_.size(); ++offset) {
        const size_t index = (nextValidationReadbackSlot_ + offset) % validationReadbackSlots_.size();
        auto& slot = validationReadbackSlots_[index];
        if (!slot.pending) {
            nextValidationReadbackSlot_ = (index + 1u) % validationReadbackSlots_.size();
            return &slot;
        }
    }
    return nullptr;
}

void GaussianTraining::recordValidationReadbackCopy(ValidationReadbackSlot& slot) {
    vk::BufferCopy copyRegion{};
    copyRegion.setSize(sizeof(TrainingValidationGpuResult));
    mainCommandBuffer_.copyBuffer(buffers_.validationFinalResultBuffer(),
                                  slot.buffer,
                                  copyRegion);
}

void GaussianTraining::collectCompletedValidationReadbacks() {
    for (auto& slot : validationReadbackSlots_) {
        if (!slot.pending) {
            continue;
        }

        const vk::Result status = device_.getFenceStatus(slot.fence);
        if (status == vk::Result::eNotReady) {
            continue;
        }
        if (status != vk::Result::eSuccess) {
            LOG_WARN("Failed to poll training validation readback fence: {}", vk::to_string(status));
            continue;
        }

        TrainingValidationGpuResult result{};
        std::memcpy(&result, slot.mapped, sizeof(result));
        if (result.validationIteration != slot.iteration) {
            LOG_WARN("Ignoring stale training validation result: expected iteration {}, got {}",
                     slot.iteration,
                     result.validationIteration);
        } else {
            validateTrainingStep(result, slot.tileItemCount, slot.gaussianCount);
        }
        slot.pending = false;
    }
}

} // namespace vulkan3DGS
