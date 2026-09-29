#include "training/core/training_step_executor.hpp"

#include "utils/logger.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <optional>
#include <stdexcept>

namespace vulkan3DGS {

TrainingStepExecutor::~TrainingStepExecutor() noexcept {
    try {
        cleanup();
    } catch (...) {
        std::fputs("TrainingStepExecutor cleanup failed during destruction\n", stderr);
    }
}

void TrainingStepExecutor::initialize(vk::Device device, vk::Queue computeQueue,
                                      uint32_t computeQueueFamilyIndex) {
    if (initialized()) return;

    device_ = device;
    computeQueue_ = computeQueue;
    computeQueueFamilyIndex_ = computeQueueFamilyIndex;
    commandPool_.create(device_, computeQueueFamilyIndex_,
                        vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    prepareCommandBuffer_ = commandPool_.allocateCommandBuffer();
    mainCommandBuffer_ = commandPool_.allocateCommandBuffer();
    prepareFence_ = device_.createFence(vk::FenceCreateInfo{vk::FenceCreateFlagBits::eSignaled});
    mainFence_ = device_.createFence(vk::FenceCreateInfo{vk::FenceCreateFlagBits::eSignaled});
}

void TrainingStepExecutor::cleanup() {
    if (prepareFence_ && device_) {
        (void)device_.waitForFences(prepareFence_, VK_TRUE, UINT64_MAX);
        device_.destroyFence(prepareFence_);
    }
    if (mainFence_ && device_) {
        (void)device_.waitForFences(mainFence_, VK_TRUE, UINT64_MAX);
        device_.destroyFence(mainFence_);
    }
    prepareFence_ = nullptr;
    mainFence_ = nullptr;
    if (prepareCommandBuffer_) {
        commandPool_.freeCommandBuffer(prepareCommandBuffer_);
        prepareCommandBuffer_ = nullptr;
    }
    if (mainCommandBuffer_) {
        commandPool_.freeCommandBuffer(mainCommandBuffer_);
        mainCommandBuffer_ = nullptr;
    }
    commandPool_.cleanup();
    device_ = nullptr;
    computeQueue_ = nullptr;
    computeQueueFamilyIndex_ = 0;
}

TrainingStepExecutionResult TrainingStepExecutor::execute(TrainingStepExecutionRequest request) {
    if (!initialized()) {
        throw std::runtime_error("Training step executor is not initialized");
    }

    using Clock = std::chrono::steady_clock;
    TrainingStepExecutionResult result{};
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    prepareCommandBuffer_.reset();
    prepareCommandBuffer_.begin(beginInfo);
    if (request.profiling.gpuTimestampsAvailable()) {
        prepareCommandBuffer_.resetQueryPool(request.profiling.queryPool(), 0,
                                             kTrainingGpuTimestampQueryCount);
    }
    request.forward.setTrainingBuffers(request.buffers, prepareCommandBuffer_,
                                       request.pushConstants);
    request.forward.prepareTileItems();
    request.buffers.recordTileItemCountReadback(prepareCommandBuffer_);
    prepareCommandBuffer_.end();

    vk::SubmitInfo prepareSubmit{};
    prepareSubmit.setCommandBufferCount(1).setPCommandBuffers(&prepareCommandBuffer_);
    const auto prepareSubmitStart = Clock::now();
    LOG_DEBUG("Submitting training prepare pass for iteration {}",
              request.pushConstants.trainingIteration);
    device_.resetFences(prepareFence_);
    computeQueue_.submit(prepareSubmit, prepareFence_);
    (void)device_.waitForFences(prepareFence_, VK_TRUE, UINT64_MAX);
    LOG_DEBUG("Training prepare pass completed for iteration {}",
              request.pushConstants.trainingIteration);
    request.profiling.recordCpu(
        TrainingCpuProfileStage::PrepareSubmit,
        std::chrono::duration<float, std::milli>(Clock::now() - prepareSubmitStart).count());

    const auto tileReadbackStart = Clock::now();
    result.tileItemCount = request.buffers.requiredTileItemCount();
    request.pushConstants.tileItemCount = result.tileItemCount;
    request.profiling.recordCpu(
        TrainingCpuProfileStage::TileCountReadback,
        std::chrono::duration<float, std::milli>(Clock::now() - tileReadbackStart).count());
    if (result.tileItemCount > request.buffers.tileItemCapacity()) {
        const auto resizeStart = Clock::now();
        const uint64_t doubled = static_cast<uint64_t>(request.buffers.tileItemCapacity()) * 2ull;
        const uint32_t capacity = static_cast<uint32_t>(
            std::min<uint64_t>(std::max<uint64_t>(result.tileItemCount, doubled),
                               std::numeric_limits<uint32_t>::max()));
        LOG_INFO("Resizing training tile item buffer from {} to {} entries",
                 request.buffers.tileItemCapacity(), capacity);
        request.buffers.resizeTileItems(capacity);
        request.profiling.recordCpu(
            TrainingCpuProfileStage::TileBufferResize,
            std::chrono::duration<float, std::milli>(Clock::now() - resizeStart).count());
    }

    if (request.runDensification) {
        const uint32_t possibleGrowth =
            request.trainableGaussianCount * std::max(request.splitChildren, 2u);
        const uint32_t requestedCapacity = std::min(
            request.maxGaussianCount, std::max(request.trainableGaussianCount, possibleGrowth));
        request.buffers.ensureDensificationCapacity(
            std::max(requestedCapacity, request.trainableGaussianCount));
    }

    std::optional<TrainingValidationService::ReadbackTicket> validationTicket;
    if (request.pushConstants.validationEnabled != 0u) {
        request.validation.collect(request.buffers.extent(), request.buffers.gaussianCapacity());
        validationTicket = request.validation.acquire(TrainingValidationReadbackMetadata{
            request.pushConstants.validationIteration, result.tileItemCount,
            request.trainableGaussianCount});
        if (!validationTicket) {
            LOG_WARN("Training validation readback ring is full at iteration {}",
                     request.pushConstants.validationIteration);
            request.pushConstants.validationEnabled = 0u;
        }
    }

    const auto mainRecordStart = Clock::now();
    mainCommandBuffer_.reset();
    mainCommandBuffer_.begin(beginInfo);
    request.forward.setTrainingBuffers(request.buffers, mainCommandBuffer_, request.pushConstants);
    LOG_DEBUG("Recording training main pass for iteration {}",
              request.pushConstants.trainingIteration);
    request.forward.renderPreparedTiles(result.tileItemCount);
    request.backward.setTrainingBuffers(request.buffers, mainCommandBuffer_, request.pushConstants);
    request.backward.backward();

    if (request.runDensification) {
        request.densification.setTrainingBuffers(request.buffers, mainCommandBuffer_,
                                                 request.densificationPushConstants);
        request.densification.densifyAndPrune();
        request.buffers.recordDensificationStatsReadback(mainCommandBuffer_);
        if (request.pushConstants.validationEnabled != 0u) {
            TrainingPushConstants validationPush = request.pushConstants;
            validationPush.gaussianCount = request.buffers.densificationCapacity();
            validationPush.validationUsesDensifiedGaussians = 1u;
            request.backward.setTrainingBuffers(request.buffers, mainCommandBuffer_,
                                                validationPush);
        }
    }
    request.backward.gradientDescent();

    if (validationTicket) {
        request.validation.recordCopy(
            mainCommandBuffer_, request.buffers.validationFinalResultBuffer(), *validationTicket);
    }
    mainCommandBuffer_.end();
    request.profiling.recordCpu(
        TrainingCpuProfileStage::MainRecord,
        std::chrono::duration<float, std::milli>(Clock::now() - mainRecordStart).count());

    vk::SubmitInfo mainSubmit{};
    mainSubmit.setCommandBufferCount(1).setPCommandBuffers(&mainCommandBuffer_);
    vk::TimelineSemaphoreSubmitInfo uploadWaitInfo{};
    vk::PipelineStageFlags uploadWaitStage = vk::PipelineStageFlagBits::eComputeShader;
    if (request.uploadWaitValue > 0u && request.uploadSemaphore) {
        uploadWaitInfo.setWaitSemaphoreValues(request.uploadWaitValue);
        mainSubmit.setPNext(&uploadWaitInfo)
            .setWaitSemaphores(request.uploadSemaphore)
            .setWaitDstStageMask(uploadWaitStage);
    }

    const auto mainSubmitStart = Clock::now();
    LOG_DEBUG("Submitting training main pass for iteration {} with image upload timeline value {}",
              request.pushConstants.trainingIteration, request.uploadWaitValue);
    if (validationTicket) {
        const vk::Fence validationFence = request.validation.submissionFence(*validationTicket);
        device_.resetFences(validationFence);
        computeQueue_.submit(mainSubmit, validationFence);
        request.validation.markSubmitted(*validationTicket);
        (void)device_.waitForFences(validationFence, VK_TRUE, UINT64_MAX);
    } else {
        device_.resetFences(mainFence_);
        computeQueue_.submit(mainSubmit, mainFence_);
        (void)device_.waitForFences(mainFence_, VK_TRUE, UINT64_MAX);
    }
    request.profiling.recordCpu(
        TrainingCpuProfileStage::MainSubmit,
        std::chrono::duration<float, std::milli>(Clock::now() - mainSubmitStart).count());

    const bool gpuProfilingWasAvailable = request.profiling.gpuTimestampsAvailable();
    result.gpuProfilingDisabled = gpuProfilingWasAvailable && !request.profiling.collectGpu();
    result.optimizerUpdated = request.pushConstants.optimizerEnabled != 0u;

    if (request.pushConstants.validationEnabled != 0u) {
        const auto validationStart = Clock::now();
        request.validation.collect(request.buffers.extent(), request.buffers.gaussianCapacity());
        request.profiling.recordCpu(
            TrainingCpuProfileStage::Validation,
            std::chrono::duration<float, std::milli>(Clock::now() - validationStart).count());
    }

    result.gaussianCount = request.trainableGaussianCount;
    if (request.runDensification) {
        const auto adoptStart = Clock::now();
        result.densification = request.buffers.densificationStats();
        const uint32_t newCount =
            std::min(result.densification.outputCount, request.maxGaussianCount);
        if (newCount > 0u) {
            request.buffers.adoptDensifiedGaussians(newCount);
            result.gaussianCount = newCount;
        }
        result.densificationRan = true;
        request.profiling.recordCpu(
            TrainingCpuProfileStage::DensificationAdopt,
            std::chrono::duration<float, std::milli>(Clock::now() - adoptStart).count());
    }
    return result;
}

} // namespace vulkan3DGS
