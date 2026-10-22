#include "training/core/training_step_executor.hpp"

#include "utils/logger.hpp"
#include "training/core/training_optimizer_capture_gpu.hpp"

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

void TrainingStepExecutor::synchronizeDiagnosticDownloads(vk::Queue downloadQueue) {
    if (!initialized() || !downloadQueue)
        throw std::runtime_error("Initialize the training executor before diagnostic downloads");
    const auto command = commandPool_.allocateCommandBuffer();
    command.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    vk::MemoryBarrier visible{vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite,
                               vk::AccessFlagBits::eTransferRead};
    command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                             vk::PipelineStageFlagBits::eTransfer, {}, visible, {}, {});
    command.end();
    auto fence = device_.createFenceUnique({});
    vk::SubmitInfo computeSubmit; computeSubmit.setCommandBuffers(command);
    if (downloadQueue == computeQueue_) {
        computeQueue_.submit(computeSubmit, *fence);
        (void)device_.waitForFences(*fence, VK_TRUE, UINT64_MAX);
    } else {
        auto ready = device_.createSemaphoreUnique({});
        computeSubmit.setSignalSemaphores(*ready);
        computeQueue_.submit(computeSubmit, *fence);
        const vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eTransfer;
        vk::SubmitInfo acquire; acquire.setWaitSemaphores(*ready).setWaitDstStageMask(stage);
        downloadQueue.submit(acquire);
        downloadQueue.waitIdle();
        (void)device_.waitForFences(*fence, VK_TRUE, UINT64_MAX);
    }
    commandPool_.freeCommandBuffer(command);
}

TrainingStepExecutionResult TrainingStepExecutor::execute(TrainingStepExecutionRequest request) {
    if (!initialized()) {
        throw std::runtime_error("Training step executor is not initialized");
    }

    using Clock = std::chrono::steady_clock;
    TrainingStepExecutionResult result{};
    SchedulingStepSample scheduling;
    scheduling.metadata = {
        .iteration = request.pushConstants.validationIteration,
        .frame = request.frameIndex,
        .gaussians = request.trainableGaussianCount,
        .width = request.pushConstants.width,
        .height = request.pushConstants.height,
        .tileSize = request.pushConstants.tileSize,
        .activeSHDegree = request.pushConstants.activeSHDegree,
        .ssimMode = request.ssimMode,
        .pixelMode = request.pixelMode,
        .compositeMode = request.compositeMode,
        .uploadWaitValue = request.uploadWaitValue,
        .fixedBenchmark = request.fixedBenchmark,
        .adc = request.runDensification,
        .validation = request.pushConstants.validationEnabled != 0u,
        .opacityReset = request.runDensification &&
                        request.densificationPushConstants.resetOpacity != 0u,
        .optimizerEnabled = request.pushConstants.optimizerEnabled != 0u,
        .optimizerCapture = request.optimizerCapture != nullptr,
    };
    auto* schedulingDiagnostics = request.profiling.schedulingDiagnostics();
    if (schedulingDiagnostics && !schedulingDiagnostics->wantsStep(scheduling.metadata))
        schedulingDiagnostics = nullptr;
    const auto milliseconds = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    const bool checkpointReadbackRequested =
        request.checkpointDiagnostics && request.pushConstants.validationEnabled != 0u;
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    prepareCommandBuffer_.reset();
    prepareCommandBuffer_.begin(beginInfo);
    if (request.profiling.gpuTimestampsAvailable()) {
        prepareCommandBuffer_.resetQueryPool(request.profiling.queryPool(), 0,
                                             kTrainingGpuTimestampQueryCount);
    }
    if (schedulingDiagnostics) schedulingDiagnostics->recordPrepareBegin(prepareCommandBuffer_);
    request.forward.setTrainingBuffers(request.buffers, prepareCommandBuffer_,
                                       request.pushConstants);
    request.forward.prepareTileItems();
    request.buffers.recordTileItemCountReadback(prepareCommandBuffer_);
    if (schedulingDiagnostics) schedulingDiagnostics->recordPrepareEnd(prepareCommandBuffer_);
    prepareCommandBuffer_.end();

    vk::SubmitInfo prepareSubmit{};
    prepareSubmit.setCommandBufferCount(1).setPCommandBuffers(&prepareCommandBuffer_);
    const auto prepareSubmitStart = Clock::now();
    LOG_DEBUG("Submitting training prepare pass for iteration {}",
              request.pushConstants.trainingIteration);
    device_.resetFences(prepareFence_);
    // Prepare tile coverage already updates maxScreenRadius. A later allocation/
    // recording failure must not be reported as an entirely untouched model.
    if (request.optimizerCapture) request.optimizerCapture->markPrepareSubmitted();
    const auto prepareCallStart = schedulingDiagnostics ? Clock::now() : Clock::time_point{};
    computeQueue_.submit(prepareSubmit, prepareFence_);
    const auto prepareWaitStart = schedulingDiagnostics ? Clock::now() : Clock::time_point{};
    (void)device_.waitForFences(prepareFence_, VK_TRUE, UINT64_MAX);
    const auto prepareWaitEnd = schedulingDiagnostics ? Clock::now() : Clock::time_point{};
    if (schedulingDiagnostics) {
        scheduling.prepareSubmitCallMs = milliseconds(prepareCallStart, prepareWaitStart);
        scheduling.prepareFenceWaitMs = milliseconds(prepareWaitStart, prepareWaitEnd);
    }
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
    const auto capacityStart = schedulingDiagnostics ? Clock::now() : Clock::time_point{};
    if (schedulingDiagnostics)
        scheduling.countReadMs = milliseconds(tileReadbackStart, capacityStart);
    const uint64_t childBudget = request.pushConstants.tileSize == 32u
                                     ? static_cast<uint64_t>(result.tileItemCount) / 4u
                                     : 0u;
    const uint64_t requiredItemCapacity = static_cast<uint64_t>(result.tileItemCount) + childBudget;
    if (requiredItemCapacity > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Adaptive training tile item capacity exceeds uint32 range");
    }
    if (requiredItemCapacity > request.buffers.tileItemCapacity()) {
        scheduling.metadata.resized = true;
        const auto resizeStart = Clock::now();
        const uint64_t doubled = static_cast<uint64_t>(request.buffers.tileItemCapacity()) * 2ull;
        const uint32_t capacity = static_cast<uint32_t>(
            std::min<uint64_t>(std::max<uint64_t>(requiredItemCapacity, doubled),
                               std::numeric_limits<uint32_t>::max()));
        LOG_INFO("Resizing training tile item buffer from {} to {} entries",
                 request.buffers.tileItemCapacity(), capacity);
        request.buffers.resizeTileItems(capacity);
        if (schedulingDiagnostics) scheduling.resizeMs = milliseconds(resizeStart, Clock::now());
        request.profiling.recordCpu(
            TrainingCpuProfileStage::TileBufferResize,
            std::chrono::duration<float, std::milli>(Clock::now() - resizeStart).count());
    }
    if (schedulingDiagnostics) {
        scheduling.capacityCheckMs = milliseconds(capacityStart, Clock::now());
        scheduling.metadata.tileItems = result.tileItemCount;
        scheduling.metadata.capacity = request.buffers.tileItemCapacity();
    }

    bool checkpointReady = false;
    if (request.checkpointDiagnostics) {
        const auto start = Clock::now();
        try {
            request.backward.prepareCheckpointDiagnostics(request.buffers, result.tileItemCount,
                                                           request.checkpointWarp);
            checkpointReady = true;
        } catch (const std::exception& error) {
            LOG_WARN("Checkpoint {} setup skipped; original Tensor remains active: {}",
                     request.checkpointWarp ? "P2" : "P1", error.what());
        }
        request.profiling.recordCpu(TrainingCpuProfileStage::CheckpointCapacity,
            std::chrono::duration<float, std::milli>(Clock::now() - start).count());
    } else {
        request.buffers.releaseCheckpointWorkspace();
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
    if (schedulingDiagnostics) schedulingDiagnostics->recordMainBegin(mainCommandBuffer_);
    request.forward.setTrainingBuffers(request.buffers, mainCommandBuffer_, request.pushConstants);
    LOG_DEBUG("Recording training main pass for iteration {}",
              request.pushConstants.trainingIteration);
    request.forward.renderPreparedTiles(result.tileItemCount);
    request.backward.setTrainingBuffers(request.buffers, mainCommandBuffer_, request.pushConstants);
    request.backward.setCheckpointWarp(checkpointReady && request.checkpointWarp);
    request.backward.setTensorEarlyCull(request.tensorEarlyCull);
    request.backward.setOptimizerCapture(request.optimizerCapture);
    struct ClearCaptureObserver {
        GaussianBackwardRenderer& renderer;
        ~ClearCaptureObserver() { renderer.setOptimizerCapture(nullptr); }
    } clearCaptureObserver{request.backward};
    if (checkpointReady) request.backward.recordCheckpointDiagnostics(request.checkpointResetCounters);
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
        if (request.pushConstants.tileSize == 32u) {
            request.buffers.recordTileItemCountReadback(mainCommandBuffer_);
        }
        request.validation.recordCopy(
            mainCommandBuffer_, request.buffers.validationFinalResultBuffer(), *validationTicket);
    }
    const bool readCheckpoint = checkpointReady && checkpointReadbackRequested;
    if (readCheckpoint) request.buffers.recordCheckpointReadback(mainCommandBuffer_);
    mainCommandBuffer_.end();
    if (schedulingDiagnostics) {
        scheduling.mainRecordMs = milliseconds(mainRecordStart, Clock::now());
        scheduling.metadata.validation = request.pushConstants.validationEnabled != 0u;
    }
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
    // Mark conservatively before submit: a queue/device failure cannot be called rollback.
    if (request.optimizerCapture) request.optimizerCapture->markSubmitted();
    LOG_DEBUG("Submitting training main pass for iteration {} with image upload timeline value {}",
              request.pushConstants.trainingIteration, request.uploadWaitValue);
    Clock::time_point mainCallStart{}, mainCallEnd{};
    if (validationTicket) {
        const vk::Fence validationFence = request.validation.submissionFence(*validationTicket);
        device_.resetFences(validationFence);
        if (schedulingDiagnostics) mainCallStart = Clock::now();
        computeQueue_.submit(mainSubmit, validationFence);
        if (schedulingDiagnostics) mainCallEnd = Clock::now();
        request.validation.markSubmitted(*validationTicket);
        (void)device_.waitForFences(validationFence, VK_TRUE, UINT64_MAX);
    } else {
        device_.resetFences(mainFence_);
        if (schedulingDiagnostics) mainCallStart = Clock::now();
        computeQueue_.submit(mainSubmit, mainFence_);
        if (schedulingDiagnostics) mainCallEnd = Clock::now();
        (void)device_.waitForFences(mainFence_, VK_TRUE, UINT64_MAX);
    }
    if (request.optimizerCapture) request.optimizerCapture->markCompleted();
    request.profiling.recordCpu(
        TrainingCpuProfileStage::MainSubmit,
        std::chrono::duration<float, std::milli>(Clock::now() - mainSubmitStart).count());

    const bool gpuProfilingWasAvailable = request.profiling.gpuTimestampsAvailable();
    result.gpuProfilingDisabled = gpuProfilingWasAvailable && !request.profiling.collectGpu();
    if (schedulingDiagnostics) {
        scheduling.postFenceToMainSubmitMs = milliseconds(prepareWaitEnd, mainCallStart);
        scheduling.mainSubmitCallMs = milliseconds(mainCallStart, mainCallEnd);
        schedulingDiagnostics->finishStep(scheduling);
    }
    result.optimizerUpdated = request.pushConstants.optimizerEnabled != 0u;
    result.checkpointDiagnosticsRan = checkpointReady;
    if (readCheckpoint) {
        const auto diag = request.buffers.checkpointDiagnostics();
        result.checkpointReadbackValid = true;
        result.checkpointFallbacks = diag.run[0];
        result.checkpointFlags = diag.run[1];
        result.checkpointTasks = diag.totals[1];
        LOG_INFO("Checkpoint {} layout=2: slots={} tasks={} flags={} runFlags={} fallbacks={} states(expected/written)={}/{}; warp={}",
                 request.checkpointWarp ? "P2" : "P1", diag.totals[0], diag.totals[1], diag.totals[2],
                 diag.run[1], diag.run[0], diag.totals[3], diag.diagnostics[0], diag.run[3]);
        if ((!request.checkpointWarp && diag.run[1] != 0u) ||
            (request.checkpointWarp && ((diag.run[1] & ~63u) != 0u ||
                (diag.totals[1] != 0u && diag.fallback[3] == 0u && diag.run[3] != 32u))))
            throw std::runtime_error("Checkpoint GPU contract failed; inspect diagnostic flags");
    }

    if (validationTicket && request.pushConstants.tileSize == 32u) {
        const auto adaptiveStats = request.buffers.adaptiveChildStats();
        request.validation.setAdaptiveStats(*validationTicket,
                                            TrainingAdaptiveTileReadbackStats{
                                                .childItemCount = adaptiveStats[0],
                                                .splitParentCount = adaptiveStats[1],
                                            });
    }

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
