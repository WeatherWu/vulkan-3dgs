#include "gaussian_training.hpp"

#include "utils/logger.hpp"
#include "utils/file_utils.hpp"
#include "training/core/training_optimizer_capture_gpu.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
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
                                  uint32_t computeQueueFamilyIndex,
                                  std::optional<bool> vulkanValidationLayersEnabled) {
    if (initialized_) {
        return;
    }

    device_ = device;
    physicalDevice_ = physicalDevice;
    transferQueue_ = transferQueue;
    transferQueueFamilyIndex_ = transferQueueFamilyIndex;
    computeQueueFamilyIndex_ = computeQueueFamilyIndex;
    vulkanValidationLayersEnabled_ = vulkanValidationLayersEnabled;
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
    mortonReorder_.reset();
    mortonStats_ = {};
    buffers_.cleanup();
    trainableGaussianCount_ = 0;
    lastSortState_ = {};
    usedRandomInitialization_ = false;
    frameScheduler_.clear();
    lastDensificationStats_ = {};
    lastDensificationStatsIteration_ = 0;
    profilingService_.resetStats();
    validationService_.resetStats();
    fixedBenchmarkStats_ = {};
    fixedBenchmarkCompletedSteps_ = 0;
    pixelCullRunStats_ = {};
    device_ = nullptr;
    physicalDevice_ = nullptr;
    transferQueue_ = nullptr;
    transferQueueFamilyIndex_ = 0;
    computeQueueFamilyIndex_ = 0;
    initialized_ = false;
    vulkanValidationLayersEnabled_.reset();
}

void GaussianTraining::resize(uint32_t gaussianCount, TrainingExtent extent) {
    mortonReorder_.reset();
    mortonStats_ = {};
    buffers_.resize(gaussianCount, extent);
    buffers_.configureSsimScratch(ssimMode_);
    trainableGaussianCount_ = gaussianCount;
    optimizerStep_ = 0;
    optimizerCaptureStateUncertain_ = false;
    sceneExtent_ = std::max(sceneExtent_, 1.0f);
}

void GaussianTraining::initializeTrainingRenderers(vk::Device device,
                                                   vk::PhysicalDevice physicalDevice,
                                                   vk::Queue computeQueue,
                                                   uint32_t computeQueueFamilyIndex,
                                                    uint32_t gaussianCount,
                                                    TrainingExtent extent) {
    mortonReorder_.reset();
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
        gaussianBackward->setAdaptiveTensorChildBatch(adaptiveTensorChildBatch_);
        gaussianBackward->setTensorSubgroupReduction(tensorSubgroupReduction_);
        gaussianBackward->setTensorGlobalPixelGrad(tensorGlobalPixelGrad_);
        gaussianBackward->setTensorRgbLatencyHiding(tensorRgbLatencyHiding_);
        gaussianBackward->setSsimMode(ssimMode_);
        gaussianBackward->setAdaptiveSpatialTiles(
            tileMode_ == TrainingTileMode::AdaptiveSpatial);
        if (tileMode_ == TrainingTileMode::AdaptiveSpatial &&
            !gaussianBackward->adaptiveSpatialModeSupported()) {
            LOG_WARN("Adaptive Spatial requires a supported VkSplat Tensor or Per-Splat "
                     "backward mode; falling back to Fixed16");
            tileMode_ = TrainingTileMode::Fixed16;
            gaussianBackward->setAdaptiveSpatialTiles(false);
        }
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
    if (auto* renderer = dynamic_cast<GaussianForwardRenderer*>(forward_.get()))
        renderer->setFasterSortEnabled(fasterSortEnabled_);
    rendererInitialized_ = true;
}

void GaussianTraining::captureOptimizerStepAndPause(const std::filesystem::path& path) {
    if (trainingMortonDue(mortonOrderingEnabled_, fixedBenchmarkStats_.active, trainingIteration_,
                          mortonStats_))
        throw std::runtime_error("Optimizer capture cannot include a scheduled Z-order reorder; "
                                 "complete the step first");
    auto* renderer = dynamic_cast<GaussianBackwardRenderer*>(backward_.get());
    if (!initialized_ || !rendererInitialized_ || !stepExecutor_.initialized() || !hasDataset() ||
        !hasTrainableModel() || !renderer || !renderer->fusedOptimizerEnabled())
        throw std::runtime_error("S4 capture requires an initialized, paused dataset/model and "
                                 "original fused optimizer");
    const uint64_t next = uint64_t(trainingIteration_) + 1;
    if (fixedBenchmarkStats_.active || isTrainingComplete() ||
        (frameScheduler_.totalIterations() && next >= frameScheduler_.totalIterations()) ||
        shouldRunDensification() ||
        next % std::max(densificationConfig_.opacityResetInterval, 1u) == 0)
        throw std::runtime_error("S4 capture rejected: next step is "
                                 "benchmark/final/ADC/opacity-reset; no step was advanced");
    if (optimizerCaptureStateUncertain_ || optimizerCapture_)
        throw std::runtime_error(
            "S4 capture requires model reinitialization after uncertain submission");
    OptimizerCaptureHeader header{};
    header.gaussianCount = trainableGaussianCount_;
    header.capacity = buffers_.gaussianCapacity();
    header.pixelMode = static_cast<uint32_t>(activePixelTo2DGSMode());
    header.ssimMode = static_cast<uint32_t>(activeSsimMode());
    header.tileMode = static_cast<uint32_t>(tileMode_);
    header.compositeMode = static_cast<uint32_t>(forwardCompositeMode_);
    header.tensorFlags =
        uint32_t(tensorSubgroupReductionActive()) | (uint32_t(tensorGlobalPixelGradActive()) << 1) |
        (uint32_t(tensorRgbLatencyHidingActive()) << 2) |
        (uint32_t(tensorEarlyCullStatus() == TrainingPixelCullStatus::Active) << 3);
    fillOptimizerCaptureIdentity(header, physicalDevice_,
                                 FileUtils::resolvePath("shaders/train_backward_2dgs_to_3dgs_optimizer.comp.spv"));
    TrainingOptimizerCaptureSession capture(device_, physicalDevice_, stepExecutor_.computeQueue(),
                                            computeQueueFamilyIndex_, header, path);
    const uint32_t before = trainingIteration_;
    const auto schedulerBefore = frameScheduler_;
    optimizerCapture_ = &capture;
    try {
        trainStep();
        optimizerCapture_ = nullptr;
        if (trainingIteration_ != before + 1 || !capture.completed())
            throw std::runtime_error("S4 capture did not complete exactly one normal step");
        // No further training/reset/ADC is submitted while the worker owns this export.
        capture.save({buffers_.gaussianParamsInfo(), buffers_.adamStatesInfo(),
                      buffers_.densificationStatesInfo()});
    } catch (const std::exception& error) {
        optimizerCapture_ = nullptr;
        if (capture.submitted() && !capture.completed()) {
            try {
                stepExecutor_.computeQueue().waitIdle();
            } catch (...) {}
        }
        if (!capture.submitted()) frameScheduler_ = schedulerBefore;
        optimizerCaptureStateUncertain_ = capture.submitted() && trainingIteration_ == before;
        throw std::runtime_error(
            std::string(error.what()) +
            "; S4 step submitted=" + (capture.submitted() ? "yes" : "no") +
            ", optimizer submitted=" + (capture.optimizerSubmitted() ? "yes" : "no") +
            ", GPU completed=" + (capture.completed() ? "yes" : "no") + ", iteration advanced=" +
            (trainingIteration_ == before + 1 ? "yes (do not repeat Adam)" : "no") +
            (optimizerCaptureStateUncertain_
                 ? "; resume blocked: reinitialize model, no rollback was performed"
                 : ""));
    } catch (...) {
        optimizerCapture_ = nullptr;
        if (capture.submitted() && !capture.completed()) {
            try {
                stepExecutor_.computeQueue().waitIdle();
            } catch (...) {}
        }
        if (!capture.submitted()) frameScheduler_ = schedulerBefore;
        optimizerCaptureStateUncertain_ = capture.submitted() && trainingIteration_ == before;
        throw std::runtime_error("Unknown S4 failure; check iteration. Submitted step was NOT "
                                 "rolled back; uncertain state blocks resume");
    }
}

void GaussianTraining::trainStep() {
    if (optimizerCaptureStateUncertain_)
        throw std::runtime_error(
            "Uncertain S4 submission: reinitialize model before resuming training");
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
    if (!fixedBenchmarkThisStep) fixedBenchmarkStats_.complete = false;
    if (!fixedBenchmarkThisStep && isTrainingComplete()) {
        return;
    }

    using Clock = std::chrono::steady_clock;
    const auto totalStart = Clock::now();
    const auto pixelCullStatus = tensorEarlyCullStatus();
    resetProfilingLastSamples();

    const bool mortonThisStep = trainingMortonDue(mortonOrderingEnabled_, fixedBenchmarkThisStep,
                                                  trainingIteration_, mortonStats_);
    if (mortonThisStep) {
        const auto start = Clock::now();
        if (!mortonReorder_)
            mortonReorder_ = std::make_unique<TrainingMortonReorder>(
                device_, physicalDevice_, stepExecutor_.computeQueue(), computeQueueFamilyIndex_);
        const auto result = mortonReorder_->execute(buffers_, trainableGaussianCount_);
        ++mortonStats_.reorderCount;
        mortonStats_.lastCompletedSteps = trainingIteration_;
        mortonStats_.lastGaussianCount = trainableGaussianCount_;
        mortonStats_.lastCpuMs =
            std::chrono::duration<float, std::milli>(Clock::now() - start).count();
        mortonStats_.gpuTimed = result.gpuMs.has_value();
        mortonStats_.lastGpuMs = result.gpuMs.value_or(0);
        mortonStats_.peakWorkspaceBytes =
            std::max(mortonStats_.peakWorkspaceBytes, result.workspaceBytes);
        profilingService_.recordCpu(TrainingCpuProfileStage::MortonReorder, mortonStats_.lastCpuMs);
        if (result.gpuMs)
            profilingService_.recordGpu(TrainingGpuProfileStage::MortonReorder, *result.gpuMs);
        LOG_INFO("Training Z-order physical reorder before iteration {}: count={}, events={}, "
                 "CPU={} ms, GPU={} ms (timed={}), workspace={} bytes",
                 trainingIteration_ + 1u, trainableGaussianCount_, mortonStats_.reorderCount,
                 mortonStats_.lastCpuMs, mortonStats_.lastGpuMs, mortonStats_.gpuTimed,
                 result.workspaceBytes);
    }

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
    const auto densificationPush = createDensificationPushConstants(pruneByScreenSize);
    const uint32_t capacityBefore = buffers_.tileItemCapacity();
    const uint32_t executedFrame = static_cast<uint32_t>(currentFrameIndex());
    if (optimizerCapture_) optimizerCapture_->setFrame(static_cast<uint32_t>(currentFrameIndex()));
    const TrainingStepExecutionResult execution = stepExecutor_.execute({
        .buffers = buffers_,
        .forward = *gaussianForward,
        .backward = *gaussianBackward,
        .densification = *densification_,
        .validation = validationService_,
        .profiling = profilingService_,
        .pushConstants = pushConstants,
        .densificationPushConstants = densificationPush,
        .uploadSemaphore = imageRuntime_.uploadSemaphore(),
        .uploadWaitValue = imageRuntime_.pendingUploadValue(),
        .trainableGaussianCount = trainableGaussianCount_,
        .maxGaussianCount = densificationConfig_.maxGaussianCount,
        .splitChildren = densificationConfig_.splitChildren,
        .runDensification = runDensificationThisStep,
        .pruneByScreenSize = pruneByScreenSize,
        .checkpointDiagnostics = fixedBenchmarkThisStep && checkpointDiagnosticsRequested_,
        .checkpointWarp = fixedBenchmarkThisStep && checkpointWarpRequested_,
        .tensorEarlyCull = trainingPixelCullForStep(fixedBenchmarkThisStep,
            benchmarkTensorEarlyCullRequested_, pixelCullStatus),
        .checkpointResetCounters = fixedBenchmarkCompletedSteps_ == 0u ||
                                   fixedBenchmarkCompletedSteps_ == fixedBenchmarkStats_.warmupSteps,
        .optimizerCapture = optimizerCapture_,
        .fixedBenchmark = fixedBenchmarkThisStep,
        .frameIndex = static_cast<uint32_t>(currentFrameIndex()),
        .ssimMode = static_cast<uint32_t>(activeSsimMode()),
        .pixelMode = static_cast<uint32_t>(activePixelTo2DGSMode()),
        .compositeMode = static_cast<uint32_t>(forwardCompositeMode_),
    });
    lastSortState_ = {true, gaussianForward->fasterSortRequested(),
                      gaussianForward->fasterSortActive(), pushConstants.tileSize};
    imageRuntime_.clearPendingUpload();
    if (!fixedBenchmarkThisStep) pixelCullRunStats_.record(pixelCullStatus);
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
        if (execution.checkpointReadbackValid) {
            fixedBenchmarkStats_.checkpointFallbackCount = execution.checkpointFallbacks;
            fixedBenchmarkStats_.checkpointErrorFlags = execution.checkpointFlags;
            fixedBenchmarkStats_.checkpointTaskCount = execution.checkpointTasks;
        }
        if (checkpointDiagnosticsRequested_ && fixedBenchmarkCompletedSteps_ > fixedBenchmarkStats_.warmupSteps) {
            if (execution.checkpointDiagnosticsRan) ++fixedBenchmarkStats_.checkpointDiagnosticSteps;
            else ++fixedBenchmarkStats_.checkpointSkippedSteps;
        }
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
            const bool pureWarp = checkpointWarpRequested_ &&
                fixedBenchmarkStats_.checkpointSkippedSteps == 0u &&
                fixedBenchmarkStats_.checkpointFallbackCount == 0u &&
                fixedBenchmarkStats_.checkpointErrorFlags == 0u && execution.checkpointReadbackValid;
            if (pureWarp) {
                fixedBenchmarkStats_.pixelMode = TrainingPixelTo2DGSMode::CheckpointedWarp;
                fixedBenchmarkStats_.tensorSubgroupReduction = false;
                fixedBenchmarkStats_.tensorGlobalPixelGrad = false;
                fixedBenchmarkStats_.tensorRgbLatencyHiding = false;
            }
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
    if (stepObservationEnabled_)
        lastStepObservation_ = {
            .iteration = trainingIteration_,
            .frame = executedFrame,
            .gaussiansBefore = pushConstants.gaussianCount,
            .gaussiansAfter = trainableGaussianCount_,
            .tileItems = execution.tileItemCount,
            .tileCapacity = buffers_.tileItemCapacity(),
            .activeSHDegree = pushConstants.activeSHDegree,
            .participatingGaussians = gaussianForward->fasterSortParticipatingCount(),
            .ssimMode = static_cast<uint32_t>(activeSsimMode()),
            .pixelMode = static_cast<uint32_t>(activePixelTo2DGSMode()),
            .tileMode = static_cast<uint32_t>(tileMode_),
            .fasterSortRequested = gaussianForward->fasterSortRequested(),
            .fasterSortActive = gaussianForward->fasterSortActive(),
            .optimizerUpdated = execution.optimizerUpdated,
            .densification = execution.densificationRan,
            .opacityReset = execution.densificationRan && densificationPush.resetOpacity != 0,
            .resized = capacityBefore != buffers_.tileItemCapacity(),
            .mortonOrderingRequested = mortonOrderingEnabled_,
            .mortonReordered = mortonThisStep,
            .morton = mortonStats_,
            .validation = validationService_.stats(),
            .profiling = profilingService_.stats()};
}

TrainingStateFiniteCheck GaussianTraining::inspectTrainingStateFinite() {
    if (!trainingIteration_ || !hasTrainableModel()) throw std::runtime_error("Finite inspection requires a completed training step");
    stepExecutor_.synchronizeDiagnosticDownloads(transferQueue_);
    validationService_.collect(buffers_.extent(), buffers_.gaussianCapacity());
    OptimizerCaptureTransfer transfer(device_, physicalDevice_, transferQueue_, transferQueueFamilyIndex_);
    TrainingStateFiniteCheck result;
    result.maxLoss = -std::numeric_limits<double>::infinity();
    const uint64_t pixels = uint64_t(buffers_.extent().width) * buffers_.extent().height;
    std::vector<float> chunk(optimizerCaptureChunkBytes / sizeof(float));
    for (uint64_t offset = 0; offset < pixels;) {
        const size_t count = static_cast<size_t>(std::min<uint64_t>(chunk.size(), pixels-offset));
        transfer.download(buffers_.lossInfo(), offset*4, std::as_writable_bytes(std::span(chunk).first(count)));
        for (size_t i=0;i<count;++i) {
            ++result.lossValues;
            if (!std::isfinite(chunk[i])) ++result.invalidLossValues;
            else { result.meanLoss += double(chunk[i])/double(pixels); result.maxLoss = std::max(result.maxLoss,double(chunk[i])); }
        }
        offset += count;
    }
    if (result.invalidLossValues) result.meanLoss = result.maxLoss = std::numeric_limits<double>::quiet_NaN();
    std::vector<GaussianTrainParam> params(optimizerCaptureChunkBytes / sizeof(GaussianTrainParam));
    for (uint64_t offset=0;offset<trainableGaussianCount_;) {
        const size_t count=static_cast<size_t>(std::min<uint64_t>(params.size(),trainableGaussianCount_-offset));
        transfer.download(buffers_.gaussianParamsInfo(), offset*sizeof(GaussianTrainParam),
            std::as_writable_bytes(std::span(params).first(count)));
        for (size_t i=0;i<count;++i) {
            std::array<float,sizeof(GaussianTrainParam)/sizeof(float)> values;
            std::memcpy(values.data(),&params[i],sizeof(GaussianTrainParam));
            for(float value:values) {++result.gaussianValues;result.invalidGaussianValues+=!std::isfinite(value);}
            for(int axis=0;axis<3;++axis) result.invalidActivatedScales+=!std::isfinite(std::exp(params[i].scale[axis]));
        }
        offset+=count;
    }
    return result;
}

uint64_t GaussianTraining::diagnosticTrainingStateHash() {
    if (!hasTrainableModel()) throw std::runtime_error("State fingerprint requires an initialized model");
    stepExecutor_.synchronizeDiagnosticDownloads(transferQueue_);
    OptimizerCaptureTransfer transfer(device_, physicalDevice_, transferQueue_, transferQueueFamilyIndex_);
    std::vector<std::byte> chunk(optimizerCaptureChunkBytes);
    uint64_t hash = optimizerHash(std::as_bytes(std::span{&trainableGaussianCount_,1}));
    const std::array<vk::DescriptorBufferInfo,3> buffers{buffers_.gaussianParamsInfo(),buffers_.adamStatesInfo(),buffers_.densificationStatesInfo()};
    const std::array<size_t,3> strides{sizeof(GaussianTrainParam),sizeof(AdamState),sizeof(GaussianDensificationState)};
    for(size_t block=0;block<buffers.size();++block) {
        const uint64_t bytes=uint64_t(trainableGaussianCount_)*strides[block];
        for(uint64_t offset=0;offset<bytes;) {
            const size_t count=static_cast<size_t>(std::min<uint64_t>(chunk.size(),bytes-offset));
            transfer.download(buffers[block],offset,std::span(chunk).first(count));
            hash=optimizerHash(std::span<const std::byte>(chunk).first(count),hash);offset+=count;
        }
    }
    return hash;
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
    lastSortState_ = {};
    optimizerStep_ = 0;
    optimizerCaptureStateUncertain_ = false;
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

void GaussianTraining::exportFixedBenchmarkGradients(const std::filesystem::path& path) {
    if (!fixedBenchmarkStats_.complete || fixedBenchmarkStats_.active) {
        throw std::runtime_error("Complete a Fixed Benchmark before exporting projected gradients");
    }
    const TrainingValidationStats& validation = validationService_.stats();
    if (fixedBenchmarkStats_.checkpointDiagnosticsRequested &&
        (fixedBenchmarkStats_.checkpointSkippedSteps != 0u ||
         fixedBenchmarkStats_.checkpointFallbackCount != 0u ||
         fixedBenchmarkStats_.checkpointErrorFlags != 0u))
        throw std::runtime_error("Checkpoint capture had skipped/fallback/failed steps; export a clean benchmark run");
    if (!validation.valid || validation.parentTileItemCount == 0u ||
        fixedBenchmarkStats_.gaussianCount != trainableGaussianCount_) {
        throw std::runtime_error("Completed benchmark has no valid gradient snapshot");
    }

    const auto gradients = buffers_.downloadProjectedGrads(trainableGaussianCount_);
    const TrainingExtent extent = buffers_.extent();
    const auto pixelGradients = buffers_.downloadPixelGrads(extent.width * extent.height);
    std::vector<uint8_t> childMembership(trainableGaussianCount_, 0u);
    if (fixedBenchmarkStats_.tileMode == TrainingTileMode::AdaptiveSpatial) {
        childMembership = buffers_.downloadAdaptiveChildMembership(
            trainableGaussianCount_, validation.parentTileItemCount,
            validation.adaptiveChildItemCount);
    }

    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Could not open projected gradient output: " + path.string());
    }

    static_assert(std::endian::native == std::endian::little);
    static_assert(sizeof(ProjectedGaussianGrad) == 12u * sizeof(float));
    constexpr char magic[8] = {'V', '3', 'D', 'G', 'G', 'R', 'D', '1'};
    output.write(magic, sizeof(magic));
    const auto writeU32 = [&output](uint32_t value) {
        char bytes[4] = {static_cast<char>(value & 0xffu),
                         static_cast<char>((value >> 8u) & 0xffu),
                         static_cast<char>((value >> 16u) & 0xffu),
                         static_cast<char>((value >> 24u) & 0xffu)};
        output.write(bytes, sizeof(bytes));
    };
    writeU32(9u); // V9 appends the actual backward pixel-culling method.
    writeU32(trainableGaussianCount_);
    writeU32(fixedBenchmarkStats_.trainingIteration);
    writeU32(fixedBenchmarkStats_.frameIndex);
    writeU32(static_cast<uint32_t>(fixedBenchmarkStats_.tileMode));
    writeU32(static_cast<uint32_t>(fixedBenchmarkStats_.pixelMode));
    writeU32(static_cast<uint32_t>(fixedBenchmarkStats_.compositeMode));
    writeU32(extent.width);
    writeU32(extent.height);
    writeU32(validation.parentTileItemCount);
    writeU32(validation.adaptiveChildItemCount);
    writeU32(validation.adaptiveSplitParentCount);
    writeU32(fixedBenchmarkStats_.adaptiveTensorChildBatch);
    writeU32(fixedBenchmarkStats_.tensorSubgroupReduction ? 1u : 0u);
    writeU32(fixedBenchmarkStats_.tensorGlobalPixelGrad ? 1u : 0u);
    writeU32(fixedBenchmarkStats_.tensorRgbLatencyHiding ? 1u : 0u);
    writeU32(static_cast<uint32_t>(fixedBenchmarkStats_.ssimMode));
    writeU32(fixedBenchmarkStats_.checkpointDiagnosticSteps != 0u ? 32u : 0u);
    writeU32(fixedBenchmarkStats_.checkpointDiagnosticSteps != 0u ? 2u : 0u);
    writeU32(fixedBenchmarkStats_.checkpointTaskCount);
    writeU32(fixedBenchmarkStats_.checkpointDiagnosticSteps != 0u ? 1u : 0u);
    writeU32(fixedBenchmarkStats_.checkpointFallbackCount);
    writeU32(fixedBenchmarkStats_.pixelCullMethod);
    writeU32(std::bit_cast<uint32_t>(validation.meanLoss));
    writeU32(std::bit_cast<uint32_t>(validation.meanRenderedAlpha));
    output.write(reinterpret_cast<const char*>(gradients.data()),
                 static_cast<std::streamsize>(gradients.size() * sizeof(ProjectedGaussianGrad)));
    output.write(reinterpret_cast<const char*>(childMembership.data()),
                 static_cast<std::streamsize>(childMembership.size()));
    output.write(reinterpret_cast<const char*>(pixelGradients.data()),
                 static_cast<std::streamsize>(pixelGradients.size() * sizeof(PixelGrad)));
    if (!output) {
        throw std::runtime_error("Failed to write projected gradients: " + path.string());
    }
}

void GaussianTraining::exportScanlineDiagnosticInputs(const std::filesystem::path& path) {
    const auto& b = fixedBenchmarkStats_;
    const auto& v = validationService_.stats();
    if (!b.complete || b.active || !v.valid || b.gaussianCount != trainableGaussianCount_ ||
        b.trainingIteration != trainingIteration_ || b.tileMode != TrainingTileMode::Fixed16 ||
        b.compositeMode != TrainingForwardCompositeMode::WorkgroupShared ||
        b.pixelMode != TrainingPixelTo2DGSMode::VkSplatTensor || !b.tensorSubgroupReduction ||
        !b.tensorGlobalPixelGrad || b.tensorRgbLatencyHiding || b.pixelCullMethod ||
        b.checkpointDiagnosticsRequested)
        throw std::runtime_error("S3-2 export requires a completed original Fixed16 Tensor Subgroup/Global Baseline benchmark, without EarlyCull/checkpoints");
    TrainingScanlineHeader header;
    header.gaussianCount = b.gaussianCount;
    header.iteration = b.trainingIteration;
    header.frame = b.frameIndex;
    header.width = buffers_.extent().width;
    header.height = buffers_.extent().height;
    header.itemCount = v.parentTileItemCount;
    header.tileCount = ((header.width + 15u) / 16u) * ((header.height + 15u) / 16u);
    header.ssimMode = static_cast<uint32_t>(b.ssimMode);
    const auto properties = physicalDevice_.getProperties();
    header.vendorID = properties.vendorID;
    header.deviceID = properties.deviceID;
    header.driverVersion = properties.driverVersion;
    stepExecutor_.synchronizeDiagnosticDownloads(transferQueue_);
    writeTrainingScanlineCapture(path, buffers_.downloadScanlineCapture(header));
    LOG_INFO("Exported S3-2 frozen inputs: {} bytes to {} (no training mutation)",
             trainingScanlineCaptureBytes(header), path.string());
}

void GaussianTraining::exportAdaptiveTileDiagnostics(const std::filesystem::path& path) {
    const auto& benchmark = fixedBenchmarkStats_;
    const auto& validation = validationService_.stats();
    if (!benchmark.complete || benchmark.active ||
        benchmark.tileMode != TrainingTileMode::AdaptiveSpatial || !validation.valid ||
        validation.parentTileItemCount == 0u || benchmark.gaussianCount != trainableGaussianCount_) {
        throw std::runtime_error("Complete an Adaptive Spatial Fixed Benchmark before exporting tile diagnostics");
    }
    const auto rows = buffers_.downloadAdaptiveTileDiagnostics(validation.parentTileItemCount);
    const bool hasClassifiedParents = std::any_of(rows.begin(), rows.end(), [&benchmark](const auto& row) {
        return row.parentItems >= benchmark.adaptiveSplitThreshold;
    });
    const auto rawMasks = hasClassifiedParents
                             ? buffers_.downloadAdaptiveChildMasks(validation.parentTileItemCount)
                             : std::vector<uint32_t>{};
    std::vector<uint8_t> packedMasks(validation.parentTileItemCount, 0u);
    size_t offset = 0u;
    for (const auto& row : rows) {
        if (offset + row.parentItems > packedMasks.size()) {
            throw std::runtime_error("Adaptive parent ranges exceed the mask export capacity");
        }
        if (row.parentItems >= benchmark.adaptiveSplitThreshold) {
            std::array<uint32_t, 4> counted{};
            for (size_t item = offset; item < offset + row.parentItems; ++item) {
                const uint32_t mask = rawMasks[item];
                if (mask > 15u) throw std::runtime_error("Adaptive diagnostics found an invalid child mask");
                packedMasks[item] = static_cast<uint8_t>(mask);
                for (uint32_t child = 0u; child < 4u; ++child) {
                    counted[child] += (mask >> child) & 1u;
                }
            }
            if (counted != row.childItems) {
                throw std::runtime_error("Adaptive mask counts do not match the classified parent lists");
            }
        }
        offset += row.parentItems;
    }
    const auto extent = buffers_.extent();
    const auto layout = makeTrainingTileLayout(extent, kTrainingTileSize16);
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Could not open adaptive tile output: " + path.string());
    output << "frame,iteration,gaussians,width,height,threshold,pixel_mode,parent_index,parent_x,parent_y,"
              "parent_items,child0,child1,child2,child3,selected,max0,max1,max2,max3,child_batch\n";
    for (size_t index = 0u; index < rows.size(); ++index) {
        const auto& row = rows[index];
        output << benchmark.frameIndex << ',' << benchmark.trainingIteration << ','
               << benchmark.gaussianCount << ',' << extent.width << ',' << extent.height << ','
               << benchmark.adaptiveSplitThreshold << ',' << static_cast<uint32_t>(benchmark.pixelMode)
               << ',' << index << ',' << index % layout.tileCountX << ',' << index / layout.tileCountX
               << ',' << row.parentItems;
        for (uint32_t count : row.childItems) output << ',' << count;
        output << ',' << (row.selected ? 1u : 0u);
        for (uint32_t count : row.maxProcessed) output << ',' << count;
        output << ',' << benchmark.adaptiveTensorChildBatch << '\n';
    }
    if (!output) throw std::runtime_error("Failed to write adaptive tile diagnostics: " + path.string());
    std::filesystem::path maskPath = path;
    maskPath.replace_extension(".masks.bin");
    std::ofstream maskOutput(maskPath, std::ios::binary | std::ios::trunc);
    if (!maskOutput) throw std::runtime_error("Could not open child mask output: " + maskPath.string());
    static_assert(std::endian::native == std::endian::little);
    maskOutput.write("V3DMSK01", 8);
    const std::array<uint32_t, 10> header = {
        1u, benchmark.gaussianCount, benchmark.trainingIteration, benchmark.frameIndex,
        extent.width, extent.height, benchmark.adaptiveSplitThreshold,
        static_cast<uint32_t>(benchmark.pixelMode), validation.parentTileItemCount,
        static_cast<uint32_t>(rows.size())};
    static_assert(sizeof(header) == 40u);
    maskOutput.write(reinterpret_cast<const char*>(header.data()), sizeof(header));
    maskOutput.write(reinterpret_cast<const char*>(packedMasks.data()),
                     static_cast<std::streamsize>(packedMasks.size()));
    if (!maskOutput) throw std::runtime_error("Failed to write child masks: " + maskPath.string());
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

    const bool checkpointRequested = config.checkpointDiagnostics || config.checkpointWarp;
    const auto* backward = dynamic_cast<const GaussianBackwardRenderer*>(backward_.get());
    if (config.tensorEarlyCull &&
        (checkpointRequested || !backward || !backward->tensorEarlyCullSupported() ||
         tileMode_ != TrainingTileMode::Fixed16 ||
         forwardCompositeMode_ != TrainingForwardCompositeMode::WorkgroupShared ||
         activePixelTo2DGSMode() != TrainingPixelTo2DGSMode::VkSplatTensor ||
         !tensorGlobalPixelGradActive() || tensorRgbLatencyHidingActive()))
        throw std::runtime_error("EarlyCull requires Fixed16 / Workgroup Shared / Tensor Subgroup Global / RGB Baseline, without checkpoint diagnostics");
    if (checkpointRequested &&
        (tileMode_ != TrainingTileMode::Fixed16 ||
         forwardCompositeMode_ != TrainingForwardCompositeMode::WorkgroupShared ||
         activePixelTo2DGSMode() != TrainingPixelTo2DGSMode::VkSplatTensor ||
         !tensorGlobalPixelGradActive() || tensorRgbLatencyHidingActive()))
        throw std::runtime_error("Checkpoint P1 diagnostics require Fixed16 / Workgroup Shared / Tensor Subgroup Global / RGB Baseline");
    checkpointDiagnosticsRequested_ = checkpointRequested;
    checkpointWarpRequested_ = config.checkpointWarp;
    benchmarkTensorEarlyCullRequested_ = config.tensorEarlyCull;

    validationService_.collect(
        buffers_.extent(), buffers_.gaussianCapacity());
    validationService_.resetStats();
    frameScheduler_.setCurrentFrame(config.frameIndex);
    fixedBenchmarkStats_ = {};
    fixedBenchmarkStats_.checkpointDiagnosticsRequested = checkpointRequested;
    fixedBenchmarkStats_.checkpointWarpRequested = config.checkpointWarp;
    fixedBenchmarkStats_.pixelCullMethod = config.tensorEarlyCull ? 1u : 0u;
    if (checkpointDiagnosticsRequested_)
        LOG_INFO("Checkpoint {} enabled for Fixed Benchmark", config.checkpointWarp ? "P2 warp backward" : "P1 state diagnostic");
    fixedBenchmarkStats_.active = true;
    fixedBenchmarkStats_.frameIndex = config.frameIndex;
    fixedBenchmarkStats_.gaussianCount = trainableGaussianCount_;
    fixedBenchmarkStats_.trainingIteration = trainingIteration_;
    fixedBenchmarkStats_.tileMode = tileMode_;
    fixedBenchmarkStats_.pixelMode = activePixelTo2DGSMode();
    fixedBenchmarkStats_.compositeMode = forwardCompositeMode_;
    fixedBenchmarkStats_.adaptiveSplitThreshold = adaptiveTileSplitThreshold_;
    fixedBenchmarkStats_.adaptiveTensorChildBatch = adaptiveTensorChildBatch_;
    fixedBenchmarkStats_.tensorSubgroupReduction = tensorSubgroupReductionActive();
    fixedBenchmarkStats_.tensorGlobalPixelGrad = tensorGlobalPixelGradActive();
    fixedBenchmarkStats_.tensorRgbLatencyHiding = tensorRgbLatencyHidingActive();
    fixedBenchmarkStats_.ssimMode = activeSsimMode();
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
    pushConstants.densificationGradientMode =
        static_cast<uint32_t>(densificationConfig_.gradientMode);
    pushConstants.tileSize = tileMode_ == TrainingTileMode::AdaptiveSpatial
                                 ? 32u
                                 : trainingTileSize(tileMode_);
    if (tileMode_ == TrainingTileMode::AdaptiveSpatial) {
        pushConstants.tileItemCount = adaptiveTileSplitThreshold_;
    }
    return pushConstants;
}

void GaussianTraining::setTileMode(TrainingTileMode mode, uint32_t adaptiveSplitThreshold) {
    tileMode_ = mode;
    adaptiveTileSplitThreshold_ = std::max(adaptiveSplitThreshold, 1u);
    if (auto* gaussianBackward = dynamic_cast<GaussianBackwardRenderer*>(backward_.get())) {
        gaussianBackward->setAdaptiveSpatialTiles(mode == TrainingTileMode::AdaptiveSpatial);
        if (mode == TrainingTileMode::AdaptiveSpatial &&
            !gaussianBackward->adaptiveSpatialModeSupported()) {
            LOG_WARN("Adaptive Spatial requires a supported VkSplat Tensor or Per-Splat "
                     "backward mode; falling back to Fixed16");
            tileMode_ = TrainingTileMode::Fixed16;
            gaussianBackward->setAdaptiveSpatialTiles(false);
        }
    }
}

void GaussianTraining::setSsimMode(TrainingSsimMode mode) {
    // Configuration is stopped-only, or applied before the worker starts.
    // Allocate before changing the actual path so allocation failure leaves it intact.
    buffers_.configureSsimScratch(mode);
    ssimMode_ = mode;
    if (auto* renderer = dynamic_cast<GaussianBackwardRenderer*>(backward_.get())) {
        renderer->setSsimMode(mode);
    }
}

TrainingSsimMode GaussianTraining::activeSsimMode() const {
    const auto* renderer = dynamic_cast<const GaussianBackwardRenderer*>(backward_.get());
    return renderer ? renderer->activeSsimMode() : TrainingSsimMode::TwoDimensional;
}

void GaussianTraining::setFasterSortEnabled(bool enabled) {
    fasterSortEnabled_ = enabled;
    if (auto* renderer = dynamic_cast<GaussianForwardRenderer*>(forward_.get()))
        renderer->setFasterSortEnabled(enabled);
}

void GaussianTraining::setTensorRgbLatencyHiding(bool enabled) {
    tensorRgbLatencyHiding_ = enabled;
    if (auto* renderer = dynamic_cast<GaussianBackwardRenderer*>(backward_.get())) {
        renderer->setTensorRgbLatencyHiding(enabled);
    }
}

TrainingPixelCullStatus GaussianTraining::tensorEarlyCullStatus() const {
    const auto* renderer = dynamic_cast<const GaussianBackwardRenderer*>(backward_.get());
    return resolveTrainingPixelCull(tensorEarlyCull_, {
        .rendererReady = rendererInitialized_ && renderer != nullptr,
        .pipelineAvailable = renderer && renderer->tensorEarlyCullSupported(),
        .tileMode = tileMode_,
        .forwardMode = forwardCompositeMode_,
        .pixelMode = activePixelTo2DGSMode(),
        .globalPixelGrad = tensorGlobalPixelGradActive(),
        .rgbLatencyHiding = tensorRgbLatencyHidingActive(),
    });
}

void GaussianTraining::resetTensorEarlyCullRunStats() {
    pixelCullRunStats_ = {};
    LOG_INFO("Training EarlyCull requested={}, selection={}", tensorEarlyCull_,
             trainingPixelCullStatusName(tensorEarlyCullStatus()));
}

bool GaussianTraining::tensorRgbLatencyHidingActive() const {
    const auto* renderer = dynamic_cast<const GaussianBackwardRenderer*>(backward_.get());
    return renderer && renderer->tensorRgbLatencyHidingActive(activeTileSize());
}

void GaussianTraining::setTensorGlobalPixelGrad(bool enabled) {
    tensorGlobalPixelGrad_ = enabled;
    if (auto* renderer = dynamic_cast<GaussianBackwardRenderer*>(backward_.get())) {
        renderer->setTensorGlobalPixelGrad(enabled);
    }
}

bool GaussianTraining::tensorGlobalPixelGradActive() const {
    const auto* renderer = dynamic_cast<const GaussianBackwardRenderer*>(backward_.get());
    return renderer && renderer->tensorGlobalPixelGradActive(activeTileSize());
}

void GaussianTraining::setTensorSubgroupReduction(bool enabled) {
    tensorSubgroupReduction_ = enabled;
    if (auto* renderer = dynamic_cast<GaussianBackwardRenderer*>(backward_.get())) {
        renderer->setTensorSubgroupReduction(enabled);
    }
}

bool GaussianTraining::tensorSubgroupReductionSupported() const {
    const auto* renderer = dynamic_cast<const GaussianBackwardRenderer*>(backward_.get());
    return renderer && renderer->tensorSubgroupReductionSupported();
}

bool GaussianTraining::tensorSubgroupReductionActive() const {
    const auto* renderer = dynamic_cast<const GaussianBackwardRenderer*>(backward_.get());
    return renderer && renderer->tensorSubgroupReductionActive(activeTileSize());
}

void GaussianTraining::setAdaptiveTensorChildBatch(uint32_t batch) {
    adaptiveTensorChildBatch_ = normalizeAdaptiveTensorChildBatch(batch);
    if (auto* gaussianBackward = dynamic_cast<GaussianBackwardRenderer*>(backward_.get())) {
        gaussianBackward->setAdaptiveTensorChildBatch(adaptiveTensorChildBatch_);
    }
}

void GaussianTraining::setPixelTo2DGSMode(TrainingPixelTo2DGSMode mode) {
    pixelTo2DGSMode_ = mode;
    if (auto* gaussianBackward = dynamic_cast<GaussianBackwardRenderer*>(backward_.get())) {
        gaussianBackward->setPixelTo2DGSMode(mode);
        gaussianBackward->setAdaptiveSpatialTiles(
            tileMode_ == TrainingTileMode::AdaptiveSpatial);
        if (tileMode_ == TrainingTileMode::AdaptiveSpatial &&
            !gaussianBackward->adaptiveSpatialModeSupported()) {
            LOG_WARN("Selected Pixel-to-2DGS mode is not supported with Adaptive Spatial; "
                     "falling back to Fixed16");
            tileMode_ = TrainingTileMode::Fixed16;
            gaussianBackward->setAdaptiveSpatialTiles(false);
        }
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

bool GaussianTraining::adaptiveSpatialBackwardSupported() const {
    const auto* gaussianBackward = dynamic_cast<const GaussianBackwardRenderer*>(backward_.get());
    return gaussianBackward && gaussianBackward->adaptiveSpatialModeSupported();
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
        device, physicalDevice, computeQueueFamilyIndex, vulkanValidationLayersEnabled_);
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
