#include "training/core/gaussian_backward_renderer.hpp"

#include "training/core/backward/backward_command_utils.hpp"
#include "utils/logger.hpp"

namespace vulkan3DGS {

void GaussianBackwardRenderer::initialize(vk::Device device, vk::PhysicalDevice physicalDevice,
                                          vk::Queue computeQueue, uint32_t computeQueueFamilyIndex,
                                          uint32_t gaussianCount, TrainingExtent extent) {
    (void)computeQueue;
    (void)computeQueueFamilyIndex;
    cleanup();
    device_ = device;
    gaussianCount_ = gaussianCount;
    extent_ = extent;
    lossPass_.initialize(device_);
    pixelDispatcher_.initialize(device_, physicalDevice);
    projectionOptimizer_.initialize(device_);
    validationPass_.initialize(device_);
    initialized_ = true;
    LOG_INFO("GaussianBackwardRenderer initialized ({} gaussians, {}x{})", gaussianCount_,
             extent_.width, extent_.height);
}

void GaussianBackwardRenderer::cleanup() {
    validationPass_.cleanup();
    projectionOptimizer_.cleanup();
    pixelDispatcher_.cleanup();
    lossPass_.cleanup();
    device_ = nullptr;
    gaussianCount_ = 0u;
    extent_ = {};
    trainingBuffers_ = nullptr;
    commandBuffer_ = nullptr;
    profilingQueryPool_ = nullptr;
    pushConstants_ = {};
    initialized_ = false;
}

void GaussianBackwardRenderer::setTrainingBuffers(const TrainingBuffers& trainingBuffers,
                                                  vk::CommandBuffer commandBuffer,
                                                  TrainingPushConstants pushConstants) {
    trainingBuffers_ = &trainingBuffers;
    commandBuffer_ = commandBuffer;
    pushConstants_ = pushConstants;
}

void GaussianBackwardRenderer::setProfilingQueryPool(vk::QueryPool queryPool) {
    profilingQueryPool_ = queryPool;
}

void GaussianBackwardRenderer::backward() {
    if (!readyForRecording("backward")) {
        return;
    }
    const BackwardPassContext passContext = context();
    writeProfilingTimestamp(TrainingGpuProfileStage::Loss, false);
    lossPass_.recordLoss(passContext);
    writeProfilingTimestamp(TrainingGpuProfileStage::Loss, true);

    writeProfilingTimestamp(TrainingGpuProfileStage::BackwardClear, false);
    lossPass_.recordClear(passContext, projectionOptimizer_.fusedEnabled());
    writeProfilingTimestamp(TrainingGpuProfileStage::BackwardClear, true);

    writeProfilingTimestamp(TrainingGpuProfileStage::LossToPixel, false);
    lossPass_.recordLossToPixel(passContext);
    writeProfilingTimestamp(TrainingGpuProfileStage::LossToPixel, true);

    writeProfilingTimestamp(TrainingGpuProfileStage::PixelTo2DGS, false);
    pixelDispatcher_.record(passContext);
    writeProfilingTimestamp(TrainingGpuProfileStage::PixelTo2DGS, true);

    if (pushConstants_.optimizerEnabled != 0u && projectionOptimizer_.fusedEnabled()) {
        writeProfilingTimestamp(TrainingGpuProfileStage::FusedProjectionOptimizer, false);
        projectionOptimizer_.recordProjectionAndOptimize(passContext);
        writeProfilingTimestamp(TrainingGpuProfileStage::FusedProjectionOptimizer, true);
    } else {
        writeProfilingTimestamp(TrainingGpuProfileStage::TwoDGSTo3DGS, false);
        projectionOptimizer_.recordProjection(passContext);
        writeProfilingTimestamp(TrainingGpuProfileStage::TwoDGSTo3DGS, true);
    }
}

void GaussianBackwardRenderer::gradientDescent() {
    if (!readyForRecording("gradientDescent")) {
        return;
    }
    const BackwardPassContext passContext = context();
    if (pushConstants_.optimizerEnabled != 0u && !projectionOptimizer_.fusedEnabled()) {
        writeProfilingTimestamp(TrainingGpuProfileStage::Optimizer, false);
        projectionOptimizer_.recordOptimizer(passContext);
        writeProfilingTimestamp(TrainingGpuProfileStage::Optimizer, true);
    }
    if (pushConstants_.validationEnabled != 0u) {
        writeProfilingTimestamp(TrainingGpuProfileStage::Validation, false);
        validationPass_.record(passContext);
        writeProfilingTimestamp(TrainingGpuProfileStage::Validation, true);
    }
}

bool GaussianBackwardRenderer::readyForRecording(const char* operation) const {
    if (!initialized_) {
        LOG_WARN("GaussianBackwardRenderer::{} called before initialization", operation);
        return false;
    }
    if (!trainingBuffers_) {
        LOG_WARN("GaussianBackwardRenderer::{} called without training buffers", operation);
        return false;
    }
    if (!commandBuffer_) {
        LOG_WARN("GaussianBackwardRenderer::{} called without a command buffer", operation);
        return false;
    }
    return true;
}

BackwardPassContext GaussianBackwardRenderer::context() const {
    return {*trainingBuffers_, commandBuffer_, pushConstants_, profilingQueryPool_};
}

void GaussianBackwardRenderer::writeProfilingTimestamp(TrainingGpuProfileStage stage,
                                                       bool end) const {
    writeBackwardProfilingTimestamp(commandBuffer_, profilingQueryPool_, stage, end);
}

} // namespace vulkan3DGS
