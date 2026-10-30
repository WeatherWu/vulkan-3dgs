#pragma once

#include "render/renderer.hpp"
#include "training/core/backward/backward_loss_pass.hpp"
#include "training/core/backward/backward_validation_pass.hpp"
#include "training/core/backward/pixel_to_2dgs_dispatcher.hpp"
#include "training/core/backward/projection_optimizer_pass.hpp"
#include "training/core/training_buffers.hpp"
#include "training/core/training_types.hpp"
#include "training/core/training_vksplat_backend.hpp"

#include <vulkan/vulkan.hpp>

namespace vulkan3DGS {

class GaussianBackwardRenderer : public BackwardRenderer {
public:
    void initialize(vk::Device device, vk::PhysicalDevice physicalDevice, vk::Queue computeQueue,
                    uint32_t computeQueueFamilyIndex, uint32_t gaussianCount,
                    TrainingExtent extent) override;
    void cleanup() override;

    void setTrainingBuffers(const TrainingBuffers& trainingBuffers, vk::CommandBuffer commandBuffer,
                            TrainingPushConstants pushConstants);
    void setProfilingQueryPool(vk::QueryPool queryPool);
    void setVkSplatLoss(bool enabled) {
        lossPass_.setVkSplat(enabled);
    }
    bool vkSplatLossActive() const {
        return lossPass_.vkSplatActive();
    }
    void setSsimMode(TrainingSsimMode mode) { lossPass_.setSsimMode(mode); }
    [[nodiscard]] TrainingSsimMode activeSsimMode() const { return lossPass_.activeSsimMode(); }
    void setPixelTo2DGSMode(TrainingPixelTo2DGSMode mode) {
        pixelDispatcher_.setRequestedMode(mode);
    }
    void setAdaptiveSpatialTiles(bool enabled) {
        pixelDispatcher_.setAdaptiveSpatialTiles(enabled);
    }
    void setAdaptiveTensorChildBatch(uint32_t batch) {
        pixelDispatcher_.setAdaptiveTensorChildBatch(batch);
    }
    void setTensorSubgroupReduction(bool enabled) {
        pixelDispatcher_.setTensorSubgroupReduction(enabled);
    }
    void setTensorGlobalPixelGrad(bool enabled) {
        pixelDispatcher_.setTensorGlobalPixelGrad(enabled);
    }
    void setTensorRgbLatencyHiding(bool enabled) {
        pixelDispatcher_.setTensorRgbLatencyHiding(enabled);
    }
    [[nodiscard]] bool tensorRgbLatencyHidingActive(uint32_t tileSize) const {
        return pixelDispatcher_.tensorRgbLatencyHidingActive(tileSize);
    }
    [[nodiscard]] bool tensorGlobalPixelGradActive(uint32_t tileSize) const {
        return pixelDispatcher_.tensorGlobalPixelGradActive(tileSize);
    }
    [[nodiscard]] bool tensorSubgroupReductionSupported() const {
        return supportsTensorSubgroupReduction(pixelDispatcher_.capabilities());
    }
    [[nodiscard]] bool tensorSubgroupReductionActive(uint32_t tileSize) const {
        return pixelDispatcher_.tensorSubgroupReductionActive(tileSize);
    }
    [[nodiscard]] bool subgroupPixelTo2DGSSupported() const {
        return pixelDispatcher_.capabilities().subgroup;
    }
    [[nodiscard]] bool tileGaussianPixelTo2DGSSupported() const {
        return pixelDispatcher_.capabilities().tileGaussian;
    }
    [[nodiscard]] bool vkSplatPerSplatSupported() const {
        return pixelDispatcher_.capabilities().vkSplatPerSplat;
    }
    [[nodiscard]] bool vkSplatTensorSupported() const {
        return pixelDispatcher_.capabilities().vkSplatTensor;
    }
    [[nodiscard]] TrainingPixelTo2DGSMode activePixelTo2DGSMode() const {
        return pixelDispatcher_.activeMode();
    }
    [[nodiscard]] bool adaptiveSpatialModeSupported() const {
        return pixelDispatcher_.adaptiveSpatialModeSupported();
    }
    void setVkSplatBackend(TrainingVkSplatBackend* backend) {
        vkSplat_ = backend;
    }
    void backward() override;
    void prepareCheckpointDiagnostics(TrainingBuffers& buffers, uint32_t itemCount, bool warp = false) {
        pixelDispatcher_.prepareCheckpointDiagnostics(buffers, itemCount, warp);
    }
    void recordCheckpointDiagnostics(bool resetRun) {
        pixelDispatcher_.recordCheckpointDiagnostics(context(), resetRun);
    }
    void setCheckpointWarp(bool enabled) { checkpointWarp_ = enabled; }
    void setTensorEarlyCull(bool enabled) { tensorEarlyCull_ = enabled; }
    void setOptimizerCapture(TrainingOptimizerCaptureSession* capture) { optimizerCapture_ = capture; }
    void setDirectIntrinsics(bool enabled) {
        directIntrinsics_ = enabled;
        if (!enabled) projectionOptimizer_.releaseDirectResources();
    }
    bool hasDirectResources() const {
        return projectionOptimizer_.hasDirectResources();
    }
    bool fusedOptimizerEnabled() const { return projectionOptimizer_.fusedEnabled(); }
    [[nodiscard]] bool tensorEarlyCullSupported() const {
        return pixelDispatcher_.tensorEarlyCullSupported();
    }
    void gradientDescent() override;

    [[nodiscard]] bool isInitialized() const override {
        return initialized_;
    }
    [[nodiscard]] uint32_t gaussianCount() const {
        return gaussianCount_;
    }
    [[nodiscard]] TrainingExtent extent() const {
        return extent_;
    }

private:
    [[nodiscard]] bool readyForRecording(const char* operation) const;
    [[nodiscard]] BackwardPassContext context() const;
    void writeProfilingTimestamp(TrainingGpuProfileStage stage, bool end) const;

    vk::Device device_ = nullptr;
    uint32_t gaussianCount_ = 0;
    TrainingExtent extent_{};
    BackwardLossPass lossPass_;
    PixelTo2DGSDispatcher pixelDispatcher_;
    ProjectionOptimizerPass projectionOptimizer_;
    BackwardValidationPass validationPass_;
    const TrainingBuffers* trainingBuffers_ = nullptr;
    vk::CommandBuffer commandBuffer_ = nullptr;
    vk::QueryPool profilingQueryPool_ = nullptr;
    TrainingPushConstants pushConstants_{};
    bool initialized_ = false;
    bool directIntrinsics_ = false;
    TrainingVkSplatBackend* vkSplat_ = nullptr;
    bool checkpointWarp_ = false;
    bool tensorEarlyCull_ = false;
    TrainingOptimizerCaptureSession* optimizerCapture_ = nullptr;
};

} // namespace vulkan3DGS
