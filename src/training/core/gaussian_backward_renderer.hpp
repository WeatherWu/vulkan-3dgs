#pragma once

#include "render/renderer.hpp"
#include "training/core/backward/backward_loss_pass.hpp"
#include "training/core/backward/backward_validation_pass.hpp"
#include "training/core/backward/pixel_to_2dgs_dispatcher.hpp"
#include "training/core/backward/projection_optimizer_pass.hpp"
#include "training/core/training_buffers.hpp"
#include "training/core/training_types.hpp"

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
    void setPixelTo2DGSMode(TrainingPixelTo2DGSMode mode) {
        pixelDispatcher_.setRequestedMode(mode);
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
    void backward() override;
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
};

} // namespace vulkan3DGS
