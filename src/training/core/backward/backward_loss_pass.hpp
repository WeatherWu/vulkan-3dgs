#pragma once

#include "training/core/backward/backward_command_utils.hpp"
#include "training/core/backward/backward_pass_context.hpp"
#include "vulkan/compute_pipeline.hpp"

namespace vulkan3DGS {

class BackwardLossPass {
public:
    void initialize(vk::Device device);
    void cleanup();
    void setSsimMode(TrainingSsimMode mode) {
        ssimMode_ = mode;
    }
    [[nodiscard]] TrainingSsimMode activeSsimMode() const {
        return ssimMode_;
    }

    void recordLoss(const BackwardPassContext& context);
    void recordClear(const BackwardPassContext& context, bool fusedProjectionOptimizerEnabled);
    void recordLossToPixel(const BackwardPassContext& context);

private:
    void recordSsimHorizontal(const BackwardPassContext& context);
    void recordScratchReadComplete(const BackwardPassContext& context);
    vk::Device device_ = nullptr;
    ComputePipeline clearPipeline_;
    ComputePipeline lossPipeline_;
    ComputePipeline lossToPixelPipeline_;
    ComputePipeline ssimCoefficientsHorizontalPipeline_;
    ComputePipeline separableLossToPixelPipeline_;
    BackwardDescriptorPool descriptorPool_;
    vk::DescriptorSet clearDescriptorSet_ = nullptr;
    vk::DescriptorSet lossDescriptorSet_ = nullptr;
    vk::DescriptorSet lossToPixelDescriptorSet_ = nullptr;
    vk::DescriptorSet ssimCoefficientsHorizontalSet_ = nullptr;
    vk::DescriptorSet separableLossToPixelSet_ = nullptr;
    TrainingSsimMode ssimMode_ = TrainingSsimMode::TwoDimensional;
    bool computeClear_ = false;
};

} // namespace vulkan3DGS
