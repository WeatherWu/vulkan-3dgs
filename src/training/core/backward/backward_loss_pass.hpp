#pragma once

#include "training/core/backward/backward_command_utils.hpp"
#include "training/core/backward/backward_pass_context.hpp"
#include "vulkan/compute_pipeline.hpp"

namespace vulkan3DGS {

class BackwardLossPass {
public:
    void initialize(vk::Device device);
    void cleanup();

    void recordLoss(const BackwardPassContext& context);
    void recordClear(const BackwardPassContext& context, bool fusedProjectionOptimizerEnabled);
    void recordLossToPixel(const BackwardPassContext& context);

private:
    vk::Device device_ = nullptr;
    ComputePipeline clearPipeline_;
    ComputePipeline lossPipeline_;
    ComputePipeline lossToPixelPipeline_;
    BackwardDescriptorPool descriptorPool_;
    vk::DescriptorSet clearDescriptorSet_ = nullptr;
    vk::DescriptorSet lossDescriptorSet_ = nullptr;
    vk::DescriptorSet lossToPixelDescriptorSet_ = nullptr;
    bool computeClear_ = false;
};

} // namespace vulkan3DGS
