#pragma once

#include "training/core/backward/backward_command_utils.hpp"
#include "training/core/backward/backward_pass_context.hpp"
#include "vulkan/compute_pipeline.hpp"

namespace vulkan3DGS {

class BackwardValidationPass {
public:
    void initialize(vk::Device device);
    void cleanup();

    void record(const BackwardPassContext& context);

private:
    void recordGaussianValidation(const BackwardPassContext& context);
    void recordFinalize(const BackwardPassContext& context);

    vk::Device device_ = nullptr;
    ComputePipeline gaussianValidationPipeline_;
    ComputePipeline finalizePipeline_;
    BackwardDescriptorPool descriptorPool_;
    vk::DescriptorSet gaussianValidationDescriptorSet_ = nullptr;
    vk::DescriptorSet finalizeDescriptorSet_ = nullptr;
};

} // namespace vulkan3DGS
