#pragma once

#include "training/core/backward/backward_command_utils.hpp"
#include "training/core/backward/backward_pass_context.hpp"
#include "vulkan/compute_pipeline.hpp"

namespace vulkan3DGS {

class ProjectionOptimizerPass {
public:
    void initialize(vk::Device device);
    void cleanup();

    [[nodiscard]] bool fusedEnabled() const noexcept {
        return fusedEnabled_;
    }
    void recordProjection(const BackwardPassContext& context);
    void recordProjectionAndOptimize(const BackwardPassContext& context);
    void recordOptimizer(const BackwardPassContext& context);

private:
    vk::Device device_ = nullptr;
    ComputePipeline projectionPipeline_;
    ComputePipeline fusedPipeline_;
    ComputePipeline optimizerPipeline_;
    BackwardDescriptorPool descriptorPool_;
    vk::DescriptorSet projectionDescriptorSet_ = nullptr;
    vk::DescriptorSet fusedDescriptorSet_ = nullptr;
    vk::DescriptorSet optimizerDescriptorSet_ = nullptr;
    bool fusedEnabled_ = true;
};

} // namespace vulkan3DGS
