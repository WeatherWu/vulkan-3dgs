#pragma once

#include "training/core/backward/backward_command_utils.hpp"
#include "training/core/backward/backward_pass_context.hpp"
#include "vulkan/compute_pipeline.hpp"

#include <array>

namespace vulkan3DGS {

class VkSplatLossPass {
public:
    VkSplatLossPass(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                    uint32_t family);
    void recordForward(const BackwardPassContext& context);
    void recordBackward(const BackwardPassContext& context);
    vk::DescriptorBufferInfo stateInfo() const {
        return state_.getDescriptorInfo();
    }

private:
    void record(const BackwardPassContext& context, uint32_t stage);
    vk::Device device_;
    vk::PhysicalDevice physical_;
    vk::Queue queue_;
    uint32_t family_;
    Buffer state_;
    std::array<ComputePipeline, 3> pipelines_;
    BackwardDescriptorPool descriptors_;
    std::vector<vk::DescriptorSet> sets_;
};

} // namespace vulkan3DGS
