#pragma once

#include "training/core/backward/backward_command_utils.hpp"
#include "training/core/backward/backward_pass_context.hpp"
#include "vulkan/compute_pipeline.hpp"
#include <filesystem>

namespace vulkan3DGS {
// Created only for an actual direct-intrinsics step, after the previous fence.
class TrainingDirectProjection {
public:
    TrainingDirectProjection(vk::Device device, bool forward,
                             const std::filesystem::path& directory = "shaders");
    void recordForward(vk::CommandBuffer command, const TrainingBuffers& buffers,
                       const TrainingPushConstants& push);
    void recordBackward(const BackwardPassContext& context, bool fused);

private:
    void bind(vk::CommandBuffer command, size_t index, const TrainingBuffers& buffers,
              const TrainingPushConstants& push, bool fused);
    vk::Device device_;
    bool forward_;
    std::vector<ComputePipeline> pipelines_;
    BackwardDescriptorPool descriptors_;
    std::vector<vk::DescriptorSet> sets_;
};
} // namespace vulkan3DGS
