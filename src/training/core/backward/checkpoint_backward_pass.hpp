#pragma once
#include "training/core/backward/backward_command_utils.hpp"
#include "training/core/training_checkpoint_resources.hpp"
#include "vulkan/compute_pipeline.hpp"
#include <string>

namespace vulkan3DGS {
class CheckpointBackwardPass {
public:
    void initialize(vk::Device device, vk::PhysicalDevice physical,
                    const std::string& shaderDirectory = "shaders",
                    uint32_t requiredSubgroupSize = 0u, bool fullSubgroups = false);
    void cleanup();
    void record(vk::CommandBuffer command, const TrainingPushConstants& params,
                const TrainingCheckpointResources& resources, vk::DescriptorBufferInfo pixelGrads,
                vk::DescriptorBufferInfo projectedGrads);

private:
    vk::Device device_ = nullptr;
    ComputePipeline pipeline_;
    BackwardDescriptorPool pool_;
    vk::DescriptorSet set_ = nullptr;
};
} // namespace vulkan3DGS
