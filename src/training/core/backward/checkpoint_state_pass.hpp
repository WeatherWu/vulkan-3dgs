#pragma once

#include "training/core/backward/backward_command_utils.hpp"
#include "training/core/training_checkpoint_resources.hpp"
#include "vulkan/compute_pipeline.hpp"
#include <string>

namespace vulkan3DGS {

// P1 diagnostic state producer. It never writes rendered pixels or gradients.
class CheckpointStatePass {
public:
    void initialize(vk::Device device, vk::PhysicalDevice physicalDevice,
                    const std::string& shaderDirectory = "shaders");
    void cleanup();
    void record(vk::CommandBuffer command, const TrainingPushConstants& params,
                const TrainingCheckpointResources& resources, vk::QueryPool queries = nullptr,
                bool resetRun = true);

private:
    vk::Device device_ = nullptr;
    vk::PhysicalDeviceLimits limits_{};
    std::array<ComputePipeline, 4> pipelines_;
    std::array<vk::DescriptorSet, 4> sets_{};
    BackwardDescriptorPool pool_;
    vk::Buffer lastControl_ = nullptr;
};

} // namespace vulkan3DGS
