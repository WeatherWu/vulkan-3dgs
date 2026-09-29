#pragma once

#include <array>
#include <vulkan/vulkan.hpp>

#include "training/core/training_types.hpp"
#include "vulkan/compute_pipeline.hpp"

namespace vulkan3DGS {

class TrainingPipelines {
public:
    void initialize(vk::Device device);
    void cleanup();

    ComputePipeline& project() { return project_; }
    ComputePipeline& forward() { return forward_; }
    ComputePipeline& loss() { return loss_; }
    ComputePipeline& packRenderBuffer() { return packRenderBuffer_; }

private:
    static ComputePipelineConfig createCommonConfig();

    ComputePipeline project_;
    ComputePipeline forward_;
    ComputePipeline loss_;
    ComputePipeline packRenderBuffer_;
};

} // namespace vulkan3DGS
