#pragma once

#include <array>
#include <vulkan/vulkan.hpp>

#include "gaussian_training/training_types.hpp"
#include "vulkan/compute_pipeline.hpp"

namespace vk_gs {

class TrainingPipelines {
public:
    void initialize(vk::Device device);
    void cleanup();

    ComputePipeline& clear() { return clear_; }
    ComputePipeline& project() { return project_; }
    ComputePipeline& forward() { return forward_; }
    ComputePipeline& loss() { return loss_; }
    ComputePipeline& packRenderBuffer() { return packRenderBuffer_; }

private:
    static ComputePipelineConfig createCommonConfig();

    ComputePipeline clear_;
    ComputePipeline project_;
    ComputePipeline forward_;
    ComputePipeline loss_;
    ComputePipeline packRenderBuffer_;
};

} // namespace vk_gs
