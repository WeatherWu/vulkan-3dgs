#pragma once

#include "training/core/training_buffers.hpp"
#include "training/core/training_types.hpp"

#include <vulkan/vulkan.hpp>

namespace vulkan3DGS {

struct BackwardPassContext {
    const TrainingBuffers& buffers;
    vk::CommandBuffer commandBuffer;
    TrainingPushConstants pushConstants;
    vk::QueryPool profilingQueryPool = nullptr;
};

} // namespace vulkan3DGS
