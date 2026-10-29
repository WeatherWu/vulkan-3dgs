#pragma once

#include "training/core/training_buffers.hpp"
#include "training/core/training_types.hpp"

#include <vulkan/vulkan.hpp>

namespace vulkan3DGS {
class TrainingOptimizerCaptureSession;

struct BackwardPassContext {
    const TrainingBuffers& buffers;
    vk::CommandBuffer commandBuffer;
    TrainingPushConstants pushConstants;
    vk::QueryPool profilingQueryPool = nullptr;
    bool checkpointWarp = false;
    bool tensorEarlyCull = false;
    TrainingOptimizerCaptureSession* optimizerCapture = nullptr; // One-shot caller owns through fence.
    bool directIntrinsics = false;
};

} // namespace vulkan3DGS
