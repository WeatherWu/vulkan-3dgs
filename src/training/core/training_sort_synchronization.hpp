#pragma once

#include <vulkan/vulkan.hpp>

namespace vulkan3DGS {

inline constexpr vk::PipelineStageFlags kTrainingSortScratchReuseSourceStages =
    vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eComputeShader;
inline constexpr vk::PipelineStageFlags kTrainingSortScratchReuseDestinationStages =
    vk::PipelineStageFlagBits::eTransfer;

inline vk::BufferMemoryBarrier trainingSortScratchReuseBarrier(vk::Buffer scratch) {
    vk::BufferMemoryBarrier barrier;
    // Radix count/histogram resets overwrite storage also accessed by its compute passes.
    barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite |
                              vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite)
        .setDstAccessMask(vk::AccessFlagBits::eTransferWrite)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setBuffer(scratch)
        .setOffset(0)
        .setSize(VK_WHOLE_SIZE);
    return barrier;
}

} // namespace vulkan3DGS
