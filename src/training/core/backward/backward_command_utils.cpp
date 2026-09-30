#include "training/core/backward/backward_command_utils.hpp"

#include <array>
#include <stdexcept>

namespace vulkan3DGS {

vk::DescriptorSetLayoutBinding backwardStorageBinding(uint32_t binding) {
    vk::DescriptorSetLayoutBinding layoutBinding{};
    layoutBinding.setBinding(binding)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eCompute);
    return layoutBinding;
}

vk::DescriptorSetLayoutBinding backwardUniformBinding(uint32_t binding) {
    vk::DescriptorSetLayoutBinding layoutBinding{};
    layoutBinding.setBinding(binding)
        .setDescriptorType(vk::DescriptorType::eUniformBuffer)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eCompute);
    return layoutBinding;
}

void recordBackwardBufferBarrier(vk::CommandBuffer commandBuffer,
                                 std::initializer_list<vk::DescriptorBufferInfo> buffers,
                                 vk::AccessFlags destinationAccessMask) {
    std::vector<vk::BufferMemoryBarrier> barriers;
    barriers.reserve(buffers.size());
    for (const vk::DescriptorBufferInfo& bufferInfo : buffers) {
        vk::BufferMemoryBarrier barrier{};
        barrier.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
            .setDstAccessMask(destinationAccessMask)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setBuffer(bufferInfo.buffer)
            .setOffset(bufferInfo.offset)
            .setSize(bufferInfo.range);
        barriers.push_back(barrier);
    }
    if (barriers.empty()) {
        return;
    }
    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer, {}, 0,
        nullptr, static_cast<uint32_t>(barriers.size()), barriers.data(), 0, nullptr);
}

void writeBackwardProfilingTimestamp(vk::CommandBuffer commandBuffer, vk::QueryPool queryPool,
                                     TrainingGpuProfileStage stage, bool end) {
    if (queryPool && commandBuffer) {
        commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, queryPool,
                                     trainingGpuTimestampQuery(stage, end));
    }
}

BackwardDescriptorPool::~BackwardDescriptorPool() {
    reset();
}

std::vector<vk::DescriptorSet>
BackwardDescriptorPool::create(vk::Device device, uint32_t storageDescriptorCount,
                               uint32_t uniformDescriptorCount,
                               const std::vector<vk::DescriptorSetLayout>& layouts) {
    reset();
    if (!device || layouts.empty()) {
        throw std::invalid_argument("Backward descriptor pool requires a device and layouts");
    }
    device_ = device;
    std::vector<vk::DescriptorPoolSize> poolSizes;
    if (storageDescriptorCount > 0u) {
        poolSizes.emplace_back(vk::DescriptorType::eStorageBuffer, storageDescriptorCount);
    }
    if (uniformDescriptorCount > 0u) {
        poolSizes.emplace_back(vk::DescriptorType::eUniformBuffer, uniformDescriptorCount);
    }
    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.setPoolSizes(poolSizes).setMaxSets(static_cast<uint32_t>(layouts.size()));
    pool_ = device_.createDescriptorPool(poolInfo);

    vk::DescriptorSetAllocateInfo allocation{};
    allocation.setDescriptorPool(pool_).setSetLayouts(layouts);
    return device_.allocateDescriptorSets(allocation);
}

void BackwardDescriptorPool::reset() {
    if (device_ && pool_) {
        device_.destroyDescriptorPool(pool_);
    }
    pool_ = nullptr;
    device_ = nullptr;
}

} // namespace vulkan3DGS
