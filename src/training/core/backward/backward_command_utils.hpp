#pragma once

#include "training/core/training_types.hpp"

#include <initializer_list>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace vulkan3DGS {

[[nodiscard]] vk::DescriptorSetLayoutBinding backwardStorageBinding(uint32_t binding);
[[nodiscard]] vk::DescriptorSetLayoutBinding backwardUniformBinding(uint32_t binding);

void recordBackwardBufferBarrier(vk::CommandBuffer commandBuffer,
                                 std::initializer_list<vk::DescriptorBufferInfo> buffers,
                                 vk::AccessFlags destinationAccessMask);

void writeBackwardProfilingTimestamp(vk::CommandBuffer commandBuffer, vk::QueryPool queryPool,
                                     TrainingGpuProfileStage stage, bool end);

class BackwardDescriptorPool {
public:
    BackwardDescriptorPool() = default;
    ~BackwardDescriptorPool();

    BackwardDescriptorPool(const BackwardDescriptorPool&) = delete;
    BackwardDescriptorPool& operator=(const BackwardDescriptorPool&) = delete;

    [[nodiscard]] std::vector<vk::DescriptorSet>
    create(vk::Device device, uint32_t storageDescriptorCount, uint32_t uniformDescriptorCount,
           const std::vector<vk::DescriptorSetLayout>& layouts);
    void reset();

private:
    vk::Device device_ = nullptr;
    vk::DescriptorPool pool_ = nullptr;
};

} // namespace vulkan3DGS
