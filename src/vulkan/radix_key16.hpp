#pragma once

#include "vulkan/buffer.hpp"
#include "vulkan/radix_key_value.hpp"

namespace vulkan3DGS {
// Layout borrowed from the caller's sorter; this owner must be destroyed first.
class RadixKey16 {
public:
    RadixKey16(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue, uint32_t family,
               VrdxSorter sorter);
    void ensure(uint32_t count);
    void record(vk::CommandBuffer command, uint32_t count, vk::Buffer values);
    vk::DescriptorBufferInfo keysInfo() const {
        return keys_.getDescriptorInfo();
    }
    vk::DescriptorBufferInfo scratchInfo() const {
        return storage_.getDescriptorInfo();
    }
    uint64_t allocatedBytes() const {
        return keys_.getSize() + storage_.getSize();
    }

private:
    vk::Device device_;
    vk::PhysicalDevice physical_;
    vk::Queue queue_;
    uint32_t family_;
    VrdxSorter sorter_;
    Buffer keys_, storage_;
    vk::UniquePipeline upsweep_, downsweep_;
};
} // namespace vulkan3DGS
