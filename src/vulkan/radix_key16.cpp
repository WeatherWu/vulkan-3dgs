#include "vulkan/radix_key16.hpp"
#include "vulkan/shader.hpp"

#include <algorithm>
#include <stdexcept>

namespace vulkan3DGS {
RadixKey16::RadixKey16(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                       uint32_t family, VrdxSorter sorter)
    : device_(device), physical_(physical), queue_(queue), family_(family), sorter_(sorter) {
    auto create = [&](const char* path) {
        Shader shader;
        shader.createFromSpv(device_, path, vk::ShaderStageFlagBits::eCompute);
        vk::ComputePipelineCreateInfo info;
        info.setStage(shader.getStageCreateInfo())
            .setLayout(vk::PipelineLayout(radixKeyValuePipelineLayout(sorter)));
        auto result = device_.createComputePipelineUnique(nullptr, info);
        return std::move(result.value);
    };
    upsweep_ = create("shaders/train_tile_key16_upsweep.comp.spv");
    downsweep_ = create("shaders/train_tile_key16_downsweep.comp.spv");
}
void RadixKey16::ensure(uint32_t count) {
    const auto layout = radixKey16Layout(
        std::max(count, 1u), physical_.getProperties().limits.minStorageBufferOffsetAlignment);
    const auto limit = physical_.getProperties().limits.maxStorageBufferRange;
    if (layout.keyBytes + 64 > limit || layout.storageBytes + 64 > limit)
        throw std::runtime_error("Physical16 radix workspace exceeds device limit");
    auto create = [&](Buffer& buffer, uint64_t bytes) {
        buffer.create(device_, physical_, queue_, family_, nullptr, bytes,
                      vk::BufferUsageFlagBits::eStorageBuffer |
                          vk::BufferUsageFlagBits::eTransferSrc |
                          vk::BufferUsageFlagBits::eTransferDst,
                      vk::MemoryPropertyFlagBits::eDeviceLocal, {family_});
    };
    if (keys_.getSize() < layout.keyBytes + 64) {
        Buffer next;
        create(next, layout.keyBytes + 64);
        keys_ = std::move(next);
    }
    if (storage_.getSize() < layout.storageBytes + 64) {
        Buffer next;
        create(next, layout.storageBytes + 64);
        storage_ = std::move(next);
    }
}
void RadixKey16::record(vk::CommandBuffer command, uint32_t count, vk::Buffer values) {
    recordRadixKeyValue16(command, sorter_, count, keys_.getBuffer(), values, storage_.getBuffer(),
                          *upsweep_, *downsweep_);
}
} // namespace vulkan3DGS
