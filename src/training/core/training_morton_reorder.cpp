#include "training/core/training_morton_reorder.hpp"

#include "vulkan/radix_key_value.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace vulkan3DGS {
namespace {
void visibility(vk::CommandBuffer command) {
    vk::MemoryBarrier barrier{vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
                              vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite};
    command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                            vk::PipelineStageFlagBits::eAllCommands, {}, barrier, {}, {});
}
void requireMemory(vk::PhysicalDevice physical, uint64_t bytes) {
    bool available = false;
    for (const auto& extension : physical.enumerateDeviceExtensionProperties())
        available |=
            std::strcmp(extension.extensionName.data(), VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0;
    if (!available) return;
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget;
    vk::PhysicalDeviceMemoryProperties2 properties;
    properties.pNext = &budget;
    physical.getMemoryProperties2(&properties);
    uint64_t freeBytes = 0;
    for (uint32_t i = 0; i < properties.memoryProperties.memoryHeapCount; ++i)
        if (properties.memoryProperties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
            freeBytes = std::max(freeBytes, budget.heapBudget[i] > budget.heapUsage[i]
                                                ? budget.heapBudget[i] - budget.heapUsage[i]
                                                : 0);
    if (freeBytes < bytes)
        throw std::runtime_error("Z-order workspace exceeds measured available device memory");
}
} // namespace

void TrainingMortonReorder::SorterDeleter::operator()(VrdxSorter_T* sorter) const noexcept {
    if (sorter) vrdxDestroySorter(sorter);
}

TrainingMortonReorder::TrainingMortonReorder(vk::Device device, vk::PhysicalDevice physical,
                                             vk::Queue queue, uint32_t family)
    : device_(device), physical_(physical), queue_(queue), family_(family) {
    ComputePipelineConfig config;
    config.pushConstantSize = 16;
    for (uint32_t binding = 0; binding < 10; ++binding)
        config.descriptorBindings.push_back(backwardStorageBinding(binding));
    pipeline_.initialize(device, "shaders/train_morton_reorder.comp.spv", config);
    set_ = descriptors_.create(device, 10, 0, {pipeline_.getDescriptorSetLayout()}).front();
    pool_.create(device, family, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    command_ = pool_.allocateCommandBuffer();
    fence_ = device.createFenceUnique({});
    timestampBits_ = physical.getQueueFamilyProperties().at(family).timestampValidBits;
    timestampPeriod_ = physical.getProperties().limits.timestampPeriod;
    if (timestampBits_) queries_ = device.createQueryPoolUnique({{}, vk::QueryType::eTimestamp, 2});
    VrdxSorter raw = nullptr;
    VrdxSorterCreateInfo info{};
    info.device = device;
    info.physicalDevice = physical;
    vrdxCreateSorter(&info, &raw);
    sorter_.reset(raw);
}
TrainingMortonReorder::~TrainingMortonReorder() = default;

TrainingMortonResult TrainingMortonReorder::execute(TrainingBuffers& buffers, uint32_t count) {
    if (!count) return {};
    if (count > buffers.gaussianCapacity())
        throw std::invalid_argument("Z-order count exceeds active Gaussian capacity");
    const auto limits = physical_.getProperties().limits;
    const uint32_t groups = uint32_t((uint64_t(count) + 255) / 256);
    if (groups > limits.maxComputeWorkGroupCount[0] || count > UINT32_MAX / 30u)
        throw std::runtime_error("Z-order dispatch exceeds device/index limits");
    const std::array<vk::DescriptorBufferInfo, 3> source{
        buffers.gaussianParamsInfo(), buffers.adamStatesInfo(), buffers.densificationStatesInfo()};
    const uint64_t keyBytes =
        uint64_t(radixKeyValueCapacity(count, limits.minStorageBufferOffsetAlignment)) * 4;
    const uint64_t boundsBytes = (uint64_t(groups) + 1) * 32;
    VrdxSorterStorageRequirements requirements{};
    vrdxGetSorterKeyValueStorageRequirements(sorter_.get(), std::max(count, 2u), &requirements);
    const std::array<uint64_t, 8> sizes{source[0].range,     source[1].range,  source[2].range,
                                        uint64_t(count) * 8, keyBytes,         keyBytes,
                                        boundsBytes,         requirements.size};
    TrainingMortonResult result;
    for (auto bytes : sizes) {
        if (bytes > limits.maxStorageBufferRange)
            throw std::runtime_error("Z-order buffer exceeds maxStorageBufferRange");
        result.workspaceBytes += bytes;
    }
    requireMemory(physical_, result.workspaceBytes);
    auto targets = buffers.createGaussianReorderTargets();
    Buffer bounds, codes, keys, indices, storage, readback;
    const auto usage = vk::BufferUsageFlagBits::eStorageBuffer |
                       vk::BufferUsageFlagBits::eTransferSrc |
                       vk::BufferUsageFlagBits::eTransferDst;
    auto create = [&](Buffer& buffer, uint64_t bytes, vk::BufferUsageFlags extra = {}) {
        buffer.create(device_, physical_, queue_, family_, nullptr, bytes, usage | extra,
                      vk::MemoryPropertyFlagBits::eDeviceLocal, {family_});
    };
    create(bounds, boundsBytes);
    create(codes, uint64_t(count) * 8);
    create(keys, keyBytes);
    create(indices, keyBytes);
    create(storage, requirements.size, vk::BufferUsageFlags(requirements.usage));
    readback.create(
        device_, physical_, queue_, family_, nullptr, 32, vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        {family_});
    const std::array<vk::DescriptorBufferInfo, 10> infos{source[0],
                                                         source[1],
                                                         source[2],
                                                         bounds.getDescriptorInfo(),
                                                         codes.getDescriptorInfo(),
                                                         keys.getDescriptorInfo(),
                                                         indices.getDescriptorInfo(),
                                                         targets[0].getDescriptorInfo(),
                                                         targets[1].getDescriptorInfo(),
                                                         targets[2].getDescriptorInfo()};
    std::array<vk::WriteDescriptorSet, 10> writes;
    for (uint32_t i = 0; i < writes.size(); ++i)
        writes[i]
            .setDstSet(set_)
            .setDstBinding(i)
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(&infos[i]);
    device_.updateDescriptorSets(writes, {});
    command_.reset();
    command_.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    if (queries_) {
        command_.resetQueryPool(*queries_, 0, 2);
        command_.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *queries_, 0);
    }
    visibility(command_);
    // Preserve inactive capacity bytes while replacing the live prefix by a complete gather.
    for (size_t i = 0; i < targets.size(); ++i)
        command_.copyBuffer(source[i].buffer, targets[i].getBuffer(),
                            vk::BufferCopy{source[i].offset, 0, source[i].range});
    visibility(command_);
    auto dispatch = [&](uint32_t stage, uint32_t workgroups) {
        const std::array<uint32_t, 4> push{stage, count, groups, 0};
        command_.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_.getPipeline());
        command_.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_.getPipelineLayout(),
                                    0, set_, {});
        command_.pushConstants(pipeline_.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0,
                               sizeof(push), push.data());
        command_.dispatch(workgroups, 1, 1);
        visibility(command_);
    };
    dispatch(0, groups);
    dispatch(1, 1);
    dispatch(2, groups);
    recordRadixKeyValue(command_, sorter_.get(), count, keys.getBuffer(), indices.getBuffer(),
                        storage.getBuffer(), 32);
    visibility(command_);
    dispatch(3, groups);
    recordRadixKeyValue(command_, sorter_.get(), count, keys.getBuffer(), indices.getBuffer(),
                        storage.getBuffer(), 32);
    visibility(command_);
    dispatch(4, groups);
    command_.copyBuffer(bounds.getBuffer(), readback.getBuffer(), vk::BufferCopy{0, 0, 32});
    vk::MemoryBarrier host{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead};
    command_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                             {}, host, {}, {});
    if (queries_) command_.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *queries_, 1);
    command_.end();
    device_.resetFences(*fence_);
    vk::SubmitInfo submit;
    submit.setCommandBuffers(command_);
    queue_.submit(submit, *fence_);
    (void)device_.waitForFences(*fence_, true, UINT64_MAX);
    std::array<float, 8> status{};
    void* mapped = device_.mapMemory(readback.getMemory(), 0, 32);
    std::memcpy(status.data(), mapped, 32);
    device_.unmapMemory(readback.getMemory());
    if (status[7] != 0)
        throw std::runtime_error(
            "Z-order rejected non-finite positions or bounding extent; model unchanged");
    if (queries_) {
        std::array<uint64_t, 2> times{};
        const auto code =
            vkGetQueryPoolResults(device_, *queries_, 0, 2, sizeof(times), times.data(),
                                  sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        if (code != VK_SUCCESS) throw std::runtime_error("Z-order GPU timestamps unavailable");
        const uint64_t mask =
            timestampBits_ == 64 ? UINT64_MAX : ((uint64_t(1) << timestampBits_) - 1);
        result.gpuMs = float((times[1] - times[0]) & mask) * timestampPeriod_ / 1.0e6f;
    }
    buffers.adoptGaussianReorder(targets);
    return result;
}

} // namespace vulkan3DGS
