#include "training/core/training_faster_sort.hpp"
#include "training/core/training_optimizer_capture_gpu.hpp"
#include "training/core/training_sort_synchronization.hpp"
#include "vulkan/radix_key_value.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace vulkan3DGS {
namespace {
void requireSort(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}
uint64_t aligned(uint64_t bytes, uint64_t alignment) {
    return (bytes + alignment - 1) / alignment * alignment;
}
} // namespace
TrainingFasterSort::TrainingFasterSort(vk::Device device, vk::PhysicalDevice physical,
                                       vk::Queue queue, uint32_t family)
    : device_(device), physical_(physical), queue_(queue), family_(family) {
    const std::array<const char*, 4> names{
        "train_full_sort_prepare_test", "train_full_sort_emit_test",
        "train_range_boundaries_sorted_keys_test", "train_forward_tile_sort"};
    const std::array<std::vector<uint32_t>, 4> bindings{
        {{3, 32, 60, 61, 62, 63}, {3, 21, 24, 32, 62}, {5, 23}, {5}}};
    std::vector<vk::DescriptorSetLayout> layouts;
    auto init = [&](ComputePipeline& pipeline, const std::string& name, const auto& bs,
                    uint32_t pushBytes) {
        ComputePipelineConfig config;
        config.pushConstantSize = pushBytes;
        for (auto b : bs)
            config.descriptorBindings.push_back(backwardStorageBinding(b));
        pipeline.initialize(device, "shaders/" + name + ".comp.spv", config);
    };
    for (size_t i = 0; i < 4; ++i) {
        init(pipelines_[i], names[i], bindings[i], sizeof(TrainingPushConstants));
        layouts.push_back(pipelines_[i].getDescriptorSetLayout());
    }
    const std::array<const char*, 5> suffixes{"ranges", "scratch", "top", "add_scratch",
                                              "add_ranges"};
    const std::array<std::vector<uint32_t>, 5> prefixBindings{
        {{32, 33}, {33}, {9, 33}, {33}, {32, 33}}};
    for (size_t i = 0; i < 5; ++i)
        init(prefixPipelines_[i], std::string("train_forward_gaussian_prefix_") + suffixes[i],
             prefixBindings[i], sizeof(TrainingPrefixPushConstants));
    for (size_t bank = 0; bank < 2; ++bank)
        for (auto& p : prefixPipelines_)
            layouts.push_back(p.getDescriptorSetLayout());
    const auto sets = descriptors_.create(device, 40, 0, layouts);
    std::copy_n(sets.begin(), 4, sets_.begin());
    for (size_t bank = 0; bank < 2; ++bank)
        for (size_t i = 0; i < 5; ++i)
            prefixSets_[bank][i] = sets[4 + bank * 5 + i];
    for (auto& counter : counters_)
        counter.create(device, physical, queue, family, nullptr, 16,
                       vk::BufferUsageFlagBits::eStorageBuffer |
                           vk::BufferUsageFlagBits::eTransferSrc |
                           vk::BufferUsageFlagBits::eTransferDst,
                       vk::MemoryPropertyFlagBits::eDeviceLocal, {family});
    readback_.create(
        device, physical, queue, family, nullptr, 16, vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        {family});
    mapped_ = device.mapMemory(readback_.getMemory(), 0, 16);
}
TrainingFasterSort::~TrainingFasterSort() {
    if (mapped_) device_.unmapMemory(readback_.getMemory());
}
void TrainingFasterSort::ensureCapacity(uint32_t count) {
    if (capacity_ >= std::max(count, 1u)) return;
    const auto limits = physical_.getProperties().limits;
    const uint64_t grown = std::max<uint64_t>(std::max(count, 1u), uint64_t(capacity_) * 3 / 2);
    requireSort(grown <= UINT32_MAX && grown * 8 <= limits.maxStorageBufferRange &&
                    (grown + 255) / 256 <= limits.maxComputeWorkGroupCount[0],
                "Faster sort Gaussian workspace exceeds device limits");
    uint32_t words = 0, blocks = uint32_t((grown + 255) / 256);
    do {
        words += blocks;
        blocks = (blocks + 255) / 256;
    } while (blocks > 1);
    words = std::max(words + 1, 1u);
    checkOptimizerCaptureBudget(physical_, 0, grown * 24 + uint64_t(words) * 4 + 1024);
    std::array<Buffer, 2> ranges;
    Buffer depth, ids, scratch;
    auto create = [&](Buffer& b, uint64_t bytes) {
        b.create(device_, physical_, queue_, family_, nullptr, bytes,
                 vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                     vk::BufferUsageFlagBits::eTransferDst,
                 vk::MemoryPropertyFlagBits::eDeviceLocal, {family_});
    };
    for (auto& range : ranges)
        create(range, grown * 8);
    create(depth, aligned(grown * 4, limits.minStorageBufferOffsetAlignment));
    create(ids, depth.getSize());
    create(scratch, uint64_t(words) * 4);
    ranges_ = std::move(ranges);
    depth_ = std::move(depth);
    ids_ = std::move(ids);
    prefixScratch_ = std::move(scratch);
    capacity_ = uint32_t(grown);
    scratchWords_ = words;
}
void TrainingFasterSort::bindInputs(TrainingBuffers& buffers) {
    const std::array<std::vector<uint32_t>, 4> bindings{
        {{3, 32, 60, 61, 62, 63}, {3, 21, 24, 32, 62}, {5, 23}, {5}}};
    const std::array<std::vector<vk::DescriptorBufferInfo>, 4> infos{
        {{buffers.projectedInfo(), buffers.gaussianTileRangesInfo(), ranges_[0].getDescriptorInfo(),
          depth_.getDescriptorInfo(), ids_.getDescriptorInfo(), ranges_[1].getDescriptorInfo()},
         {buffers.projectedInfo(), buffers.tileSortScratchInfo(), buffers.tileItemsSortedInfo(),
          ranges_[1].getDescriptorInfo(), ids_.getDescriptorInfo()},
         {buffers.tileRangesInfo(), buffers.tileSortScratchInfo()},
         {buffers.tileRangesInfo()}}};
    auto write = [&](vk::DescriptorSet set, const auto& bs, const auto& ds) {
        std::vector<vk::WriteDescriptorSet> writes(bs.size());
        for (size_t i = 0; i < bs.size(); ++i)
            writes[i]
                .setDstSet(set)
                .setDstBinding(bs[i])
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                .setPBufferInfo(&ds[i]);
        device_.updateDescriptorSets(writes, {});
    };
    for (size_t i = 0; i < 4; ++i)
        write(sets_[i], bindings[i], infos[i]);
    const std::array<std::vector<uint32_t>, 5> bs{{{32, 33}, {33}, {9, 33}, {33}, {32, 33}}};
    for (size_t bank = 0; bank < 2; ++bank) {
        const auto r = ranges_[bank].getDescriptorInfo(), s = prefixScratch_.getDescriptorInfo(),
                   c = counters_[bank].getDescriptorInfo();
        const std::array<std::vector<vk::DescriptorBufferInfo>, 5> ds{
            {{r, s}, {s}, {c, s}, {s}, {r, s}}};
        for (size_t i = 0; i < 5; ++i)
            write(prefixSets_[bank][i], bs[i], ds[i]);
    }
}
void TrainingFasterSort::barrier(vk::CommandBuffer command) {
    vk::MemoryBarrier b{vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                            vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eTransferRead,
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                            vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eTransferRead};
    command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                            vk::PipelineStageFlagBits::eAllCommands, {}, b, {}, {});
}
void TrainingFasterSort::dispatch(vk::CommandBuffer command, size_t i, uint32_t count,
                                  const TrainingPushConstants& push) {
    if (!count) return;
    auto& p = pipelines_[i];
    command.bindPipeline(vk::PipelineBindPoint::eCompute, p.getPipeline());
    command.bindDescriptorSets(vk::PipelineBindPoint::eCompute, p.getPipelineLayout(), 0, sets_[i],
                               {});
    command.pushConstants(p.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(push),
                          &push);
    command.dispatch((count + 255) / 256, 1, 1);
}
void TrainingFasterSort::prefix(vk::CommandBuffer command, uint32_t bank, uint32_t count) {
    if (!count) return;
    struct Level {
        uint32_t offset, count;
    };
    std::vector<Level> levels{{0, (count + 255) / 256}};
    uint32_t next = levels[0].count;
    auto run = [&](size_t i, TrainingPrefixPushConstants push, uint32_t groups) {
        auto& p = prefixPipelines_[i];
        command.bindPipeline(vk::PipelineBindPoint::eCompute, p.getPipeline());
        command.bindDescriptorSets(vk::PipelineBindPoint::eCompute, p.getPipelineLayout(), 0,
                                   prefixSets_[bank][i], {});
        command.pushConstants(p.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0,
                              sizeof(push), &push);
        command.dispatch(groups, 1, 1);
        barrier(command);
    };
    TrainingPrefixPushConstants p{};
    p.elementCount = count;
    run(0, p, levels[0].count);
    while (levels.back().count > 256) {
        auto child = levels.back();
        Level parent{next, (child.count + 255) / 256};
        levels.push_back(parent);
        next += parent.count;
        p = {};
        p.elementCount = child.count;
        p.inputOffset = child.offset;
        p.outputOffset = parent.offset;
        run(1, p, parent.count);
    }
    requireSort(next <= scratchWords_, "Faster sort prefix scratch bounds");
    p = {};
    p.elementCount = levels.back().count;
    p.inputOffset = levels.back().offset;
    run(2, p, 1);
    for (size_t i = levels.size(); i > 1; --i) {
        p = {};
        p.elementCount = levels[i - 2].count;
        p.inputOffset = levels[i - 2].offset;
        p.parentOffset = levels[i - 1].offset;
        run(3, p, (p.elementCount + 255) / 256);
    }
    p = {};
    p.elementCount = count;
    run(4, p, (count + 255) / 256);
}
void TrainingFasterSort::prepare(vk::CommandBuffer command, TrainingBuffers& buffers,
                                 const TrainingPushConstants& push) {
    requireSort(push.tileSize == 16, "Faster sort only supports Fixed16");
    ensureCapacity(push.gaussianCount);
    bindInputs(buffers);
    barrier(command);
    for (auto& c : counters_)
        command.fillBuffer(c.getBuffer(), 0, 16, 0);
    barrier(command);
    auto p = push;
    p.validationIteration = 0;
    dispatch(command, 0, p.gaussianCount, p);
    barrier(command);
    prefix(command, 0, p.gaussianCount);
    p.validationIteration = 1;
    dispatch(command, 0, p.gaussianCount, p);
    barrier(command);
    command.copyBuffer(counters_[0].getBuffer(), readback_.getBuffer(), vk::BufferCopy{0, 0, 16});
    vk::MemoryBarrier host{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead};
    command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                            {}, host, {}, {});
    preparedPush_ = push;
    prepared_ = true;
}
void TrainingFasterSort::build(vk::CommandBuffer command, TrainingBuffers& buffers,
                               const TrainingPushConstants& push, VrdxSorter sorter,
                               bool physical16) {
    requireSort(!physical16 || buffers.storage16Enabled(),
                "Physical16 requires enabled device features");
    requireSort(prepared_ && push.gaussianCount == preparedPush_.gaussianCount &&
                    push.width == preparedPush_.width && push.height == preparedPush_.height &&
                    push.tileSize == 16,
                "Faster sort prepare/main mismatch");
    std::memcpy(&visibleCount_, mapped_, 4);
    requireSort(visibleCount_ <= push.gaussianCount &&
                    (visibleCount_ == 0) == (push.tileItemCount == 0),
                "Faster sort GPU count invalid");
    const uint64_t alignment = physical_.getProperties().limits.minStorageBufferOffsetAlignment;
    const uint64_t alignedItems = radixKeyValueCapacity(push.tileItemCount, alignment);
    const auto limits = physical_.getProperties().limits;
    requireSort(alignedItems <= UINT32_MAX && alignedItems * 4 <= limits.maxStorageBufferRange &&
                    (uint64_t(push.tileItemCount) + 256) / 256 <=
                        limits.maxComputeWorkGroupCount[0],
                "Faster sort aligned item capacity/device limit");
    if (alignedItems > buffers.tileItemCapacity()) buffers.resizeTileItems(uint32_t(alignedItems));
    VrdxSorterStorageRequirements requirements{};
    vrdxGetSorterKeyValueStorageRequirements(
        sorter,
        physical16 ? std::max(visibleCount_, 2u)
                   : std::max({push.gaussianCount, push.tileItemCount, 2u}),
        &requirements);
    requireSort(requirements.size <= limits.maxStorageBufferRange,
                "Faster sort radix workspace limit");
    if (sortStorage_.getSize() < requirements.size) {
        checkOptimizerCaptureBudget(physical_, 0, requirements.size);
        Buffer next;
        next.create(device_, physical_, queue_, family_, nullptr, requirements.size,
                    vk::BufferUsageFlags(requirements.usage) |
                        vk::BufferUsageFlagBits::eTransferSrc |
                        vk::BufferUsageFlagBits::eTransferDst,
                    vk::MemoryPropertyFlagBits::eDeviceLocal, {family_});
        sortStorage_ = std::move(next);
    }
    bindInputs(buffers);
    if (physical16) {
        if (!key16_) {
            key16_ = std::make_unique<RadixKey16>(device_, physical_, queue_, family_, sorter);
            const std::array<const char*, 2> names{"train_full_sort_emit_test_u16",
                                                   "train_range_boundaries_sorted_keys_test_u16"};
            const std::array<std::vector<uint32_t>, 2> bindings{{{3, 21, 24, 32, 62}, {5, 23}}};
            std::vector<vk::DescriptorSetLayout> layouts;
            for (size_t i = 0; i < 2; ++i) {
                ComputePipelineConfig config;
                config.pushConstantSize = sizeof(TrainingPushConstants);
                for (auto binding : bindings[i])
                    config.descriptorBindings.push_back(backwardStorageBinding(binding));
                key16Pipelines_[i].initialize(
                    device_, std::string("shaders/") + names[i] + ".comp.spv", config);
                layouts.push_back(key16Pipelines_[i].getDescriptorSetLayout());
            }
            const auto sets = key16Descriptors_.create(device_, 7, 0, layouts);
            std::copy_n(sets.begin(), 2, key16Sets_.begin());
        }
        key16_->ensure(push.tileItemCount);
        const std::array<std::vector<uint32_t>, 2> bindings{{{3, 21, 24, 32, 62}, {5, 23}}};
        const std::array<std::vector<vk::DescriptorBufferInfo>, 2> infos{
            {{buffers.projectedInfo(), key16_->keysInfo(), buffers.tileItemsSortedInfo(),
              ranges_[1].getDescriptorInfo(), ids_.getDescriptorInfo()},
             {buffers.tileRangesInfo(), key16_->keysInfo()}}};
        for (size_t bank = 0; bank < 2; ++bank) {
            std::vector<vk::WriteDescriptorSet> writes(bindings[bank].size());
            for (size_t i = 0; i < writes.size(); ++i)
                writes[i]
                    .setDstSet(key16Sets_[bank])
                    .setDstBinding(bindings[bank][i])
                    .setDescriptorCount(1)
                    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                    .setPBufferInfo(&infos[bank][i]);
            device_.updateDescriptorSets(writes, {});
        }
    }
    auto dispatch16 = [&](size_t index, uint32_t count, const TrainingPushConstants& p) {
        if (!count) return;
        auto& pipeline = key16Pipelines_[index];
        command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getPipeline());
        command.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline.getPipelineLayout(), 0,
                                   key16Sets_[index], {});
        command.pushConstants(pipeline.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0,
                              sizeof(p), &p);
        command.dispatch((count + 255u) / 256u, 1, 1);
    };
    barrier(command);
    auto sort = [&](uint32_t count, vk::Buffer keys, vk::Buffer values, uint32_t bits) {
        if (count <= 1) return;
        const auto reuse = trainingSortScratchReuseBarrier(sortStorage_.getBuffer());
        command.pipelineBarrier(kTrainingSortScratchReuseSourceStages,
                                kTrainingSortScratchReuseDestinationStages, {}, 0, nullptr, 1,
                                &reuse, 0, nullptr);
        recordRadixKeyValue(command, sorter, count, keys, values, sortStorage_.getBuffer(), bits);
        barrier(command);
    };
    sort(visibleCount_, depth_.getBuffer(), ids_.getBuffer(), 32);
    auto p = push;
    p.gaussianCount = visibleCount_;
    p.validationIteration = 2;
    dispatch(command, 0, visibleCount_, p);
    barrier(command);
    prefix(command, 1, visibleCount_);
    if (physical16)
        dispatch16(0, visibleCount_, p);
    else
        dispatch(command, 1, visibleCount_, p);
    barrier(command);
    const uint64_t tiles = ((uint64_t(push.width) + 15) / 16) * ((uint64_t(push.height) + 15) / 16);
    if (physical16) {
        requireSort(tiles <= 65536, "Physical16 tile overflow");
        key16_->record(command, push.tileItemCount, buffers.tileItemsSortedInfo().buffer);
        barrier(command);
        dispatch16(1, push.tileItemCount + 1, push);
    } else {
        sort(push.tileItemCount, buffers.tileSortScratchInfo().buffer,
             buffers.tileItemsSortedInfo().buffer, tiles <= 65536 ? 16 : 32);
        dispatch(command, 2, push.tileItemCount + 1, push);
    }
    barrier(command);
    requireSort(tiles <= UINT32_MAX, "Faster sort tile count overflow");
    dispatch(command, 3, uint32_t(tiles), push);
    barrier(command);
    prepared_ = false;
}
} // namespace vulkan3DGS
