#include "training/core/training_faster_sort_key16.hpp"
#include "training/core/training_faster_sort.hpp"
#include "training/core/training_optimizer_capture_gpu.hpp"
#include "training/core/training_sort_synchronization.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
namespace vulkan3DGS {
namespace {
void requireSort(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
} // namespace
void TrainingFasterSortKey16::build(TrainingFasterSort& owner, vk::CommandBuffer command,
                                    TrainingBuffers& buffers, const TrainingPushConstants& push,
                                    VrdxSorter sorter) {
    auto& prepared_ = owner.prepared_;
    auto& preparedPush_ = owner.preparedPush_;
    auto& mapped_ = owner.mapped_;
    auto& visibleCount_ = owner.visibleCount_;
    auto& physical_ = owner.physical_;
    auto& sortStorage_ = owner.sortStorage_;
    auto& device_ = owner.device_;
    auto& queue_ = owner.queue_;
    auto& family_ = owner.family_;
    auto& ranges_ = owner.ranges_;
    auto& ids_ = owner.ids_;
    auto& depth_ = owner.depth_;

    requireSort(buffers.storage16Enabled(), "Physical16 requires enabled device features");
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
    vrdxGetSorterKeyValueStorageRequirements(sorter, std::max(visibleCount_, 2u), &requirements);
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
    owner.bindInputs(buffers);
    {
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
    TrainingFasterSort::barrier(command);
    auto sort = [&](uint32_t count, vk::Buffer keys, vk::Buffer values, uint32_t bits) {
        if (count <= 1) return;
        const auto reuse = trainingSortScratchReuseBarrier(sortStorage_.getBuffer());
        command.pipelineBarrier(kTrainingSortScratchReuseSourceStages,
                                kTrainingSortScratchReuseDestinationStages, {}, 0, nullptr, 1,
                                &reuse, 0, nullptr);
        recordRadixKeyValue(command, sorter, count, keys, values, sortStorage_.getBuffer(), bits);
        TrainingFasterSort::barrier(command);
    };
    sort(visibleCount_, depth_.getBuffer(), ids_.getBuffer(), 32);
    auto p = push;
    p.gaussianCount = visibleCount_;
    p.validationIteration = 2;
    owner.dispatch(command, 0, visibleCount_, p);
    TrainingFasterSort::barrier(command);
    owner.prefix(command, 1, visibleCount_);
    dispatch16(0, visibleCount_, p);
    TrainingFasterSort::barrier(command);
    const uint64_t tiles = ((uint64_t(push.width) + 15) / 16) * ((uint64_t(push.height) + 15) / 16);
    requireSort(tiles <= 65536, "Physical16 tile overflow");
    key16_->record(command, push.tileItemCount, buffers.tileItemsSortedInfo().buffer);
    TrainingFasterSort::barrier(command);
    dispatch16(1, push.tileItemCount + 1, push);
    TrainingFasterSort::barrier(command);
    requireSort(tiles <= UINT32_MAX, "Faster sort tile count overflow");
    owner.dispatch(command, 3, uint32_t(tiles), push);
    TrainingFasterSort::barrier(command);
    prepared_ = false;
}
} // namespace vulkan3DGS
