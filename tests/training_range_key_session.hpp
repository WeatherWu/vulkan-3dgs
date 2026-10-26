#pragma once

#include "training_range_key_reference.hpp"
#include "training_sort_replay_transfer.hpp"
#include "training/core/backward/backward_command_utils.hpp"
#include "training/core/training_optimizer_capture_gpu.hpp"
#include "training/core/training_scheduling_diagnostics.hpp"
#include "training/core/training_sort_synchronization.hpp"
#include "vulkan/compute_pipeline.hpp"
#include "utils/system_memory.hpp"
#include <vk_radix_sort.h>

#include <chrono>

namespace range_keys {
inline constexpr uint32_t guardWords = 16, canary = 0xcdcdcdcd;
struct Sample { double sequenceMs = 0, rangeMs = 0, boundaryMs = 0, restoreCpuMs = 0, submitWaitMs = 0; };

// Owns private buffers only; no renderer policy or live model access.
class Session {
public:
    Session(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue, uint32_t family,
            const Inputs& input)
        : device_(device), physical_(physical), queue_(queue), family_(family), input_(input),
          expected_(oracle(input)), transfer_(device, physical, queue, family) {}
    ~Session() { try { device_.waitIdle(); } catch (...) {} }
    void initialize() {
        const auto& h = input_.header.source;
        n_ = h.itemCount; capacity_ = input_.header.capacity;
        const auto limits = physical_.getProperties().limits;
        const uint64_t alignment = limits.minStorageBufferOffsetAlignment;
        const uint64_t alignedActiveBytes = ((uint64_t(n_) * 4 + alignment - 1) / alignment) * alignment;
        bytes_ = std::max(uint64_t(capacity_ + guardWords) * 4, alignedActiveBytes);
        rangeBytes_ = uint64_t(h.tileCount) * 16;
        check(capacity_ >= std::max(n_, 1u) && capacity_ <= (1u << 24) &&
                  n_ <= (1u << 22) && bytes_ <= limits.maxStorageBufferRange &&
                  rangeBytes_ + 64 <= limits.maxStorageBufferRange &&
                  limits.maxComputeWorkGroupInvocations >= 512 && limits.maxComputeWorkGroupSize[0] >= 512 &&
                  (n_ + 256ull) / 256 <= limits.maxComputeWorkGroupCount[0],
              "Range diagnostic exceeds device/workload bounds");
        check(availableSystemMemoryBytes() >= bytes_ * 5 + (256ull << 20), "Range diagnostic RAM budget");
        VrdxSorterCreateInfo info{physical_, device_, VK_NULL_HANDLE};
        VrdxSorter sorter = nullptr;
        vrdxCreateSorter(&info, &sorter);
        sorter_.reset(sorter);
        VrdxSorterStorageRequirements storage{};
        vrdxGetSorterKeyValueStorageRequirements(sorter, std::max(n_, 2u), &storage);
        scratchBytes_ = storage.size;
        check(scratchBytes_ + 64 <= limits.maxStorageBufferRange, "Range sorter descriptor limit");
        checkOptimizerCaptureBudget(physical_, 0, bytes_ * 8 + scratchBytes_ + rangeBytes_ + 128);
        auto create = [&](Buffer& buffer, uint64_t bytes) {
            buffer.create(device_, physical_, queue_, family_, nullptr, bytes,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, {family_});
        };
        for (auto& buffer : seeds_) create(buffer, bytes_);
        for (auto& buffer : work_) create(buffer, bytes_);
        create(storage_, scratchBytes_ + 64);
        create(ranges_, rangeBytes_ + 64);
        const std::array<const std::vector<uint32_t>*, 4> vectors{
            &input_.depthKeys, &input_.indices, &input_.tileKeys, &input_.items};
        for (size_t i = 0; i < vectors.size(); ++i) {
            std::vector<uint32_t> padded(bytes_ / 4, canary);
            std::copy(vectors[i]->begin(), vectors[i]->end(), padded.begin());
            uploadSortReplay(transfer_, seeds_[i].getDescriptorInfo(), std::as_bytes(std::span(padded)));
        }
        const std::array<std::vector<uint32_t>, 5> bindings{{{5, 21, 22}, {5, 23}, {21, 22, 23}, {4, 22, 24}, {5}}};
        const std::array<const char*, 5> names{"train_forward_tile_range_boundaries",
            "train_range_boundaries_sorted_keys_test", "train_forward_tile_gather_high",
            "train_forward_tile_gather_items", "train_forward_tile_sort"};
        std::vector<vk::DescriptorSetLayout> layouts;
        for (size_t i = 0; i < pipelines_.size(); ++i) {
            ComputePipelineConfig config;
            for (auto binding : bindings[i]) config.descriptorBindings.push_back(backwardStorageBinding(binding));
            config.pushConstantSize = sizeof(TrainingPushConstants);
            pipelines_[i].initialize(device_, std::string("shaders/") + names[i] + ".comp.spv", config);
            layouts.push_back(pipelines_[i].getDescriptorSetLayout());
        }
        const auto sets = descriptors_.create(device_, 12, 0, layouts);
        std::copy(sets.begin(), sets.end(), sets_.begin());
        auto rangeInfo = ranges_.getDescriptorInfo(); rangeInfo.range = rangeBytes_;
        const std::array<std::vector<vk::DescriptorBufferInfo>, 5> infos{{
            {rangeInfo, seeds_[2].getDescriptorInfo(), work_[1].getDescriptorInfo()},
            {rangeInfo, n_ > 1 ? work_[2].getDescriptorInfo() : seeds_[2].getDescriptorInfo()},
            {seeds_[2].getDescriptorInfo(), work_[1].getDescriptorInfo(), work_[2].getDescriptorInfo()},
            {seeds_[3].getDescriptorInfo(), work_[1].getDescriptorInfo(), work_[3].getDescriptorInfo()},
            {rangeInfo}}};
        for (size_t i = 0; i < sets_.size(); ++i) {
            std::vector<vk::WriteDescriptorSet> writes(bindings[i].size());
            for (size_t j = 0; j < writes.size(); ++j)
                writes[j].setDstSet(sets_[i]).setDstBinding(bindings[i][j]).setDescriptorCount(1)
                    .setDescriptorType(vk::DescriptorType::eStorageBuffer).setPBufferInfo(&infos[i][j]);
            device_.updateDescriptorSets(writes, {});
        }
        push_.gaussianCount = h.gaussianCount; push_.tileItemCount = n_;
        push_.width = h.width; push_.height = h.height; push_.pixelCount = h.width * h.height;
        push_.tileSize = 16; push_.optimizerEnabled = 0;
        pool_.create(device_, family_, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        command_ = pool_.allocateCommandBuffer();
        fence_ = device_.createFenceUnique({});
        bits_ = physical_.getQueueFamilyProperties()[family_].timestampValidBits;
        period_ = limits.timestampPeriod;
        check(bits_ && period_ > 0, "Range diagnostic compute timestamps unavailable");
        queries_ = device_.createQueryPoolUnique({{}, vk::QueryType::eTimestamp, 4});
    }
    Sample run(bool direct) {
        Sample out;
        begin();
        allBarrier();
        for (size_t i = 0; i < 2; ++i)
            command_.copyBuffer(seeds_[i].getBuffer(), work_[i].getBuffer(), vk::BufferCopy{0, 0, bytes_});
        for (size_t i : {2u, 3u}) command_.fillBuffer(work_[i].getBuffer(), 0, bytes_, canary);
        command_.fillBuffer(storage_.getBuffer(), 0, scratchBytes_ + 64, canary);
        command_.fillBuffer(ranges_.getBuffer(), 0, rangeBytes_, 0);
        command_.fillBuffer(ranges_.getBuffer(), rangeBytes_, 64, canary);
        allBarrier();
        out.restoreCpuMs = submit();
        begin();
        allBarrier();
        command_.resetQueryPool(*queries_, 0, 4);
        stamp(0);
        if (n_ > 1) {
            sort(work_[0]); computeBarrier();
            dispatch(2, n_); computeBarrier();
            sort(work_[2]); computeBarrier();
        }
        if (n_) { dispatch(3, n_); computeBarrier(); }
        stamp(1);
        dispatch(direct ? 1 : 0, n_ + 1);
        computeBarrier();
        stamp(2);
        dispatch(4, input_.header.source.tileCount);
        computeBarrier();
        stamp(3);
        out.submitWaitMs = submit();
        std::array<uint64_t, 4> ticks{};
        check(device_.getQueryPoolResults(*queries_, 0, 4, sizeof(ticks), ticks.data(), sizeof(uint64_t),
                  vk::QueryResultFlagBits::e64) == vk::Result::eSuccess, "Range timestamps unavailable");
        auto interval = [&](size_t a, size_t b) {
            auto value = schedulingTimestampInterval(ticks[a], ticks[b], bits_, period_);
            check(value.has_value(), "Invalid range timestamp interval");
            return *value;
        };
        out.sequenceMs = interval(0, 3); out.rangeMs = interval(1, 3); out.boundaryMs = interval(1, 2);
        return out;
    }
    void verify() {
        const std::array<const std::vector<uint32_t>*, 4> expected{
            &expected_.low, &expected_.order, &expected_.high, &expected_.items};
        for (size_t i = 0; i < work_.size(); ++i) {
            const auto values = download(work_[i], bytes_ / 4);
            if (i == 2 && n_ <= 1) {
                check(std::all_of(values.begin(), values.end(), [](auto v) { return v == canary; }),
                      "Zero/one item unexpectedly produced sorted scratch");
            } else {
                check(std::equal(expected[i]->begin(), expected[i]->end(), values.begin()),
                      "GPU sorted key/permutation/item relation differs from CPU oracle");
                check(std::all_of(values.begin() + n_, values.end(), [](auto v) { return v == canary; }),
                      "Range producer tail/canary changed");
            }
        }
        const auto result = download(ranges_, rangeBytes_ / 4 + guardWords);
        check(std::memcmp(result.data(), expected_.ranges.data(), rangeBytes_) == 0,
              "GPU range output differs from exact CPU oracle");
        check(std::all_of(result.begin() + rangeBytes_ / 4, result.end(), [](auto v) { return v == canary; }),
              "Range output canary changed");
        std::array<uint32_t, guardWords> guard{};
        transfer_.download(storage_.getDescriptorInfo(), scratchBytes_, std::as_writable_bytes(std::span(guard)));
        check(std::all_of(guard.begin(), guard.end(), [](auto v) { return v == canary; }), "Radix scratch canary changed");
        const std::array<const std::vector<uint32_t>*, 4> original{
            &input_.depthKeys, &input_.indices, &input_.tileKeys, &input_.items};
        for (size_t i = 0; i < seeds_.size(); ++i) {
            const auto values = download(seeds_[i], bytes_ / 4);
            check(std::equal(original[i]->begin(), original[i]->end(), values.begin()) &&
                      std::all_of(values.begin() + n_, values.end(), [](auto v) { return v == canary; }),
                  "Readonly source/canary changed");
        }
    }
private:
    void begin() {
        command_.reset();
        command_.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    }
    double submit() {
        command_.end(); device_.resetFences(*fence_);
        vk::SubmitInfo info; info.setCommandBuffers(command_);
        const auto start = std::chrono::steady_clock::now();
        queue_.submit(info, *fence_);
        check(device_.waitForFences(*fence_, true, UINT64_MAX) == vk::Result::eSuccess, "Range fence wait failed");
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    void allBarrier() {
        vk::MemoryBarrier memory{vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
                                 vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite};
        command_.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands,
                                 {}, memory, {}, {});
    }
    void computeBarrier() {
        vk::MemoryBarrier memory{vk::AccessFlagBits::eShaderWrite,
                                 vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite};
        command_.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                                 {}, memory, {}, {});
    }
    void sort(Buffer& keys) {
        const auto reuse = trainingSortScratchReuseBarrier(storage_.getBuffer());
        command_.pipelineBarrier(kTrainingSortScratchReuseSourceStages, kTrainingSortScratchReuseDestinationStages,
                                 {}, 0, nullptr, 1, &reuse, 0, nullptr);
        vrdxCmdSortKeyValue(command_, sorter_.get(), n_, keys.getBuffer(), 0, work_[1].getBuffer(), 0,
                            storage_.getBuffer(), 0, VK_NULL_HANDLE, 0);
    }
    void dispatch(size_t i, uint32_t elements) {
        command_.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_[i].getPipeline());
        command_.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelines_[i].getPipelineLayout(), 0, sets_[i], {});
        command_.pushConstants(pipelines_[i].getPipelineLayout(), vk::ShaderStageFlagBits::eCompute,
                               0, sizeof(push_), &push_);
        command_.dispatch((elements + 255u) / 256u, 1, 1);
    }
    void stamp(uint32_t point) {
        command_.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, *queries_, point);
    }
    std::vector<uint32_t> download(Buffer& buffer, size_t words) {
        std::vector<uint32_t> values(words);
        downloadSortReplay(transfer_, buffer.getDescriptorInfo(), std::as_writable_bytes(std::span(values)));
        return values;
    }
    vk::Device device_; vk::PhysicalDevice physical_; vk::Queue queue_; uint32_t family_;
    const Inputs& input_; Oracle expected_;
    uint32_t n_ = 0, capacity_ = 0, bits_ = 0;
    uint64_t bytes_ = 0, rangeBytes_ = 0, scratchBytes_ = 0;
    double period_ = 0;
    OptimizerCaptureTransfer transfer_;
    std::unique_ptr<std::remove_pointer_t<VrdxSorter>, decltype(&vrdxDestroySorter)> sorter_{nullptr, &vrdxDestroySorter};
    std::array<Buffer, 4> seeds_, work_;
    Buffer storage_, ranges_;
    std::array<ComputePipeline, 5> pipelines_;
    BackwardDescriptorPool descriptors_;
    std::array<vk::DescriptorSet, 5> sets_{};
    TrainingPushConstants push_;
    CommandPool pool_; vk::CommandBuffer command_;
    vk::UniqueFence fence_; vk::UniqueQueryPool queries_;
};
} // namespace range_keys
