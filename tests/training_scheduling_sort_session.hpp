#pragma once
#include "training/core/backward/backward_command_utils.hpp"
#include "training/core/training_optimizer_capture_gpu.hpp"
#include "training/core/training_scheduling_diagnostics.hpp"
#include "utils/system_memory.hpp"
#include "vulkan/compute_pipeline.hpp"
#include <vk_radix_sort.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <numeric>
#include <span>

namespace scheduling_test {
using namespace vulkan3DGS;
inline void check(bool okay, const char* reason) {
    if (!okay) throw std::runtime_error(reason);
}
constexpr uint32_t canary = 0xcdcdcdcd, guardWords = 16, tileCount = 64;
struct Sample {
    std::optional<double> workMs, restoreMs;
    double cpuSubmitWaitMs = 0;
};
// Synthetic two-32-bit stable sort + high-key gather + range construction.
// This does NOT bind live training buffers or implement the proposed scheduling.
class SortSession {
public:
    SortSession(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue, uint32_t family,
                uint32_t count, uint32_t capacity, bool allTies,
                const std::filesystem::path& shader)
        : device_(device), queue_(queue), n_(count), capacity_(capacity),
          transfer_(device, physical, queue, family) {
        check(capacity && capacity <= (1u << 22) && count <= capacity, "Invalid count/capacity");
        const auto limits = physical.getProperties().limits;
        check(limits.maxComputeWorkGroupInvocations >= 512 &&
                  limits.maxComputeWorkGroupSize[0] >= 512,
              "Sorter requires group512 support");
        VrdxSorterCreateInfo info{};
        info.device = device;
        info.physicalDevice = physical;
        VrdxSorter raw = nullptr;
        vrdxCreateSorter(&info, &raw);
        sorter_.reset(raw);
        VrdxSorterStorageRequirements storage{};
        vrdxGetSorterKeyValueStorageRequirements(sorter_.get(), capacity, &storage);
        const uint64_t bytes = (uint64_t(capacity) + guardWords) * sizeof(uint32_t);
        check(bytes <= limits.maxStorageBufferRange &&
                  storage.size + 64 <= limits.maxStorageBufferRange,
              "Sorter diagnostic storage range exceeded");
        check(availableSystemMemoryBytes() >= bytes * 8 + (128ull << 20),
              "Insufficient measured RAM for bounded synthetic arrays plus 128 MiB reserve");
        // Reuse the existing bounded private-diagnostic RAM/VRAM policy, no capture file.
        checkOptimizerCaptureBudget(physical, 0, bytes * 6 + storage.size + 1024);
        const auto usage = vk::BufferUsageFlagBits::eStorageBuffer |
                           vk::BufferUsageFlagBits::eTransferSrc |
                           vk::BufferUsageFlagBits::eTransferDst;
        auto create = [&](Buffer& b, uint64_t size, vk::BufferUsageFlags flags) {
            b.create(device, physical, queue, family, nullptr, size, flags,
                     vk::MemoryPropertyFlagBits::eDeviceLocal, {family});
        };
        for (auto& b : seeds_)
            create(b, bytes, usage);
        for (auto& b : work_)
            create(b, bytes, usage);
        create(control_, 64, usage);
        create(ranges_, (tileCount * 2 + guardWords) * 4, usage);
        create(scratch_, storage.size + 64,
               vk::BufferUsageFlags(storage.usage) | vk::BufferUsageFlagBits::eTransferDst |
                   vk::BufferUsageFlagBits::eTransferSrc);
        scratchBytes_ = storage.size;
        for (auto& v : input_)
            v.assign(capacity + guardWords, canary);
        uint32_t state = 0x12345678;
        for (uint32_t i = 0; i < count; ++i) {
            state = state * 1664525u + 1013904223u;
            input_[0][i] = allTies ? 42u : (state >> 8) & 1023u; // exact depth-bit keys
            input_[1][i] = i;
            input_[2][i] = allTies ? 7u : (state >> 20) & 63u; // original tile keys
        }
        for (size_t i = 0; i < seeds_.size(); ++i)
            upload(seeds_[i], input_[i]);
        expected_.resize(count);
        std::iota(expected_.begin(), expected_.end(), 0u);
        std::stable_sort(expected_.begin(), expected_.end(), [&](uint32_t a, uint32_t b) {
            return (uint64_t(input_[2][a]) << 32 | input_[0][a]) <
                   (uint64_t(input_[2][b]) << 32 | input_[0][b]);
        });
        expectedRanges_.assign(tileCount * 2 + guardWords, canary);
        std::fill_n(expectedRanges_.begin(), tileCount * 2, 0u);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t tile = input_[2][expected_[i]];
            if (!expectedRanges_[2 * tile + 1]) expectedRanges_[2 * tile] = i;
            ++expectedRanges_[2 * tile + 1];
        }
        ComputePipelineConfig config;
        for (uint32_t binding = 0; binding < 5; ++binding)
            config.descriptorBindings.push_back(backwardStorageBinding(binding));
        config.pushConstantSize = 8;
        pipeline_.initialize(device, shader.string(), config);
        set_ = descriptors_.create(device, 5, 0, {pipeline_.getDescriptorSetLayout()}).front();
        const std::array infos{seeds_[2].getDescriptorInfo(), work_[1].getDescriptorInfo(),
                               work_[2].getDescriptorInfo(), control_.getDescriptorInfo(),
                               ranges_.getDescriptorInfo()};
        std::array<vk::WriteDescriptorSet, 5> writes;
        for (uint32_t i = 0; i < 5; ++i)
            writes[i]
                .setDstSet(set_)
                .setDstBinding(i)
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                .setPBufferInfo(&infos[i]);
        device.updateDescriptorSets(writes, {});
        pool_.create(device, family, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        command_ = pool_.allocateCommandBuffer();
        fence_ = device.createFenceUnique({});
        bits_ = physical.getQueueFamilyProperties()[family].timestampValidBits;
        periodNs_ = limits.timestampPeriod;
        if (bits_) queries_ = device.createQueryPoolUnique({{}, vk::QueryType::eTimestamp, 4});
        // Count is produced by GPU transfer commands, not a host-mapped readback.
        // Synthetic input generation is outside the timed sort/gather/range stage.
        begin();
        command_.fillBuffer(control_.getBuffer(), 8, 56, canary);
        command_.fillBuffer(control_.getBuffer(), 0, 4, count);
        command_.fillBuffer(control_.getBuffer(), 4, 4, tileCount);
        barrier();
        submit();
    }
    Sample dispatch(bool indirect) {
        begin();
        barrier();
        if (queries_) {
            command_.resetQueryPool(*queries_, 0, 4);
            stamp(0);
        }
        for (uint32_t i = 0; i < 2; ++i)
            command_.copyBuffer(seeds_[i].getBuffer(), work_[i].getBuffer(),
                                vk::BufferCopy{0, 0, seeds_[i].getSize()});
        command_.fillBuffer(work_[2].getBuffer(), 0, work_[2].getSize(), canary);
        command_.fillBuffer(ranges_.getBuffer(), tileCount * 8, guardWords * 4, canary);
        command_.fillBuffer(ranges_.getBuffer(), 0, tileCount * 8, 0u);
        command_.fillBuffer(scratch_.getBuffer(), 0, scratch_.getSize(), canary);
        barrier();
        stamp(1);
        submit(); // Same separate seed restore/fence for both variants.
        begin();
        stamp(2);
        sort(work_[0], indirect);
        barrier();
        dispatchHelper(0, indirect ? capacity_ : n_);
        barrier();
        sort(work_[2], indirect);
        barrier();
        dispatchHelper(1, (indirect ? capacity_ : n_) + 1);
        barrier();
        dispatchHelper(2, tileCount);
        barrier();
        stamp(3);
        Sample sample;
        sample.cpuSubmitWaitMs = submit();
        if (queries_) {
            std::array<uint64_t, 4> ticks{};
            check(device_.getQueryPoolResults(*queries_, 0, 4, sizeof(ticks), ticks.data(),
                                              sizeof(uint64_t),
                                              vk::QueryResultFlagBits::e64) == vk::Result::eSuccess,
                  "Sorter timestamps unavailable after fence");
            sample.restoreMs = schedulingTimestampInterval(ticks[0], ticks[1], bits_, periodNs_);
            sample.workMs = schedulingTimestampInterval(ticks[2], ticks[3], bits_, periodNs_);
        }
        return sample;
    }
    void verify() {
        const auto indices = download(work_[1]);
        const auto high = download(work_[2]);
        const auto low = download(work_[0]);
        auto expectedLow = input_[0];
        std::sort(expectedLow.begin(), expectedLow.begin() + n_);
        for (uint32_t i = 0; i < n_; ++i)
            check(indices[i] == expected_[i] && high[i] == input_[2][expected_[i]] &&
                      low[i] == expectedLow[i],
                  "Two-pass stable sort differs from independent CPU oracle");
        for (uint32_t i = n_; i < capacity_ + guardWords; ++i)
            check(indices[i] == canary && high[i] == canary && low[i] == canary,
                  "Sort modified unused tail or canary");
        check(download(ranges_) == expectedRanges_,
              "Sorted tile ranges/canary differ from CPU oracle");
        auto count = download(control_);
        check(count[0] == n_ && count[1] == tileCount, "GPU count changed");
        for (size_t i = 2; i < count.size(); ++i)
            check(count[i] == canary, "Count canary modified");
        std::array<uint32_t, guardWords> scratchGuard{};
        transfer_.download(scratch_.getDescriptorInfo(), scratchBytes_,
                           std::as_writable_bytes(std::span(scratchGuard)));
        check(std::all_of(scratchGuard.begin(), scratchGuard.end(),
                          [](uint32_t x) { return x == canary; }),
              "Sorter scratch overrun");
        for (size_t i = 0; i < seeds_.size(); ++i)
            check(download(seeds_[i]) == input_[i], "Read-only input seed changed");
    }

private:
    void upload(Buffer& b, const std::vector<uint32_t>& values) {
        auto bytes = std::as_bytes(std::span(values));
        for (size_t offset = 0; offset < bytes.size();) {
            const auto part = bytes.subspan(
                offset, std::min<size_t>(optimizerCaptureChunkBytes, bytes.size() - offset));
            transfer_.upload(b.getDescriptorInfo(), offset, part);
            offset += part.size();
        }
    }
    std::vector<uint32_t> download(Buffer& b) {
        std::vector<uint32_t> values(size_t(b.getSize() / 4));
        auto bytes = std::as_writable_bytes(std::span(values));
        for (size_t offset = 0; offset < bytes.size();) {
            auto part = bytes.subspan(
                offset, std::min<size_t>(optimizerCaptureChunkBytes, bytes.size() - offset));
            transfer_.download(b.getDescriptorInfo(), offset, part);
            offset += part.size();
        }
        return values;
    }
    void begin() {
        command_.reset();
        command_.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    }
    double submit() {
        command_.end();
        device_.resetFences(*fence_);
        vk::SubmitInfo info;
        info.setCommandBuffers(command_);
        const auto start = std::chrono::steady_clock::now();
        queue_.submit(info, *fence_);
        (void)device_.waitForFences(*fence_, true, UINT64_MAX);
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
            .count();
    }
    void barrier() {
        vk::MemoryBarrier memory;
        memory
            .setSrcAccessMask(vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
                              vk::AccessFlagBits::eHostWrite)
            .setDstAccessMask(vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite);
        command_.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands |
                                     vk::PipelineStageFlagBits::eHost,
                                 vk::PipelineStageFlagBits::eAllCommands, {}, memory, {}, {});
    }
    void stamp(uint32_t i) {
        if (queries_)
            command_.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, *queries_, i);
    }
    void sort(Buffer& keys, bool indirect) {
        if (indirect)
            vrdxCmdSortKeyValueIndirect(command_, sorter_.get(), capacity_, control_.getBuffer(), 0,
                                        keys.getBuffer(), 0, work_[1].getBuffer(), 0,
                                        scratch_.getBuffer(), 0, VK_NULL_HANDLE, 0);
        else if (n_ >
                 1) // Same host zero/one bypass as production; never pass zero-sized descriptors.
            vrdxCmdSortKeyValue(command_, sorter_.get(), n_, keys.getBuffer(), 0,
                                work_[1].getBuffer(), 0, scratch_.getBuffer(), 0, VK_NULL_HANDLE,
                                0);
    }
    void dispatchHelper(uint32_t mode, uint32_t elements) {
        const std::array push{capacity_, mode};
        command_.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_.getPipeline());
        command_.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_.getPipelineLayout(),
                                    0, set_, {});
        command_.pushConstants(pipeline_.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0,
                               sizeof(push), push.data());
        if (elements) command_.dispatch((elements + 255u) / 256u, 1, 1);
    }
    vk::Device device_;
    vk::Queue queue_;
    uint32_t n_, capacity_, bits_ = 0;
    double periodNs_ = 0;
    OptimizerCaptureTransfer transfer_;
    std::unique_ptr<std::remove_pointer_t<VrdxSorter>, decltype(&vrdxDestroySorter)> sorter_{
        nullptr, &vrdxDestroySorter};
    std::array<Buffer, 3> seeds_, work_;
    Buffer control_, ranges_, scratch_;
    uint64_t scratchBytes_ = 0;
    std::array<std::vector<uint32_t>, 3> input_;
    std::vector<uint32_t> expected_, expectedRanges_;
    ComputePipeline pipeline_;
    BackwardDescriptorPool descriptors_;
    vk::DescriptorSet set_;
    CommandPool pool_;
    vk::CommandBuffer command_;
    vk::UniqueFence fence_;
    vk::UniqueQueryPool queries_;
};
} // namespace scheduling_test
