#pragma once

#include "training_sort_replay_reference.hpp"
#include "training/core/gaussian_forward_renderer.hpp"
#include "training/core/training_optimizer_capture_gpu.hpp"
#include "training/core/training_scheduling_diagnostics.hpp"

#include <chrono>
#include <span>

namespace sort_replay {
struct Sample {
    std::array<std::optional<double>, kTrainingSortTimestampCount> cumulative;
    double restoreCpuMs = 0, prepareCpuMs = 0, countReadCpuMs = 0, mainCpuMs = 0;
};
class BaselineSession {
public:
    BaselineSession(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue, uint32_t family)
        : device_(device), physical_(physical), queue_(queue), family_(family),
          transfer_(device, physical, queue, family) {}
    ~BaselineSession() {
        try { device_.waitIdle(); } catch (...) {}
        renderer_.cleanup();
        buffers_.cleanup();
    }
    void initialize(const TrainingScanlineCapture& source) {
        header_ = source.header;
        buffers_.initialize(device_, physical_, queue_, family_, family_);
        buffers_.resize(header_.gaussianCount, {header_.width, header_.height});
        if (buffers_.tileItemCapacity() < header_.itemCount)
            buffers_.resizeTileItems(header_.itemCount);
        renderer_.initialize(device_, physical_, queue_, family_, header_.gaussianCount,
                             {header_.width, header_.height});
        pool_.create(device_, family_, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        command_ = pool_.allocateCommandBuffer();
        fence_ = device_.createFenceUnique({});
        queries_ = device_.createQueryPoolUnique({{}, vk::QueryType::eTimestamp,
                                                  kTrainingSortTimestampCount});
        transfer_.upload(buffers_.projectedInfo(), 0,
                          std::as_bytes(std::span(source.projected)));
        push_.gaussianCount = header_.gaussianCount;
        push_.pixelCount = header_.width * header_.height;
        push_.width = header_.width;
        push_.height = header_.height;
        push_.optimizerEnabled = 0;
        push_.tileSize = 16;
    }
    uint32_t capacity() const { return buffers_.tileItemCapacity(); }
    Inputs captureEmit() {
        restore();
        prepare(false);
        begin();
        renderer_.setTrainingBuffers(buffers_, command_, push_);
        renderer_.emitPreparedTileItems(header_.itemCount);
        submit();
        Inputs result;
        result.header.source = header_;
        result.header.capacity = capacity();
        result.projected = download<ProjectedGaussian>(buffers_.projectedInfo(), header_.gaussianCount);
        result.gaussianRanges = download<glm::uvec2>(buffers_.gaussianTileRangesInfo(), header_.gaussianCount);
        result.items = download<uint32_t>(buffers_.tileItemsUnsortedInfo(), header_.itemCount);
        result.depthKeys = download<uint32_t>(buffers_.tileKeyLowInfo(), header_.itemCount);
        result.tileKeys = download<uint32_t>(buffers_.tileKeyHighInfo(), header_.itemCount);
        result.indices = download<uint32_t>(buffers_.tileSortIndicesInfo(), header_.itemCount);
        return result;
    }
    Sample run(bool segmented) {
        Sample sample;
        sample.restoreCpuMs = restore();
        renderer_.setSortDiagnosticQueryPool(segmented ? *queries_ : vk::QueryPool{});
        begin();
        command_.resetQueryPool(*queries_, 0, kTrainingSortTimestampCount);
        if (!segmented) stamp(TrainingSortTimestamp::CountBegin);
        renderer_.setTrainingBuffers(buffers_, command_, push_);
        renderer_.prepareProjectedTileItems();
        if (!segmented) stamp(TrainingSortTimestamp::PrefixEnd);
        buffers_.recordTileItemCountReadback(command_);
        sample.prepareCpuMs = submit();
        const auto readStart = Clock::now();
        check(buffers_.requiredTileItemCount() == header_.itemCount,
              "GPU count differs from captured I; no timing admission");
        sample.countReadCpuMs = milliseconds(readStart);
        begin();
        renderer_.setTrainingBuffers(buffers_, command_, push_);
        if (!segmented) stamp(TrainingSortTimestamp::EmitBegin);
        renderer_.buildPreparedTileLists(header_.itemCount);
        if (!segmented) stamp(TrainingSortTimestamp::RangesEnd);
        sample.mainCpuMs = submit();
        struct Query { uint64_t ticks, available; };
        std::array<Query, kTrainingSortTimestampCount> values{};
        const auto result = vkGetQueryPoolResults(device_, *queries_, 0, kTrainingSortTimestampCount,
            sizeof(values), values.data(), sizeof(Query),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        check(result == VK_SUCCESS || result == VK_NOT_READY, "Timestamp read failed");
        const auto bits = physical_.getQueueFamilyProperties()[family_].timestampValidBits;
        for (size_t i = 0; i < values.size(); ++i)
            if (values[0].available && values[i].available)
                sample.cumulative[i] = schedulingTimestampInterval(values[0].ticks, values[i].ticks,
                    bits, physical_.getProperties().limits.timestampPeriod);
        return sample;
    }
    void verifySorted(const TrainingScanlineCapture& source) {
        const auto actual = download<uint32_t>(buffers_.tileItemsInfo(), header_.itemCount);
        const auto ranges = download<glm::uvec4>(buffers_.tileRangesInfo(), header_.tileCount);
        const auto projected = download<ProjectedGaussian>(buffers_.projectedInfo(), header_.gaussianCount);
        check(actual == source.items &&
                  std::memcmp(ranges.data(), source.ranges.data(), ranges.size() * sizeof(glm::uvec4)) == 0,
              "GPU list/ranges disagree with original capture");
        check(std::memcmp(projected.data(), source.projected.data(), projected.size() * sizeof(ProjectedGaussian)) == 0,
              "Readonly projected input changed");
    }
private:
    using Clock = std::chrono::steady_clock;
    static double milliseconds(Clock::time_point start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
    void begin() {
        command_.reset();
        command_.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        vk::MemoryBarrier barrier;
        barrier.setSrcAccessMask(vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite)
            .setDstAccessMask(vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite);
        command_.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                vk::PipelineStageFlagBits::eAllCommands, {}, barrier, {}, {});
    }
    double submit() {
        command_.end();
        device_.resetFences(*fence_);
        vk::SubmitInfo info;
        info.setCommandBuffers(command_);
        const auto start = Clock::now();
        queue_.submit(info, *fence_);
        check(device_.waitForFences(*fence_, true, UINT64_MAX) == vk::Result::eSuccess,
              "Replay fence wait failed");
        return milliseconds(start);
    }
    void stamp(TrainingSortTimestamp point) {
        command_.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, *queries_, uint32_t(point));
    }
    double restore() {
        renderer_.setSortDiagnosticQueryPool(nullptr);
        begin();
        for (const auto& info : {buffers_.densificationStatesInfo(), buffers_.gaussianTileRangesInfo(),
                 buffers_.gaussianTilePrefixScratchInfo(), buffers_.countersInfo(), buffers_.tileRangesInfo(),
                 buffers_.tileItemsUnsortedInfo(), buffers_.tileItemsSortedInfo(), buffers_.tileKeyLowInfo(),
                 buffers_.tileKeyHighInfo(), buffers_.tileSortIndicesInfo(), buffers_.tileSortScratchInfo()})
            command_.fillBuffer(info.buffer, info.offset, info.range, 0u);
        // Internal radix scratch is initialized by the sorter on every invocation.
        return submit();
    }
    void prepare(bool segmented) {
        renderer_.setSortDiagnosticQueryPool(segmented ? *queries_ : vk::QueryPool{});
        begin();
        renderer_.setTrainingBuffers(buffers_, command_, push_);
        renderer_.prepareProjectedTileItems();
        buffers_.recordTileItemCountReadback(command_);
        submit();
        check(buffers_.requiredTileItemCount() == header_.itemCount, "GPU count does not match capture");
    }
    template<class T> std::vector<T> download(vk::DescriptorBufferInfo source, size_t count) {
        std::vector<T> values(count);
        if (count) transfer_.download(source, 0, std::as_writable_bytes(std::span(values)));
        return values;
    }
    vk::Device device_;
    vk::PhysicalDevice physical_;
    vk::Queue queue_;
    uint32_t family_;
    TrainingScanlineHeader header_;
    TrainingPushConstants push_;
    TrainingBuffers buffers_;
    GaussianForwardRenderer renderer_;
    OptimizerCaptureTransfer transfer_;
    CommandPool pool_;
    vk::CommandBuffer command_;
    vk::UniqueFence fence_;
    vk::UniqueQueryPool queries_;
};
} // namespace sort_replay
