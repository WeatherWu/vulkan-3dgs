#include "training/core/training_profiling_service.hpp"

#include "utils/logger.hpp"

#include <array>
#include <vector>

namespace vulkan3DGS {

TrainingProfilingService::~TrainingProfilingService() {
    cleanup();
}

void TrainingProfilingService::initialize(vk::Device device, vk::PhysicalDevice physicalDevice,
                                          uint32_t computeQueueFamilyIndex) {
    cleanup();
    device_ = device;
    stats_ = {};

    const std::vector<vk::QueueFamilyProperties> families =
        physicalDevice.getQueueFamilyProperties();
    if (computeQueueFamilyIndex >= families.size() ||
        families[computeQueueFamilyIndex].timestampValidBits == 0u) {
        LOG_WARN("Training GPU timestamps are unavailable on the selected compute queue; CPU "
                 "timings remain enabled");
        return;
    }

    timestampPeriodNanoseconds_ = physicalDevice.getProperties().limits.timestampPeriod;
    if (timestampPeriodNanoseconds_ <= 0.0f) {
        LOG_WARN("Training GPU timestamps report an invalid timestamp period; CPU timings remain "
                 "enabled");
        return;
    }

    vk::QueryPoolCreateInfo createInfo{};
    createInfo.setQueryType(vk::QueryType::eTimestamp)
        .setQueryCount(kTrainingGpuTimestampQueryCount);
    queryPool_ = device_.createQueryPool(createInfo);
    gpuTimestampsAvailable_ = true;
    stats_.gpuTimestampsAvailable = true;
    LOG_INFO("Training profiling enabled: {} GPU timestamp stages, timestamp period {} ns",
             kTrainingGpuProfileStageCount, timestampPeriodNanoseconds_);
}

void TrainingProfilingService::cleanup() {
    gpuTimestampsAvailable_ = false;
    timestampPeriodNanoseconds_ = 0.0f;
    if (queryPool_ && device_) device_.destroyQueryPool(queryPool_);
    queryPool_ = nullptr;
    device_ = nullptr;
    stats_ = {};
}

void TrainingProfilingService::resetLastSamples() {
    for (TrainingTiming& timing : stats_.cpu)
        timing.lastMs = 0.0f;
    for (TrainingTiming& timing : stats_.gpu)
        timing.lastMs = 0.0f;
}

void TrainingProfilingService::resetStats() {
    const bool available = stats_.gpuTimestampsAvailable;
    stats_ = {};
    stats_.gpuTimestampsAvailable = available;
}

void TrainingProfilingService::recordCpu(TrainingCpuProfileStage stage, float milliseconds) {
    TrainingTiming& timing = stats_.cpu[static_cast<size_t>(stage)];
    timing.lastMs = milliseconds;
    ++timing.sampleCount;
    timing.averageMs += (milliseconds - timing.averageMs) / static_cast<float>(timing.sampleCount);
}

bool TrainingProfilingService::collectGpu() {
    if (!gpuTimestampsAvailable_) return false;

    struct TimestampQueryResult {
        uint64_t timestamp = 0;
        uint64_t available = 0;
    };
    std::array<TimestampQueryResult, kTrainingGpuTimestampQueryCount> timestamps{};
    const VkResult result =
        vkGetQueryPoolResults(static_cast<VkDevice>(device_), static_cast<VkQueryPool>(queryPool_),
                              0, kTrainingGpuTimestampQueryCount, sizeof(timestamps),
                              timestamps.data(), sizeof(TimestampQueryResult),
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (result != VK_SUCCESS && result != VK_NOT_READY) {
        LOG_WARN("Failed to read training GPU timestamps: {}. GPU timing has been disabled for "
                 "this session.",
                 static_cast<int>(result));
        gpuTimestampsAvailable_ = false;
        stats_.gpuTimestampsAvailable = false;
        return false;
    }

    static constexpr std::array<const char*, kTrainingGpuProfileStageCount> stageNames = {
        "Gaussian projection",
        "Tile coverage count",
        "Tile prefix",
        "Tile emit",
        "Tile sort and ranges",
        "Composite",
        "Loss",
        "Backward clear",
        "Loss to pixel",
        "Pixel to 2DGS",
        "Tile-local backward",
        "2DGS to 3DGS",
        "Fused projection/optimizer",
        "Optimizer",
        "Validation",
        "Densification",
    };
    for (uint32_t index = 0; index < kTrainingGpuProfileStageCount; ++index) {
        const auto stage = static_cast<TrainingGpuProfileStage>(index);
        const auto& begin = timestamps[trainingGpuTimestampQuery(stage, false)];
        const auto& end = timestamps[trainingGpuTimestampQuery(stage, true)];
        if (begin.available == 0u || end.available == 0u) continue;
        const float milliseconds = static_cast<float>(end.timestamp - begin.timestamp) *
                                   timestampPeriodNanoseconds_ / 1.0e6f;
        recordGpu(stage, milliseconds);
        LOG_DEBUG("Training GPU stage {}: {} ms", stageNames[index], milliseconds);
    }
    return true;
}

void TrainingProfilingService::recordGpu(TrainingGpuProfileStage stage, float milliseconds) {
    TrainingTiming& timing = stats_.gpu[static_cast<size_t>(stage)];
    timing.lastMs = milliseconds;
    ++timing.sampleCount;
    timing.averageMs += (milliseconds - timing.averageMs) / static_cast<float>(timing.sampleCount);
}

} // namespace vulkan3DGS
