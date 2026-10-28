#pragma once

#include "training/core/training_types.hpp"
#include "training/core/training_scheduling_diagnostics.hpp"

#include <vulkan/vulkan.hpp>

namespace vulkan3DGS {

class TrainingProfilingService {
public:
    TrainingProfilingService() = default;
    ~TrainingProfilingService();

    TrainingProfilingService(const TrainingProfilingService&) = delete;
    TrainingProfilingService& operator=(const TrainingProfilingService&) = delete;

    void initialize(vk::Device device, vk::PhysicalDevice physicalDevice,
                    uint32_t computeQueueFamilyIndex,
                    std::optional<bool> vulkanValidationLayersEnabled = std::nullopt);
    void cleanup();

    void resetLastSamples();
    void resetStats();
    void recordCpu(TrainingCpuProfileStage stage, float milliseconds);
    void recordGpu(TrainingGpuProfileStage stage, float milliseconds);
    [[nodiscard]] bool collectGpu();

    [[nodiscard]] bool gpuTimestampsAvailable() const noexcept {
        return gpuTimestampsAvailable_;
    }
    [[nodiscard]] vk::QueryPool queryPool() const noexcept {
        return queryPool_;
    }
    [[nodiscard]] const TrainingProfilingStats& stats() const noexcept {
        return stats_;
    }
    TrainingSchedulingDiagnostics* schedulingDiagnostics() const noexcept {
        return scheduling_.get();
    }

private:
    vk::Device device_ = nullptr;
    vk::QueryPool queryPool_ = nullptr;
    float timestampPeriodNanoseconds_ = 0.0f;
    bool gpuTimestampsAvailable_ = false;
    TrainingProfilingStats stats_{};
    std::unique_ptr<TrainingSchedulingDiagnostics> scheduling_;
};

} // namespace vulkan3DGS
