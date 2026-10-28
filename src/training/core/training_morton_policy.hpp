#pragma once

#include <cstdint>

namespace vulkan3DGS {

struct TrainingMortonStats {
    uint32_t reorderCount = 0;
    uint32_t lastCompletedSteps = 0;
    uint32_t lastGaussianCount = 0;
    float lastCpuMs = 0;
    float lastGpuMs = 0;
    uint64_t peakWorkspaceBytes = 0;
    bool gpuTimed = false;
};

constexpr bool trainingMortonDue(bool enabled, bool fixedBenchmark, uint32_t completedSteps,
                                 const TrainingMortonStats& stats) noexcept {
    return enabled && !fixedBenchmark && completedSteps <= 15000u && completedSteps % 5000u == 0u &&
           (!stats.reorderCount || stats.lastCompletedSteps != completedSteps);
}

} // namespace vulkan3DGS
