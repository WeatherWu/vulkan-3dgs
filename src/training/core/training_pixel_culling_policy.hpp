#pragma once

#include "training/core/training_types.hpp"

namespace vulkan3DGS {

enum class TrainingPixelCullStatus {
    Off,
    Active,
    RendererNotReady,
    RequiresFixed16,
    RequiresWorkgroupForward,
    RequiresTensor,
    RequiresSubgroupGlobal,
    RequiresBaselineRgb,
    PipelineUnavailable,
};

struct TrainingPixelCullContext {
    bool rendererReady = false;
    bool pipelineAvailable = false;
    TrainingTileMode tileMode = TrainingTileMode::Fixed16;
    TrainingForwardCompositeMode forwardMode = TrainingForwardCompositeMode::WorkgroupShared;
    TrainingPixelTo2DGSMode pixelMode = TrainingPixelTo2DGSMode::Direct;
    bool globalPixelGrad = false;
    bool rgbLatencyHiding = false;
};

[[nodiscard]] constexpr TrainingPixelCullStatus
resolveTrainingPixelCull(bool requested, const TrainingPixelCullContext& context) noexcept {
    if (!requested) return TrainingPixelCullStatus::Off;
    if (!context.rendererReady) return TrainingPixelCullStatus::RendererNotReady;
    if (context.tileMode != TrainingTileMode::Fixed16)
        return TrainingPixelCullStatus::RequiresFixed16;
    if (context.forwardMode != TrainingForwardCompositeMode::WorkgroupShared)
        return TrainingPixelCullStatus::RequiresWorkgroupForward;
    if (context.pixelMode != TrainingPixelTo2DGSMode::VkSplatTensor)
        return TrainingPixelCullStatus::RequiresTensor;
    if (!context.globalPixelGrad) return TrainingPixelCullStatus::RequiresSubgroupGlobal;
    if (context.rgbLatencyHiding) return TrainingPixelCullStatus::RequiresBaselineRgb;
    if (!context.pipelineAvailable) return TrainingPixelCullStatus::PipelineUnavailable;
    return TrainingPixelCullStatus::Active;
}

[[nodiscard]] constexpr const char*
trainingPixelCullStatusName(TrainingPixelCullStatus status) noexcept {
    switch (status) {
        case TrainingPixelCullStatus::Off:
            return "Off";
        case TrainingPixelCullStatus::Active:
            return "Active (guarded FP32 v1)";
        case TrainingPixelCullStatus::RendererNotReady:
            return "Pending: renderer not initialized";
        case TrainingPixelCullStatus::RequiresFixed16:
            return "Fallback: requires Fixed16";
        case TrainingPixelCullStatus::RequiresWorkgroupForward:
            return "Fallback: requires Workgroup Shared forward";
        case TrainingPixelCullStatus::RequiresTensor:
            return "Fallback: requires active VkSplat Tensor";
        case TrainingPixelCullStatus::RequiresSubgroupGlobal:
            return "Fallback: requires Tensor Subgroup + Global";
        case TrainingPixelCullStatus::RequiresBaselineRgb:
            return "Fallback: disable RGB Latency Hiding";
        case TrainingPixelCullStatus::PipelineUnavailable:
            return "Fallback: EarlyCull pipeline unavailable (native subgroup32 / resources / "
                   "setup)";
    }
    return "Fallback: unknown EarlyCull status";
}

// Benchmark buttons deliberately override the persisted NORMAL-training
// request: ordinary/P1/P2 remain Off; only the S3 command requests method 1.
[[nodiscard]] constexpr bool
trainingPixelCullForStep(bool fixedBenchmark, bool benchmarkRequest,
                         TrainingPixelCullStatus trainingStatus) noexcept {
    return fixedBenchmark ? benchmarkRequest : trainingStatus == TrainingPixelCullStatus::Active;
}

struct TrainingPixelCullRunStats {
    uint64_t completedSteps = 0;
    uint64_t activeSteps = 0;
    uint64_t fallbackSteps = 0;

    void record(TrainingPixelCullStatus status) noexcept {
        ++completedSteps;
        if (status == TrainingPixelCullStatus::Active)
            ++activeSteps;
        else if (status != TrainingPixelCullStatus::Off)
            ++fallbackSteps;
    }
};

} // namespace vulkan3DGS
