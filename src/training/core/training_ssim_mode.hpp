#pragma once

#include <cstdint>

namespace vulkan3DGS {

// Values 0/1 retain the meaning of v6 exports and the old separable_ssim flag.
enum class TrainingSsimMode : uint32_t {
    TwoDimensional = 0,
    Separable = 1,
    BackwardSeparable = 2,
};

constexpr bool validTrainingSsimMode(TrainingSsimMode mode) {
    return mode == TrainingSsimMode::TwoDimensional || mode == TrainingSsimMode::Separable ||
           mode == TrainingSsimMode::BackwardSeparable;
}

constexpr bool ssimUsesSeparableLoss(TrainingSsimMode mode) {
    return mode == TrainingSsimMode::Separable;
}

constexpr bool ssimUsesSeparableBackward(TrainingSsimMode mode) {
    return mode == TrainingSsimMode::Separable || mode == TrainingSsimMode::BackwardSeparable;
}

constexpr uint32_t ssimScratchPlaneCount(TrainingSsimMode mode) {
    return ssimUsesSeparableLoss(mode) ? 5u : ssimUsesSeparableBackward(mode) ? 3u : 0u;
}

constexpr const char* trainingSsimModeName(TrainingSsimMode mode) {
    switch (mode) {
        case TrainingSsimMode::TwoDimensional:
            return "2D";
        case TrainingSsimMode::Separable:
            return "Separable";
        case TrainingSsimMode::BackwardSeparable:
            return "2D Loss + Separable Backward";
    }
    return "Invalid";
}

constexpr const char* trainingSsimModeFileTag(TrainingSsimMode mode) {
    switch (mode) {
        case TrainingSsimMode::TwoDimensional:
            return "ssim_2d";
        case TrainingSsimMode::Separable:
            return "ssim_separable";
        case TrainingSsimMode::BackwardSeparable:
            return "ssim_backward_separable";
    }
    return "ssim_invalid";
}

} // namespace vulkan3DGS
