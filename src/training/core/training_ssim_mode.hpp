#pragma once

#include <cstdint>

namespace vulkan3DGS {

// Persisted values and gradient-export IDs are explicit; supported modes are 0 and 2.
enum class TrainingSsimMode : uint32_t {
    TwoDimensional = 0,
    BackwardSeparable = 2,
};

constexpr bool validTrainingSsimMode(TrainingSsimMode mode) {
    return mode == TrainingSsimMode::TwoDimensional || mode == TrainingSsimMode::BackwardSeparable;
}

constexpr bool ssimUsesSeparableBackward(TrainingSsimMode mode) {
    return mode == TrainingSsimMode::BackwardSeparable;
}

constexpr uint32_t ssimScratchPlaneCount(TrainingSsimMode mode) {
    return ssimUsesSeparableBackward(mode) ? 3u : 0u;
}

constexpr const char* trainingSsimModeName(TrainingSsimMode mode) {
    switch (mode) {
        case TrainingSsimMode::TwoDimensional:
            return "2D";
        case TrainingSsimMode::BackwardSeparable:
            return "2D Loss + Separable Backward";
    }
    return "Invalid";
}

constexpr const char* trainingSsimModeFileTag(TrainingSsimMode mode) {
    switch (mode) {
        case TrainingSsimMode::TwoDimensional:
            return "ssim_2d";
        case TrainingSsimMode::BackwardSeparable:
            return "ssim_backward_separable";
    }
    return "ssim_invalid";
}

} // namespace vulkan3DGS
