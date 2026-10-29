#pragma once

#include <cstdint>

namespace vulkan3DGS {
enum class TrainingProjectionFallback : uint32_t { None, MissingIntrinsics };
struct TrainingProjectionState {
    bool executed = false;
    bool requested = false;
    bool active = false;
    TrainingProjectionFallback fallback = TrainingProjectionFallback::None;
};
enum class TrainingKey16Fallback : uint32_t {
    None,
    FasterSortInactive,
    TileMode,
    DeviceFeature,
    TileCount
};
struct TrainingKey16State {
    bool requested = false;
    bool active = false;
    TrainingKey16Fallback fallback = TrainingKey16Fallback::None;
    uint64_t logicalKeyBytes = 0;
    uint64_t allocatedBytes = 0;
};
constexpr const char* trainingKey16FallbackName(TrainingKey16Fallback reason) noexcept {
    switch (reason) {
        case TrainingKey16Fallback::FasterSortInactive:
            return "Faster-GS sorting is inactive";
        case TrainingKey16Fallback::TileMode:
            return "requires effective Fixed16 tiles";
        case TrainingKey16Fallback::DeviceFeature:
            return "16-bit storage/int16 not enabled on device";
        case TrainingKey16Fallback::TileCount:
            return "tile count exceeds 65536";
        default:
            return "none";
    }
}
constexpr TrainingKey16State resolveTrainingKey16(bool requested, bool faster, uint32_t tileSize,
                                                  bool enabledFeature,
                                                  uint64_t tileCount) noexcept {
    TrainingKey16State state;
    state.requested = requested;
    if (!requested) return state;
    if (!faster)
        state.fallback = TrainingKey16Fallback::FasterSortInactive;
    else if (tileSize != 16)
        state.fallback = TrainingKey16Fallback::TileMode;
    else if (!enabledFeature)
        state.fallback = TrainingKey16Fallback::DeviceFeature;
    else if (tileCount > 65536)
        state.fallback = TrainingKey16Fallback::TileCount;
    else
        state.active = true;
    return state;
}
} // namespace vulkan3DGS
