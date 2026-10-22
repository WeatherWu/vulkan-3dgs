#pragma once

#include "training/core/training_types.hpp"

#include <algorithm>
#include <cstdint>

namespace vulkan3DGS {

inline constexpr uint32_t kTrainingTileSize8 = 8u;
inline constexpr uint32_t kTrainingTileSize16 = 16u;
struct TrainingTileLayout {
    uint32_t tileSize = kTrainingTileSize16;
    uint32_t tileCountX = 0u;
    uint32_t tileCountY = 0u;
    uint32_t tileCount = 0u;
    uint32_t pixelCountPerTile = kTrainingTileSize16 * kTrainingTileSize16;
};

[[nodiscard]] constexpr uint32_t trainingTileSize(TrainingTileMode mode) noexcept {
    return mode == TrainingTileMode::Fixed8 ? kTrainingTileSize8 : kTrainingTileSize16;
}

[[nodiscard]] constexpr TrainingTileLayout
makeTrainingTileLayout(TrainingExtent extent, uint32_t requestedTileSize) noexcept {
    const uint32_t tileSize =
        requestedTileSize == kTrainingTileSize8 ? kTrainingTileSize8 : kTrainingTileSize16;
    const uint32_t width = std::max(extent.width, 1u);
    const uint32_t height = std::max(extent.height, 1u);
    const uint32_t countX = (width + tileSize - 1u) / tileSize;
    const uint32_t countY = (height + tileSize - 1u) / tileSize;
    return TrainingTileLayout{
        .tileSize = tileSize,
        .tileCountX = countX,
        .tileCountY = countY,
        .tileCount = countX * countY,
        .pixelCountPerTile = tileSize * tileSize,
    };
}

[[nodiscard]] constexpr TrainingTileLayout makeTrainingTileLayout(TrainingExtent extent,
                                                                  TrainingTileMode mode) noexcept {
    return makeTrainingTileLayout(extent, trainingTileSize(mode));
}

} // namespace vulkan3DGS
