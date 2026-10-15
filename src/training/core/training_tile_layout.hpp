#pragma once

#include "training/core/training_types.hpp"

#include <algorithm>
#include <cstdint>

namespace vulkan3DGS {

inline constexpr uint32_t kTrainingTileSize8 = 8u;
inline constexpr uint32_t kTrainingTileSize16 = 16u;
// Must match TRAIN_ADAPTIVE_CHILD_DISPATCH_WIDTH in train_forward_common.slang.
inline constexpr uint32_t kAdaptiveTrainingChildDispatchWidth = 65535u;

[[nodiscard]] constexpr uint32_t normalizeAdaptiveTensorChildBatch(uint32_t batch) noexcept {
    return batch == 16u || batch == 32u ? batch : 8u;
}
inline constexpr uint32_t kAdaptiveTrainingTileSplitFactor = 2u;
inline constexpr uint32_t kAdaptiveTrainingTileLeafFactor =
    kAdaptiveTrainingTileSplitFactor * kAdaptiveTrainingTileSplitFactor;

struct TrainingTileLayout {
    uint32_t tileSize = kTrainingTileSize16;
    uint32_t tileCountX = 0u;
    uint32_t tileCountY = 0u;
    uint32_t tileCount = 0u;
    uint32_t pixelCountPerTile = kTrainingTileSize16 * kTrainingTileSize16;
};

[[nodiscard]] constexpr uint32_t trainingAdaptiveParentIndex(uint32_t parentX, uint32_t parentY,
                                                             uint32_t parentCountX) noexcept {
    return parentY * parentCountX + parentX;
}

[[nodiscard]] constexpr uint32_t trainingAdaptiveLeafIndex(uint32_t parentX, uint32_t parentY,
                                                           uint32_t childX, uint32_t childY,
                                                           uint32_t parentCountX) noexcept {
    const uint32_t parentIndex = trainingAdaptiveParentIndex(parentX, parentY, parentCountX);
    const uint32_t childIndex = childY * kAdaptiveTrainingTileSplitFactor + childX;
    return parentIndex * kAdaptiveTrainingTileLeafFactor + childIndex;
}

[[nodiscard]] constexpr uint32_t trainingAdaptiveLeafIndexForPixel(uint32_t pixelX, uint32_t pixelY,
                                                                   uint32_t parentCountX,
                                                                   bool parentIsSplit) noexcept {
    const uint32_t parentX = pixelX / kTrainingTileSize16;
    const uint32_t parentY = pixelY / kTrainingTileSize16;
    const uint32_t parentLeaf = trainingAdaptiveLeafIndex(parentX, parentY, 0u, 0u, parentCountX);
    if (!parentIsSplit) return parentLeaf;

    const uint32_t childX = (pixelX % kTrainingTileSize16) / kTrainingTileSize8;
    const uint32_t childY = (pixelY % kTrainingTileSize16) / kTrainingTileSize8;
    return trainingAdaptiveLeafIndex(parentX, parentY, childX, childY, parentCountX);
}

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

// Leaf IDs reserve four slots for every 16x16 parent, including partial edge
// parents. A compact 8x8 grid can have fewer entries when an extent is not a
// multiple of 16, so it is not a safe adaptive range-buffer capacity.
[[nodiscard]] constexpr uint32_t trainingAdaptiveRangeCapacity(TrainingExtent extent) noexcept {
    return makeTrainingTileLayout(extent, kTrainingTileSize16).tileCount *
           kAdaptiveTrainingTileLeafFactor;
}

} // namespace vulkan3DGS
