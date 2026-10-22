#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace vulkan3DGS {

// Host ABI for CPU reference references and opt-in state construction GPU state diagnostics. Training
// gradients still use the original Tensor path. Forward-prefix and reverse-
// boundary layouts deliberately have separate task mappings.
constexpr uint32_t kCheckpointBucketSize = 32u;
constexpr uint32_t kCheckpointTileSize = 16u;
constexpr uint32_t kCheckpointTilePixels = 256u;
constexpr uint32_t kCheckpointBankCount = 4u;
constexpr uint32_t kCheckpointBankPixels = 64u;
constexpr uint32_t kCheckpointWarpsPerGroup = 4u;
constexpr uint32_t kCheckpointImplicitSlot = std::numeric_limits<uint32_t>::max();

enum class TrainingCheckpointLayoutVersion : uint32_t {
    ForwardPrefix = 1u,
    ReverseBoundary = 2u,
};

constexpr uint32_t checkpointBucketCount(uint32_t rangeCount) noexcept {
    return rangeCount / kCheckpointBucketSize + (rangeCount % kCheckpointBucketSize != 0u);
}

constexpr uint32_t checkpointStoredCount(uint32_t rangeCount) noexcept {
    return rangeCount == 0u ? 0u : (rangeCount - 1u) / kCheckpointBucketSize;
}

struct alignas(16) TrainingCheckpointTileMetadata {
    std::array<uint32_t, 4> range{};    // start, count, checkpoint base, stored count
    std::array<uint32_t, 4> activity{}; // max processed, tasks, task base, flags
};

struct alignas(16) TrainingCheckpointTask {
    uint32_t tileIndex = 0u;
    uint32_t bucketIndex = 0u;
    // Layout-version dependent: near edge for prefix, far edge for reverse.
    uint32_t checkpointSlot = kCheckpointImplicitSlot;
    uint32_t itemCount = 0u;
};

struct alignas(16) TrainingCheckpointControl {
    std::array<uint32_t, 4> capacity{};    // slots, tasks, tiles, layout version
    std::array<uint32_t, 4> totals{};      // slots, tasks, error flags, expected pixel states
    std::array<uint32_t, 4> dispatch{};    // indirect x/y/z, active task count
    std::array<uint32_t, 4> diagnostics{}; // written pixel states, reserved
    std::array<uint32_t, 4> fallback{};    // original Tensor indirect x/y/z, fallback flag
    std::array<uint32_t, 4> run{};         // cumulative fallbacks, flags, steps, observed warp size
};

struct TrainingCheckpointPrefixPush {
    uint32_t mode = 0, count = 0, input = 0, output = 0;
    uint32_t parent = 0, maxGroupsX = 0, maxGroupsY = 0, padding = 0;
};

constexpr uint32_t checkpointScratchElements(uint32_t tileCount) noexcept {
    uint32_t count = tileCount / 256u + (tileCount % 256u != 0u);
    uint32_t total = count;
    while (count > 256u) {
        count = count / 256u + (count % 256u != 0u);
        total += count;
    }
    return total;
}

struct TrainingCheckpointCapacity {
    uint64_t tileCount = 0u;
    uint64_t slotCapacity = 0u;
    uint64_t taskCapacity = 0u;
    uint64_t bankElements = 0u;
    uint64_t bankBytes = 0u;
    uint64_t metadataBytes = 0u;
    uint64_t taskBytes = 0u;
    bool shaderIndexSafe = false;

    constexpr bool fitsDescriptorRange(uint64_t maxStorageBufferRange) const noexcept {
        return shaderIndexSafe && bankBytes <= maxStorageBufferRange &&
               metadataBytes <= maxStorageBufferRange && taskBytes <= maxStorageBufferRange;
    }
};

constexpr TrainingCheckpointCapacity
makeTrainingCheckpointCapacity(uint32_t tileItemCount, uint32_t width, uint32_t height) noexcept {
    const uint64_t tilesX = (static_cast<uint64_t>(width) + 15u) / 16u;
    const uint64_t tilesY = (static_cast<uint64_t>(height) + 15u) / 16u;
    TrainingCheckpointCapacity result{};
    result.tileCount = tilesX * tilesY;
    result.slotCapacity = checkpointBucketCount(tileItemCount);
    result.taskCapacity = result.slotCapacity + result.tileCount;
    result.bankElements = result.slotCapacity * kCheckpointBankPixels;
    result.bankBytes = result.bankElements * 16u;
    result.metadataBytes = result.tileCount * sizeof(TrainingCheckpointTileMetadata);
    result.taskBytes = result.taskCapacity * sizeof(TrainingCheckpointTask);
    constexpr uint64_t limit = std::numeric_limits<uint32_t>::max();
    result.shaderIndexSafe = width != 0u && height != 0u && result.tileCount <= limit &&
                             result.taskCapacity <= limit && result.bankElements <= limit &&
                             static_cast<uint64_t>(width) * height <= limit;
    return result;
}

struct TrainingCheckpointAddress {
    uint32_t bank = 0u;
    uint32_t element = 0u;
};

constexpr std::optional<TrainingCheckpointTask>
makeTrainingCheckpointTask(uint32_t tileIndex, uint32_t rangeCount, uint32_t checkpointBase,
                           uint32_t bucketIndex, uint64_t slotCapacity) noexcept {
    if (bucketIndex >= checkpointBucketCount(rangeCount)) return std::nullopt;
    const uint32_t offset = bucketIndex * kCheckpointBucketSize;
    const uint32_t remaining = rangeCount - offset;
    const uint32_t count = remaining < kCheckpointBucketSize ? remaining : kCheckpointBucketSize;
    if (bucketIndex == 0u)
        return TrainingCheckpointTask{tileIndex, bucketIndex, kCheckpointImplicitSlot, count};
    const uint64_t slot = static_cast<uint64_t>(checkpointBase) + bucketIndex - 1u;
    if (slot >= slotCapacity || slot >= kCheckpointImplicitSlot) return std::nullopt;
    return TrainingCheckpointTask{tileIndex, bucketIndex, static_cast<uint32_t>(slot), count};
}

constexpr uint32_t reverseCheckpointBoundary(uint32_t rangeCount, uint32_t bucketIndex) noexcept {
    const uint64_t requested = (static_cast<uint64_t>(bucketIndex) + 1u) * kCheckpointBucketSize;
    return static_cast<uint32_t>(requested < rangeCount ? requested : rangeCount);
}

constexpr bool reverseCheckpointReadRequired(uint32_t processedCount, uint32_t bucketBegin,
                                             uint32_t boundary) noexcept {
    return processedCount > bucketBegin && processedCount > boundary;
}

constexpr bool reverseCheckpointMustBeWritten(uint32_t processedCount, uint32_t boundary) noexcept {
    return boundary != 0u && boundary < processedCount && boundary % kCheckpointBucketSize == 0u;
}

// Reverse traversal starts each bucket from the state at its far boundary.
// The final raw-list bucket uses the implicit (suffix=0, T=finalT) state.
constexpr std::optional<TrainingCheckpointTask>
makeReverseCheckpointTask(uint32_t tileIndex, uint32_t rangeCount, uint32_t checkpointBase,
                          uint32_t bucketIndex, uint64_t slotCapacity) noexcept {
    if (bucketIndex >= checkpointBucketCount(rangeCount)) return std::nullopt;
    const uint64_t begin64 = static_cast<uint64_t>(bucketIndex) * kCheckpointBucketSize;
    const uint32_t remaining = rangeCount - static_cast<uint32_t>(begin64);
    const uint32_t count = remaining < kCheckpointBucketSize ? remaining : kCheckpointBucketSize;
    if (bucketIndex + 1u == checkpointBucketCount(rangeCount))
        return TrainingCheckpointTask{tileIndex, bucketIndex, kCheckpointImplicitSlot, count};

    const uint64_t slot = static_cast<uint64_t>(checkpointBase) + bucketIndex;
    if (slot >= slotCapacity || slot >= kCheckpointImplicitSlot) return std::nullopt;
    return TrainingCheckpointTask{tileIndex, bucketIndex, static_cast<uint32_t>(slot), count};
}

constexpr std::optional<TrainingCheckpointAddress>
trainingCheckpointAddress(uint32_t slot, uint32_t pixelRank, uint64_t slotCapacity) noexcept {
    if (slot == kCheckpointImplicitSlot || slot >= slotCapacity ||
        pixelRank >= kCheckpointTilePixels)
        return std::nullopt;
    const uint64_t element =
        static_cast<uint64_t>(slot) * kCheckpointBankPixels + pixelRank % kCheckpointBankPixels;
    if (element > std::numeric_limits<uint32_t>::max()) return std::nullopt;
    return TrainingCheckpointAddress{pixelRank / kCheckpointBankPixels,
                                     static_cast<uint32_t>(element)};
}

static_assert(sizeof(TrainingCheckpointTileMetadata) == 32u);
static_assert(offsetof(TrainingCheckpointTileMetadata, activity) == 16u);
static_assert(sizeof(TrainingCheckpointTask) == 16u);
static_assert(sizeof(TrainingCheckpointControl) == 96u);
static_assert(offsetof(TrainingCheckpointControl, dispatch) == 32u);
static_assert(offsetof(TrainingCheckpointControl, fallback) == 64u);
static_assert(offsetof(TrainingCheckpointControl, run) == 80u);
static_assert(sizeof(TrainingCheckpointPrefixPush) == 32u);
static_assert(offsetof(TrainingCheckpointTask, itemCount) == 12u);

} // namespace vulkan3DGS
