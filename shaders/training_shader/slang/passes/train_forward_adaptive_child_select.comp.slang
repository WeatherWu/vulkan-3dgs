#include "../common/train_forward_tile_common.slang"

[[vk::binding(5, 0)]]
StructuredBuffer<uint4> parentRanges;

[[vk::binding(9, 0)]]
RWStructuredBuffer<uint4> counters;

[[vk::binding(34, 0)]]
RWStructuredBuffer<uint> adaptiveTileSplitFlags;

[[vk::binding(35, 0)]]
StructuredBuffer<uint4> childCounts;

[[vk::binding(37, 0)]]
RWStructuredBuffer<uint4> childOffsets;

[[vk::binding(38, 0)]]
StructuredBuffer<uint> childPrefix;

[[vk::binding(39, 0)]]
StructuredBuffer<uint> childBlockOffsets;

[[vk::binding(40, 0)]]
RWStructuredBuffer<uint> activeParents;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint parentIndex = dispatchThreadID.x;
    uint parentCount = tileCountX() * tileCountY();
    if (parentIndex >= parentCount) return;

    uint childBudget = params.tileItemCount / 4u;
    uint4 counts = childCounts[parentIndex];
    uint childItems = trainAdaptiveChildItemCount(counts);
    uint parentItems = parentRanges[parentIndex * 4u].y;
    bool profitable = trainAdaptiveRefinementIsProfitable(parentItems, counts);

    uint blockOffset = childBlockOffsets[parentIndex / 256u];
    uint localOffset = childPrefix[parentIndex];
    uint globalOffset = childBudget + 1u;
    if (blockOffset <= childBudget && localOffset <= childBudget - blockOffset) {
        globalOffset = blockOffset + localOffset;
    }

    bool fits = globalOffset <= childBudget && childItems <= childBudget - globalOffset;
    if (profitable && fits) {
        childOffsets[parentIndex] = uint4(globalOffset,
                                           globalOffset + counts.x,
                                           globalOffset + counts.x + counts.y,
                                           globalOffset + counts.x + counts.y + counts.z);
        adaptiveTileSplitFlags[parentIndex] = 1u;
        // The runtime count is needed even on steps without validation. The
        // append order only schedules independent leaves; their depth order
        // continues to come from the unchanged child lists.
        uint activeIndex = 0u;
        InterlockedAdd(counters[0].z, 1u, activeIndex);
        activeParents[activeIndex] = parentIndex;
        if (params.validationEnabled != 0u) {
            uint ignored = 0u;
            InterlockedAdd(counters[0].y, childItems, ignored);
        }
    } else {
        childOffsets[parentIndex] = uint4(0u);
        adaptiveTileSplitFlags[parentIndex] = 0u;
    }
}
