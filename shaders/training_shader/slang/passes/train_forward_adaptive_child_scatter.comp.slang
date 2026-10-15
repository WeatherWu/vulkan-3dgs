#include "../common/train_forward_tile_common.slang"

[[vk::binding(5, 0)]]
StructuredBuffer<uint4> parentRanges;

[[vk::binding(24, 0)]]
RWStructuredBuffer<uint> sortedTileItems;

[[vk::binding(34, 0)]]
StructuredBuffer<uint> adaptiveTileSplitFlags;

[[vk::binding(36, 0)]]
StructuredBuffer<uint> childMasks;

[[vk::binding(37, 0)]]
StructuredBuffer<uint4> childOffsets;

groupshared uint4 sharedPrefix[256];
groupshared uint4 sharedWritten;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 groupID : SV_GroupID,
          uint3 groupThreadID : SV_GroupThreadID) {
    uint parentIndex = groupID.y * tileCountX() + groupID.x;
    if (parentIndex >= tileCountX() * tileCountY() ||
        adaptiveTileSplitFlags[parentIndex] == 0u) return;

    uint4 range = parentRanges[parentIndex * 4u];
    uint4 offsets = childOffsets[parentIndex];
    uint lane = groupThreadID.x;
    if (lane == 0u) sharedWritten = uint4(0u);
    GroupMemoryBarrierWithGroupSync();

    for (uint batchStart = 0u; batchStart < range.y; batchStart += 256u) {
        bool valid = batchStart + lane < range.y;
        uint mask = 0u;
        uint gaussianIndex = 0u;
        if (valid) {
            uint itemIndex = range.x + batchStart + lane;
            mask = childMasks[itemIndex];
            gaussianIndex = sortedTileItems[itemIndex];
        }
        uint4 membership = uint4(mask & 1u, (mask >> 1u) & 1u,
                                  (mask >> 2u) & 1u, (mask >> 3u) & 1u);
        sharedPrefix[lane] = membership;
        GroupMemoryBarrierWithGroupSync();

        for (uint step = 1u; step < 256u; step <<= 1u) {
            uint4 addend = lane >= step ? sharedPrefix[lane - step] : uint4(0u);
            GroupMemoryBarrierWithGroupSync();
            sharedPrefix[lane] += addend;
            GroupMemoryBarrierWithGroupSync();
        }

        uint4 localOffset = sharedPrefix[lane] - membership;
        uint4 batchOffset = sharedWritten;
        for (uint child = 0u; child < 4u; ++child) {
            if ((mask & (1u << child)) != 0u) {
                uint outputIndex = params.tileItemCount + offsets[child] +
                                   batchOffset[child] + localOffset[child];
                sortedTileItems[outputIndex] = gaussianIndex;
            }
        }

        GroupMemoryBarrierWithGroupSync();
        if (lane == 0u) sharedWritten += sharedPrefix[255u];
        GroupMemoryBarrierWithGroupSync();
    }
}
