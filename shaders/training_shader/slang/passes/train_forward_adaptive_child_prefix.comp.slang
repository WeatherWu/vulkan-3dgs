#include "../common/train_forward_tile_common.slang"

[[vk::binding(5, 0)]]
StructuredBuffer<uint4> parentRanges;

[[vk::binding(35, 0)]]
StructuredBuffer<uint4> childCounts;

[[vk::binding(38, 0)]]
RWStructuredBuffer<uint> childPrefix;

[[vk::binding(39, 0)]]
RWStructuredBuffer<uint> childBlockSums;

groupshared uint scanValues[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 groupThreadID : SV_GroupThreadID,
          uint3 groupID : SV_GroupID) {
    uint lane = groupThreadID.x;
    uint parentIndex = groupID.x * 256u + lane;
    uint parentCount = tileCountX() * tileCountY();
    uint cost = 0u;
    if (parentIndex < parentCount) {
        uint parentItems = parentRanges[parentIndex * 4u].y;
        uint4 counts = childCounts[parentIndex];
        if (trainAdaptiveRefinementIsProfitable(parentItems, counts)) {
            cost = trainAdaptiveChildItemCount(counts);
        }
    }
    scanValues[lane] = cost;
    GroupMemoryBarrierWithGroupSync();

    for (uint offset = 1u; offset < 256u; offset <<= 1u) {
        uint addend = lane >= offset ? scanValues[lane - offset] : 0u;
        GroupMemoryBarrierWithGroupSync();
        scanValues[lane] += addend;
        GroupMemoryBarrierWithGroupSync();
    }

    if (parentIndex < parentCount) {
        childPrefix[parentIndex] = lane == 0u ? 0u : scanValues[lane - 1u];
    }
    if (lane == 255u) {
        childBlockSums[groupID.x] = scanValues[lane];
    }
}
