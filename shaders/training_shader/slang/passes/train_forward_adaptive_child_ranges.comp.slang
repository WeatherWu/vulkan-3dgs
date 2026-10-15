#include "../common/train_forward_tile_common.slang"

[[vk::binding(5, 0)]]
RWStructuredBuffer<uint4> tileRanges;

[[vk::binding(34, 0)]]
StructuredBuffer<uint> adaptiveTileSplitFlags;

[[vk::binding(35, 0)]]
StructuredBuffer<uint4> childCounts;

[[vk::binding(37, 0)]]
StructuredBuffer<uint4> childOffsets;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint parentIndex = dispatchThreadID.x;
    if (parentIndex >= tileCountX() * tileCountY() ||
        adaptiveTileSplitFlags[parentIndex] == 0u) return;

    uint4 counts = childCounts[parentIndex];
    uint4 offsets = childOffsets[parentIndex];
    uint leafBase = parentIndex * 4u;
    for (uint child = 0u; child < 4u; ++child) {
        tileRanges[leafBase + child] = uint4(
            params.tileItemCount + offsets[child], counts[child], 0u, 0u);
    }
}
