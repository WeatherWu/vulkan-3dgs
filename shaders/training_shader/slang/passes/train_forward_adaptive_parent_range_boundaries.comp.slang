#include "../common/train_forward_tile_common.slang"

[[vk::binding(5, 0)]]
RWStructuredBuffer<uint4> tileRanges;

[[vk::binding(21, 0)]]
StructuredBuffer<uint> tileKeyHigh;

[[vk::binding(22, 0)]]
StructuredBuffer<uint> tileSortIndices;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;
    if (index > params.tileItemCount) return;

    uint currentParent = index == params.tileItemCount
        ? tileCountX() * trainTileCountY()
        : tileKeyHigh[tileSortIndices[index]];
    uint previousParent = index == 0u
        ? tileCountX() * trainTileCountY()
        : tileKeyHigh[tileSortIndices[index - 1u]];
    uint currentLeaf = currentParent * 4u;
    uint previousLeaf = previousParent * 4u;
    if (index < params.tileItemCount && currentLeaf < tileRangeCapacity() &&
        (index == 0u || currentParent != previousParent)) {
        uint ignored = 0u;
        InterlockedExchange(tileRanges[currentLeaf].x, index, ignored);
    }
    if (index > 0u && previousLeaf < tileRangeCapacity() &&
        (index == params.tileItemCount || currentParent != previousParent)) {
        uint ignored = 0u;
        InterlockedExchange(tileRanges[previousLeaf].y, index, ignored);
    }
}
