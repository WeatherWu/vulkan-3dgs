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
    if (index > params.tileItemCount) {
        return;
    }

    uint currentTile = index == params.tileItemCount
        ? trainTileCount()
        : tileKeyHigh[tileSortIndices[index]];
    uint previousTile = index == 0u
        ? trainTileCount()
        : tileKeyHigh[tileSortIndices[index - 1u]];
    if (index < params.tileItemCount && currentTile < trainTileCount() &&
        (index == 0u || currentTile != previousTile)) {
        uint ignored = 0u;
        InterlockedExchange(tileRanges[currentTile].x, index, ignored);
    }
    if (index > 0u && previousTile < trainTileCount() &&
        (index == params.tileItemCount || currentTile != previousTile)) {
        uint ignored = 0u;
        InterlockedExchange(tileRanges[previousTile].y, index, ignored);
    }
}
