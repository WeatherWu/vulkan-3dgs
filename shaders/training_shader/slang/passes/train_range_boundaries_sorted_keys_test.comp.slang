// Range boundaries read the GPU-produced sorted tile keys directly.
#include "../common/train_forward_tile_common.slang"

[[vk::binding(5, 0)]] RWStructuredBuffer<uint4> tileRanges;
[[vk::binding(23, 0)]]
#if TRAIN_TILE_KEY16
StructuredBuffer<uint16_t> sortedTileKeys;
#else
StructuredBuffer<uint> sortedTileKeys;
#endif

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;
    if (index > params.tileItemCount) return;
    uint currentTile = index == params.tileItemCount ? tileRangeCapacity() : sortedTileKeys[index];
    uint previousTile = index == 0u ? tileRangeCapacity() : sortedTileKeys[index - 1u];
    if (index < params.tileItemCount && currentTile < tileRangeCapacity() &&
        (index == 0u || currentTile != previousTile)) {
        uint ignored = 0u;
        InterlockedExchange(tileRanges[currentTile].x, index, ignored);
    }
    if (index > 0u && previousTile < tileRangeCapacity() &&
        (index == params.tileItemCount || currentTile != previousTile)) {
        uint ignored = 0u;
        InterlockedExchange(tileRanges[previousTile].y, index, ignored);
    }
}
