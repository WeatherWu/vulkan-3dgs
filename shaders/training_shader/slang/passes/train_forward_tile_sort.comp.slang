#include "../common/train_forward_tile_common.slang"

[[vk::binding(4, 0)]]
StructuredBuffer<uint> tileItems;

[[vk::binding(5, 0)]]
RWStructuredBuffer<uint4> tileRanges;

[[vk::binding(21, 0)]]
StructuredBuffer<uint> tileKeyHigh;

[[vk::binding(22, 0)]]
StructuredBuffer<uint> tileSortIndices;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint tileIndex = dispatchThreadID.x;
    if (tileIndex >= trainTileCount()) {
        return;
    }

    uint4 oldRange = tileRanges[tileIndex];
    uint expectedCount = oldRange.y;
    if (expectedCount == 0 || params.tileItemCount == 0) {
        tileRanges[tileIndex] = uint4(0, 0, 0, 0);
        return;
    }
    if (oldRange.x >= params.tileItemCount) {
        tileRanges[tileIndex] = uint4(params.tileItemCount, 0, 0, 0);
        return;
    }

    uint searchBegin = oldRange.x;
    while (searchBegin > 0) {
        uint sourceIndex = tileSortIndices[searchBegin - 1u];
        if (tileKeyHigh[sourceIndex] != tileIndex) {
            break;
        }
        --searchBegin;
    }

    while (searchBegin < params.tileItemCount) {
        uint sourceIndex = tileSortIndices[searchBegin];
        if (tileKeyHigh[sourceIndex] >= tileIndex) {
            break;
        }
        ++searchBegin;
    }

    uint searchEnd = min(searchBegin + expectedCount, params.tileItemCount);
    while (searchEnd < params.tileItemCount) {
        uint sourceIndex = tileSortIndices[searchEnd];
        if (tileKeyHigh[sourceIndex] != tileIndex) {
            break;
        }
        ++searchEnd;
    }

    uint count = 0;
    if (searchBegin < params.tileItemCount) {
        uint sourceIndex = tileSortIndices[searchBegin];
        if (tileKeyHigh[sourceIndex] == tileIndex) {
            count = searchEnd - searchBegin;
        }
    }

    tileRanges[tileIndex] = uint4(searchBegin, count, 0, 0);
}
