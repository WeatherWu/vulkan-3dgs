#include "../common/train_forward_tile_common.slang"

[[vk::binding(3, 0)]]
StructuredBuffer<ProjectedGaussian> projectedGaussians;

[[vk::binding(4, 0)]]
RWStructuredBuffer<uint> tileItems;

[[vk::binding(5, 0)]]
StructuredBuffer<uint4> tileRanges;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint tileIndex = dispatchThreadID.x;
    if (tileIndex >= trainTileCount()) {
        return;
    }

    TileRange range = unpackTileRange(tileRanges[tileIndex]);
    if (range.count <= 1) {
        return;
    }

    for (uint i = 1; i < range.count; ++i) {
        uint current = tileItems[range.start + i];
        uint j = i;
        while (j > 0) {
            uint previous = tileItems[range.start + j - 1];
            if (trainGaussianDepthBefore(previous, current, projectedGaussians)) {
                break;
            }
            tileItems[range.start + j] = previous;
            --j;
        }
        tileItems[range.start + j] = current;
    }
}
