#include "train_forward_tile_common.slang"

[[vk::binding(3, 0)]]
StructuredBuffer<ProjectedGaussian> projectedGaussians;

[[vk::binding(4, 0)]]
RWStructuredBuffer<uint> tileItems;

[[vk::binding(5, 0)]]
RWStructuredBuffer<uint4> tileRanges;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint gaussianIndex = dispatchThreadID.x;
    if (gaussianIndex >= params.gaussianCount) {
        return;
    }

    TileBounds bounds = trainProjectedTileBounds(projectedGaussians[gaussianIndex]);
    if (!bounds.valid) {
        return;
    }

    for (uint tileY = bounds.minTile.y; tileY <= bounds.maxTile.y; ++tileY) {
        for (uint tileX = bounds.minTile.x; tileX <= bounds.maxTile.x; ++tileX) {
            uint tileIndex = trainTileIndex(uint2(tileX, tileY));
            uint localOffset = 0;
            InterlockedAdd(tileRanges[tileIndex].z, 1u, localOffset);

            uint4 range = tileRanges[tileIndex];
            if (localOffset < range.y) {
                tileItems[range.x + localOffset] = gaussianIndex;
            }
        }
    }
}
