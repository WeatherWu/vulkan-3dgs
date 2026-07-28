#include "../common/train_forward_tile_common.slang"

[[vk::binding(3, 0)]]
StructuredBuffer<ProjectedGaussian> projectedGaussians;

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

    for (uint tileY = bounds.minTile.y; tileY < bounds.maxTileExclusive.y; ++tileY) {
        for (uint tileX = bounds.minTile.x; tileX < bounds.maxTileExclusive.x; ++tileX) {
            uint ignored = 0;
            InterlockedAdd(tileRanges[trainTileIndex(uint2(tileX, tileY))].y, 1u, ignored);
        }
    }
}
