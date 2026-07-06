#include "../common/train_forward_tile_common.slang"

[[vk::binding(3, 0)]]
StructuredBuffer<ProjectedGaussian> projectedGaussians;

[[vk::binding(4, 0)]]
RWStructuredBuffer<uint> tileItems;

[[vk::binding(5, 0)]]
RWStructuredBuffer<uint4> tileRanges;

[[vk::binding(20, 0)]]
RWStructuredBuffer<uint> tileKeyLow;

[[vk::binding(21, 0)]]
RWStructuredBuffer<uint> tileKeyHigh;

[[vk::binding(22, 0)]]
RWStructuredBuffer<uint> tileSortIndices;

uint floatDepthSortKey(float depth) {
    uint bits = asuint(depth);
    return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
}

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

    ProjectedGaussian gaussian = projectedGaussians[gaussianIndex];
    for (uint tileY = bounds.minTile.y; tileY <= bounds.maxTile.y; ++tileY) {
        for (uint tileX = bounds.minTile.x; tileX <= bounds.maxTile.x; ++tileX) {
            uint tileIndex = trainTileIndex(uint2(tileX, tileY));
            uint localOffset = 0;
            InterlockedAdd(tileRanges[tileIndex].z, 1u, localOffset);

            uint4 range = tileRanges[tileIndex];
            if (localOffset < range.y) {
                uint outputIndex = range.x + localOffset;
                tileItems[outputIndex] = gaussianIndex;
                tileKeyLow[outputIndex] = floatDepthSortKey(gaussian.centerRadius.z);
                tileKeyHigh[outputIndex] = tileIndex;
                tileSortIndices[outputIndex] = outputIndex;
            }
        }
    }
}
