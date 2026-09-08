#include "../common/train_forward_tile_common.slang"

[[vk::binding(3, 0)]]
StructuredBuffer<ProjectedGaussian> projectedGaussians;

[[vk::binding(4, 0)]]
RWStructuredBuffer<uint> tileItems;

[[vk::binding(20, 0)]]
RWStructuredBuffer<uint> tileKeyLow;

[[vk::binding(21, 0)]]
RWStructuredBuffer<uint> tileKeyHigh;

[[vk::binding(22, 0)]]
RWStructuredBuffer<uint> tileSortIndices;

[[vk::binding(32, 0)]]
StructuredBuffer<uint2> gaussianTileRanges;

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

    ProjectedGaussian gaussian = projectedGaussians[gaussianIndex];
    GaussianTileCullingData culling = trainPrepareGaussianTileCulling(gaussian);
    if (!culling.active || !culling.bounds.valid) {
        return;
    }

    uint2 gaussianRange = gaussianTileRanges[gaussianIndex];
    if (gaussianRange.y == 0u) {
        return;
    }
    uint outputIndex = gaussianRange.x;
    uint outputEnd = min(gaussianRange.x + gaussianRange.y, params.tileItemCount);
    uint depthKey = floatDepthSortKey(gaussian.centerRadius.z);
    for (uint tileY = culling.bounds.minTile.y;
         tileY < culling.bounds.maxTileExclusive.y; ++tileY) {
        for (uint tileX = culling.bounds.minTile.x;
             tileX < culling.bounds.maxTileExclusive.x; ++tileX) {
            if (!trainProjectedGaussianMayContributeToTile(culling, uint2(tileX, tileY))) {
                continue;
            }
            if (outputIndex < outputEnd) {
                uint tileIndex = trainTileIndex(uint2(tileX, tileY));
                tileItems[outputIndex] = gaussianIndex;
                tileKeyLow[outputIndex] = depthKey;
                tileKeyHigh[outputIndex] = tileIndex;
                tileSortIndices[outputIndex] = outputIndex;
            }
            ++outputIndex;
        }
    }
}
