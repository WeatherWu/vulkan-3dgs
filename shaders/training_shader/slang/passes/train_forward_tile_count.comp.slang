#include "../common/train_forward_tile_common.slang"

[[vk::binding(3, 0)]]
StructuredBuffer<ProjectedGaussian> projectedGaussians;

[[vk::binding(32, 0)]]
RWStructuredBuffer<uint2> gaussianTileRanges;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint gaussianIndex = dispatchThreadID.x;
    if (gaussianIndex >= params.gaussianCount) {
        return;
    }

    ProjectedGaussian gaussian = projectedGaussians[gaussianIndex];
    GaussianTileCullingData culling = trainPrepareGaussianTileCulling(gaussian);
    gaussianTileRanges[gaussianIndex] = uint2(
        0u, trainProjectedGaussianTileCount(culling));
}
