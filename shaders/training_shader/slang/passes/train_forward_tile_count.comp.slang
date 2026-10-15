#include "../common/train_forward_tile_common.slang"

[[vk::binding(3, 0)]]
StructuredBuffer<ProjectedGaussian> projectedGaussians;

[[vk::binding(32, 0)]]
RWStructuredBuffer<uint2> gaussianTileRanges;

[[vk::binding(16, 0)]]
RWStructuredBuffer<GaussianDensificationState> densificationStates;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint gaussianIndex = dispatchThreadID.x;
    if (gaussianIndex >= params.gaussianCount) {
        return;
    }

    ProjectedGaussian gaussian = projectedGaussians[gaussianIndex];
    GaussianTileCullingData culling = trainPrepareGaussianTileCulling(gaussian);
    uint tileCount = trainProjectedGaussianTileCount(culling);
    gaussianTileRanges[gaussianIndex] = uint2(
        0u, tileCount);

    if (tileCount > 0u) {
        GaussianDensificationState densification = densificationStates[gaussianIndex];
        densification.maxScreenRadius = max(densification.maxScreenRadius,
                                            gaussian.centerRadius.w);
        densificationStates[gaussianIndex] = densification;
    }
}
