#include "train_common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<GaussianTrainParam> gaussianParams;

[[vk::binding(3, 0)]]
RWStructuredBuffer<ProjectedGaussian> projectedGaussians;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;
    if (index >= params.gaussianCount) {
        return;
    }

    GaussianTrainParam gaussian = gaussianParams[index];
    ProjectedGaussian projected;
    projected.centerRadius = float4(gaussian.positionOpacity.xyz, max(gaussian.scale.x, 1e-4));
    projected.conicOpacity = float4(1.0, 0.0, 1.0, gaussian.positionOpacity.w);
    projected.color = max(evaluateSH(gaussian, float3(0.0, 0.0, 1.0)), 0.0);
    projectedGaussians[index] = projected;
}
