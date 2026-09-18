#include "../common/train_backward_common.slang"

[[vk::binding(1, 0)]]
RWStructuredBuffer<GaussianGrad> gaussianGrads;

[[vk::binding(13, 0)]]
RWStructuredBuffer<ProjectedGaussianGrad> projectedGrads;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;

    if (index < params.gaussianCount) {
        GaussianGrad gaussianGrad;
        gaussianGrad.positionOpacity = float4(0.0);
        gaussianGrad.scale = float4(0.0);
        gaussianGrad.rotation = float4(0.0);
        for (uint i = 0; i < 12; ++i) {
            gaussianGrad.sh[i] = float4(0.0);
        }
        if (params.clearGaussianGrads != 0u) {
            gaussianGrads[index] = gaussianGrad;
        }
        projectedGrads[index] = makeZeroProjectedGaussianGrad();
    }
}
