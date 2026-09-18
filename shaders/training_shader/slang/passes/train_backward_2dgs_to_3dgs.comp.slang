#include "../common/train_backward_project_common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<GaussianTrainParam> gaussianParams;

[[vk::binding(1, 0)]]
RWStructuredBuffer<GaussianGrad> gaussianGrads;

[[vk::binding(3, 0)]]
StructuredBuffer<ProjectedGaussian> projectedGaussians;

[[vk::binding(13, 0)]]
StructuredBuffer<ProjectedGaussianGrad> projectedGrads;

[[vk::binding(16, 0)]]
RWStructuredBuffer<GaussianDensificationState> densificationStates;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint gaussianIndex = dispatchThreadID.x;
    if (gaussianIndex >= params.gaussianCount) {
        return;
    }

    GaussianTrainParam gaussian = gaussianParams[gaussianIndex];
    ProjectedGaussian projectedGaussian = projectedGaussians[gaussianIndex];
    if (!projectedGaussianHasBackwardSupport(projectedGaussian)) {
        return;
    }

    GaussianGrad grad = gaussianGrads[gaussianIndex];
    GaussianDensificationState densification = densificationStates[gaussianIndex];
    accumulateProjectedGradientToGaussian(
        gaussian,
        projectedGrads[gaussianIndex],
        grad,
        densification);
    gaussianGrads[gaussianIndex] = grad;
    densificationStates[gaussianIndex] = densification;
}
