#include "../common/train_backward_project_common.slang"
#include "../common/train_optimizer_common.slang"

[[vk::binding(0, 0)]]
RWStructuredBuffer<GaussianTrainParam> gaussianParams;

[[vk::binding(2, 0)]]
RWStructuredBuffer<AdamState> adamStates;

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
    GaussianGrad grad = makeZeroGaussianGrad();
    ProjectedGaussian projectedGaussian = projectedGaussians[gaussianIndex];
    if (projectedGaussianHasBackwardSupport(projectedGaussian)) {
        GaussianDensificationState densification =
            densificationStates[gaussianIndex];
        accumulateProjectedGradientToGaussian(
            gaussian,
            projectedGrads[gaussianIndex],
            grad,
            densification);
        densificationStates[gaussianIndex] = densification;
    }

    AdamState adamState = adamStates[gaussianIndex];
    updateGaussianParam(gaussian, adamState, grad);
    gaussianParams[gaussianIndex] = gaussian;
    adamStates[gaussianIndex] = adamState;
}
