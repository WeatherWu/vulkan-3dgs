#include "train_backward_project_common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<GaussianTrainParam> gaussianParams;

[[vk::binding(1, 0)]]
RWStructuredBuffer<GaussianGrad> gaussianGrads;

[[vk::binding(13, 0)]]
StructuredBuffer<ProjectedGaussianGrad> projectedGrads;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint gaussianIndex = dispatchThreadID.x;
    if (gaussianIndex >= params.gaussianCount) {
        return;
    }

    GaussianTrainParam gaussian = gaussianParams[gaussianIndex];
    ProjectedGaussianGrad projectedGrad = projectedGrads[gaussianIndex];
    ProjectionLocalState projectionState = trainBuildProjectionLocalState(gaussian);

    GaussianGrad grad = gaussianGrads[gaussianIndex];
    grad.positionOpacity.xyz += backwardPixelCenterToPosition(projectionState, projectedGrad.centerRadius.xy);
    grad.positionOpacity.xyz += backwardDepthToPosition(projectedGrad.centerRadius.z);
    grad.positionOpacity.w += projectedGrad.conicOpacity.w;

    float3x3 dLoss_dCovariancePixel = backwardConicToCovariance(projectionState.covariancePixel, projectedGrad.conicOpacity.xyz);
    float3x3 dLoss_dCovariance3D = backwardCovariancePixelToCovariance3D(projectionState, dLoss_dCovariancePixel);
    grad.scale.xyz += backwardCovariance3DToScale(gaussian, dLoss_dCovariance3D);

    backwardProjectedColorToSH(grad, gaussian, projectedGrad.color.xyz);
    gaussianGrads[gaussianIndex] = grad;
}
