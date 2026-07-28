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
    if (!(projectedGaussian.centerRadius.w > 0.0) || !isfinite(projectedGaussian.centerRadius.w)) {
        return;
    }

    ProjectedGaussianGrad projectedGrad = projectedGrads[gaussianIndex];
    ProjectionLocalState projectionState = trainBuildProjectionLocalState(gaussian);

    GaussianGrad grad = gaussianGrads[gaussianIndex];
    float2 dLoss_dCenter = projectedGrad.centerRadius.xy;
    grad.positionOpacity.xyz += backwardPixelCenterToPosition(projectionState, dLoss_dCenter);
    grad.positionOpacity.xyz += backwardDepthToPosition(projectedGrad.centerRadius.z);
    grad.positionOpacity.w += projectedGrad.conicOpacity.w *
                              trainOpacityActivationBackward(gaussian.positionOpacity.w);

    float3x3 dLoss_dCovariancePixel = backwardConicToCovariance(projectionState.covariancePixel, projectedGrad.conicOpacity.xyz);
    grad.positionOpacity.xyz += backwardCovariancePixelToPosition(projectionState, dLoss_dCovariancePixel);
    float3x3 dLoss_dCovariance3D = backwardCovariancePixelToCovariance3D(projectionState, dLoss_dCovariancePixel);
    grad.scale.xyz += backwardCovariance3DToScale(gaussian, dLoss_dCovariance3D);
    grad.rotation += backwardCovariance3DToRotation(gaussian, dLoss_dCovariance3D);

    grad.positionOpacity.xyz += backwardProjectedColorToPosition(gaussian, projectedGrad.color.xyz);
    backwardProjectedColorToSH(grad, gaussian, projectedGrad.color.xyz);
    gaussianGrads[gaussianIndex] = grad;

    float2 densificationGrad = dLoss_dCenter *
        (0.5 * max(trainingCamera.viewport.zw, float2(1.0)));
    float screenGradMagnitude = length(densificationGrad);
    GaussianDensificationState densification = densificationStates[gaussianIndex];
    densification.screenGradSum += screenGradMagnitude;
    densification.screenGradCount += 1u;
    densificationStates[gaussianIndex] = densification;
}
