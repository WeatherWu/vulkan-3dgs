// S4-1R offline-only candidate. Same AoS bytes/bindings, FP32 and group256.
// Original projection/SH gradients finish BEFORE any parameter/moment update.
#include "../common/train_backward_project_common.slang"
#include "../common/train_optimizer_common.slang"

[[vk::binding(0, 0)]]
RWStructuredBuffer<GaussianTrainParam> gaussianParams;

// Original typed AoS buffer. Only each pair of float4 moments is local.
[[vk::binding(2, 0)]]
RWStructuredBuffer<AdamState> adamStates;

[[vk::binding(3, 0)]]
StructuredBuffer<ProjectedGaussian> projectedGaussians;
[[vk::binding(13, 0)]]
StructuredBuffer<ProjectedGaussianGrad> projectedGrads;
[[vk::binding(16, 0)]]
RWStructuredBuffer<GaussianDensificationState> densificationStates;
[[vk::binding(32, 0)]]
StructuredBuffer<uint2> gaussianTileRanges;

void updateGaussianStreamed(uint gaussianIndex, inout GaussianTrainParam gaussian,
                            GaussianGrad grad) {
    float4 positionOpacityLearningRate = float4(
        max(params.positionLearningRate, 0.0),
        max(params.positionLearningRate, 0.0),
        max(params.positionLearningRate, 0.0),
        max(params.opacityLearningRate, 0.0));
    // Preserve original ordering: Adam(rate=1) * vector position/opacity LR.
    {
        float4 firstMoment = adamStates[gaussianIndex].firstMoment.positionOpacity;
        float4 secondMoment = adamStates[gaussianIndex].secondMoment.positionOpacity;
        gaussian.positionOpacity -= adamUpdate(firstMoment, secondMoment, grad.positionOpacity, 1.0)
            * positionOpacityLearningRate;
        adamStates[gaussianIndex].firstMoment.positionOpacity = firstMoment;
        adamStates[gaussianIndex].secondMoment.positionOpacity = secondMoment;
    }
    {
        float4 firstMoment = adamStates[gaussianIndex].firstMoment.scale;
        float4 secondMoment = adamStates[gaussianIndex].secondMoment.scale;
        gaussian.scale -= adamUpdate(firstMoment, secondMoment, grad.scale, max(params.scaleLearningRate, 0.0));
        adamStates[gaussianIndex].firstMoment.scale = firstMoment;
        adamStates[gaussianIndex].secondMoment.scale = secondMoment;
    }
    {
        float4 firstMoment = adamStates[gaussianIndex].firstMoment.rotation;
        float4 secondMoment = adamStates[gaussianIndex].secondMoment.rotation;
        gaussian.rotation -= adamUpdate(firstMoment, secondMoment, grad.rotation, max(params.rotationLearningRate, 0.0));
        adamStates[gaussianIndex].firstMoment.rotation = firstMoment;
        adamStates[gaussianIndex].secondMoment.rotation = secondMoment;
    }

    uint activeCoefficientCount = params.activeSHDegree < 1u
        ? 1u
        : (params.activeSHDegree < 2u ? 4u : (params.activeSHDegree < 3u ? 9u : 16u));
    for (uint packedIndex = 0u; packedIndex < 12u; ++packedIndex) {
        uint coefficientBase = (packedIndex % 4u) * 4u;
        float4 activeMask = float4(
            coefficientBase + 0u < activeCoefficientCount ? 1.0 : 0.0,
            coefficientBase + 1u < activeCoefficientCount ? 1.0 : 0.0,
            coefficientBase + 2u < activeCoefficientCount ? 1.0 : 0.0,
            coefficientBase + 3u < activeCoefficientCount ? 1.0 : 0.0);
        float4 learningRate = float4(
            coefficientBase == 0u ? max(params.featureLearningRate, 0.0)
                                  : max(params.featureRestLearningRate, 0.0),
            max(params.featureRestLearningRate, 0.0),
            max(params.featureRestLearningRate, 0.0),
            max(params.featureRestLearningRate, 0.0));
        float4 firstMoment = adamStates[gaussianIndex].firstMoment.sh[packedIndex];
        float4 secondMoment = adamStates[gaussianIndex].secondMoment.sh[packedIndex];
        gaussian.sh[packedIndex] -= adamUpdateMasked(firstMoment, secondMoment,
            grad.sh[packedIndex], learningRate, activeMask);
        adamStates[gaussianIndex].firstMoment.sh[packedIndex] = firstMoment;
        adamStates[gaussianIndex].secondMoment.sh[packedIndex] = secondMoment;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint gaussianIndex = dispatchThreadID.x;
    if (gaussianIndex >= params.gaussianCount) return;

    GaussianTrainParam gaussian = gaussianParams[gaussianIndex];
    GaussianGrad grad = makeZeroGaussianGrad();
    ProjectedGaussian projectedGaussian = projectedGaussians[gaussianIndex];
    if (projectedGaussianHasBackwardSupport(projectedGaussian)) {
        GaussianDensificationState densification = densificationStates[gaussianIndex];
        accumulateProjectedGradientToGaussian(gaussian, projectedGrads[gaussianIndex], grad,
            densification, gaussianTileRanges[gaussianIndex].y > 0u);
        densificationStates[gaussianIndex] = densification;
    }
    // Invisible/zero-gradient Gaussians still perform dense Adam decay/update.
    updateGaussianStreamed(gaussianIndex, gaussian, grad);
    gaussianParams[gaussianIndex] = gaussian;
}
