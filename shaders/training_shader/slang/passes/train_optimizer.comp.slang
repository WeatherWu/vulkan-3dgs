#include "../common/train_common.slang"

[[vk::binding(0, 0)]]
RWStructuredBuffer<GaussianTrainParam> gaussianParams;

[[vk::binding(1, 0)]]
StructuredBuffer<GaussianGrad> gaussianGrads;

[[vk::binding(2, 0)]]
RWStructuredBuffer<AdamState> adamStates;

float4 clipGrad(float4 grad) {
    if (params.optimizerGradClip <= 0.0) {
        return grad;
    }

    float gradClip = max(params.optimizerGradClip, 1e-15);
    return clamp(grad, float4(-gradClip), float4(gradClip));
}

float4 adamUpdate(inout float4 firstMoment,
                  inout float4 secondMoment,
                  float4 grad,
                  float learningRate) {
    grad = clipGrad(grad);
    float beta1 = clamp(params.optimizerBeta1, 0.0, 0.999999);
    float beta2 = clamp(params.optimizerBeta2, 0.0, 0.999999);
    firstMoment = beta1 * firstMoment + params.optimizerOneMinusBeta1 * grad;
    secondMoment = beta2 * secondMoment + params.optimizerOneMinusBeta2 * grad * grad;

    float4 correctedFirstMoment = firstMoment * params.optimizerInvFirstMomentCorrection;
    float4 correctedSecondMoment = secondMoment * params.optimizerInvSecondMomentCorrection;
    return max(learningRate, 0.0) *
           correctedFirstMoment /
           (sqrt(correctedSecondMoment) + float4(max(params.optimizerEpsilon, 1e-15)));
}

void updateGaussianParam(inout GaussianTrainParam gaussian,
                          inout AdamState adamState,
                          GaussianGrad grad) {
    float4 positionOpacityLearningRate = float4(
        max(params.positionLearningRate, 0.0),
        max(params.positionLearningRate, 0.0),
        max(params.positionLearningRate, 0.0),
        max(params.opacityLearningRate, 0.0)
    );
    gaussian.positionOpacity -= adamUpdate(adamState.firstMoment.positionOpacity,
                                           adamState.secondMoment.positionOpacity,
                                           grad.positionOpacity,
                                           1.0) * positionOpacityLearningRate;
    gaussian.scale -= adamUpdate(adamState.firstMoment.scale,
                                 adamState.secondMoment.scale,
                                 grad.scale,
                                 max(params.scaleLearningRate, 0.0));
    gaussian.rotation -= adamUpdate(adamState.firstMoment.rotation,
                                     adamState.secondMoment.rotation,
                                     grad.rotation,
                                     max(params.rotationLearningRate, 0.0));

    for (uint i = 0; i < 16; ++i) {
        if ((params.activeSHDegree < 1 && i >= 1) ||
            (params.activeSHDegree < 2 && i >= 4) ||
            (params.activeSHDegree < 3 && i >= 9)) {
            continue;
        }
        float shLearningRate = i == 0
            ? max(params.featureLearningRate, 0.0)
            : max(params.featureRestLearningRate, 0.0);
        gaussian.sh[i] -= adamUpdate(adamState.firstMoment.sh[i],
                                      adamState.secondMoment.sh[i],
                                      grad.sh[i],
                                      shLearningRate);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;
    if (index >= params.gaussianCount) {
        return;
    }

    GaussianTrainParam gaussian = gaussianParams[index];
    AdamState adamState = adamStates[index];
    updateGaussianParam(gaussian, adamState, gaussianGrads[index]);
    gaussianParams[index] = gaussian;
    adamStates[index] = adamState;
}
