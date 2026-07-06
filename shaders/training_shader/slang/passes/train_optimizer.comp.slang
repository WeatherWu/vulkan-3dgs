#include "../common/train_common.slang"

[[vk::binding(0, 0)]]
RWStructuredBuffer<GaussianTrainParam> gaussianParams;

[[vk::binding(1, 0)]]
StructuredBuffer<GaussianGrad> gaussianGrads;

[[vk::binding(2, 0)]]
RWStructuredBuffer<AdamState> adamStates;

[[vk::binding(9, 0)]]
RWStructuredBuffer<uint4> counters;

float4 clipGrad(float4 grad) {
    float gradClip = max(params.optimizerGradClip, 1e-8);
    return clamp(grad, float4(-gradClip), float4(gradClip));
}

float4 safeNormalizeQuaternion(float4 rotation) {
    float lengthSquared = dot(rotation, rotation);
    if (lengthSquared <= 1e-16) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    return rotation * rsqrt(lengthSquared);
}

float positionLearningRate(float step) {
    float maxSteps = max(params.positionLearningRateMaxSteps, 1.0);
    float t = clamp((step - 1.0) / maxSteps, 0.0, 1.0);
    float lrInit = max(params.positionLearningRate, 0.0);
    float lrFinal = max(params.positionLearningRateFinal, 0.0);
    float lr = exp(lerp(log(max(lrInit, 1e-12)), log(max(lrFinal, 1e-12)), t));
    float delayMult = clamp(params.positionLearningRateDelayMult, 0.0, 1.0);
    if (delayMult < 1.0) {
        float delayRate = delayMult + (1.0 - delayMult) * sin(0.5 * 3.141592653589793 * t);
        lr *= delayRate;
    }
    return lr;
}

float4 adamUpdate(inout float4 firstMoment,
                  inout float4 secondMoment,
                  float4 grad,
                  float step,
                  float learningRate) {
    grad = clipGrad(grad);
    float beta1 = clamp(params.optimizerBeta1, 0.0, 0.999999);
    float beta2 = clamp(params.optimizerBeta2, 0.0, 0.999999);
    firstMoment = beta1 * firstMoment + (1.0 - beta1) * grad;
    secondMoment = beta2 * secondMoment + (1.0 - beta2) * grad * grad;

    float firstMomentCorrection = 1.0 - pow(beta1, step);
    float secondMomentCorrection = 1.0 - pow(beta2, step);
    float4 correctedFirstMoment = firstMoment / max(firstMomentCorrection, 1e-8);
    float4 correctedSecondMoment = secondMoment / max(secondMomentCorrection, 1e-8);
    return max(learningRate, 0.0) *
           correctedFirstMoment /
           (sqrt(correctedSecondMoment) + float4(max(params.optimizerEpsilon, 1e-12)));
}

void updateGaussianParam(inout GaussianTrainParam gaussian,
                         inout AdamState adamState,
                         GaussianGrad grad,
                         float step) {
    float4 positionOpacityLearningRate = float4(
        positionLearningRate(step),
        positionLearningRate(step),
        positionLearningRate(step),
        max(params.opacityLearningRate, 0.0)
    );
    gaussian.positionOpacity -= adamUpdate(adamState.firstMoment.positionOpacity,
                                           adamState.secondMoment.positionOpacity,
                                           grad.positionOpacity,
                                           step,
                                           1.0) * positionOpacityLearningRate;
    gaussian.scale -= adamUpdate(adamState.firstMoment.scale,
                                 adamState.secondMoment.scale,
                                 grad.scale,
                                 step,
                                 max(params.scaleLearningRate, 0.0));
    gaussian.rotation -= adamUpdate(adamState.firstMoment.rotation,
                                    adamState.secondMoment.rotation,
                                    grad.rotation,
                                    step,
                                    max(params.rotationLearningRate, 0.0));

    gaussian.rotation = safeNormalizeQuaternion(gaussian.rotation);

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
                                     step,
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
    float step = float(counters[0].w + 1u);
    updateGaussianParam(gaussian, adamState, gaussianGrads[index], step);
    gaussianParams[index] = gaussian;
    adamStates[index] = adamState;

    if (index == 0) {
        uint ignored = 0;
        InterlockedAdd(counters[0].w, 1u, ignored);
    }
}
