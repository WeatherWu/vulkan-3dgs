#include "../common/train_common.slang"

[[vk::binding(0, 0)]]
RWStructuredBuffer<GaussianTrainParam> gaussianParams;

[[vk::binding(1, 0)]]
StructuredBuffer<GaussianGrad> gaussianGrads;

[[vk::binding(2, 0)]]
RWStructuredBuffer<AdamState> adamStates;

[[vk::binding(19, 0)]]
StructuredBuffer<uint> densificationCounters;

[[vk::binding(30, 0)]]
RWStructuredBuffer<TrainingGaussianValidationPartial> gaussianValidationPartials;

groupshared uint validationNonFiniteGaussianCount[256];
groupshared uint validationNonFinitePositionCount[256];
groupshared uint validationNonFiniteOpacityCount[256];
groupshared uint validationNonFiniteRawScaleCount[256];
groupshared uint validationNonFiniteActivatedScaleCount[256];
groupshared uint validationNonFiniteRotationCount[256];
groupshared uint validationNonFiniteSHCount[256];
groupshared uint validationFirstNonFiniteGaussianIndex[256];

float4 clipGrad(float4 grad) {
    if (params.optimizerGradClip <= 0.0) {
        return grad;
    }

    float gradClip = max(params.optimizerGradClip, 1e-15);
    return clamp(grad, float4(-gradClip), float4(gradClip));
}

float positionLearningRate(float step) {
    float lrInit = max(params.positionLearningRate, 0.0);
    float lrFinal = max(params.positionLearningRateFinal, 0.0);
    if (lrInit == 0.0 && lrFinal == 0.0) {
        return 0.0;
    }

    float maxSteps = max(params.positionLearningRateMaxSteps, 1.0);
    float t = clamp(step / maxSteps, 0.0, 1.0);
    float lr = exp(lerp(log(max(lrInit, 1e-12)), log(max(lrFinal, 1e-12)), t));
    float delaySteps = params.positionLearningRateDelaySteps;
    if (delaySteps > 0.0) {
        float delayMult = clamp(params.positionLearningRateDelayMult, 0.0, 1.0);
        float delayT = clamp(step / delaySteps, 0.0, 1.0);
        float delayRate = delayMult + (1.0 - delayMult) * sin(0.5 * 3.141592653589793 * delayT);
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
           (sqrt(correctedSecondMoment) + float4(max(params.optimizerEpsilon, 1e-15)));
}

void updateGaussianParam(inout GaussianTrainParam gaussian,
                         inout AdamState adamState,
                         GaussianGrad grad,
                         float optimizerStep,
                         float learningRateStep) {
    float4 positionOpacityLearningRate = float4(
        positionLearningRate(learningRateStep),
        positionLearningRate(learningRateStep),
        positionLearningRate(learningRateStep),
        max(params.opacityLearningRate, 0.0)
    );
    gaussian.positionOpacity -= adamUpdate(adamState.firstMoment.positionOpacity,
                                           adamState.secondMoment.positionOpacity,
                                           grad.positionOpacity,
                                           optimizerStep,
                                           1.0) * positionOpacityLearningRate;
    gaussian.scale -= adamUpdate(adamState.firstMoment.scale,
                                 adamState.secondMoment.scale,
                                 grad.scale,
                                 optimizerStep,
                                 max(params.scaleLearningRate, 0.0));
    gaussian.rotation -= adamUpdate(adamState.firstMoment.rotation,
                                    adamState.secondMoment.rotation,
                                    grad.rotation,
                                    optimizerStep,
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
                                     optimizerStep,
                                     shLearningRate);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID,
          uint3 groupThreadID : SV_GroupThreadID,
          uint3 groupID : SV_GroupID) {
    uint index = dispatchThreadID.x;
    uint gaussianCount = params.validationUsesDensifiedGaussians != 0u
        ? min(densificationCounters[1], params.gaussianCount)
        : params.gaussianCount;
    bool active = index < gaussianCount;
    uint nonFiniteCount = 0u;
    uint nonFinitePositionCount = 0u;
    uint nonFiniteOpacityCount = 0u;
    uint nonFiniteRawScaleCount = 0u;
    uint nonFiniteActivatedScaleCount = 0u;
    uint nonFiniteRotationCount = 0u;
    uint nonFiniteSHCount = 0u;
    uint firstNonFiniteGaussianIndex = 0xffffffffu;

    if (active) {
        GaussianTrainParam gaussian = gaussianParams[index];
        if (params.optimizerEnabled != 0u) {
            AdamState adamState = adamStates[index];
            float optimizerStep = float(max(params.optimizerStep, 1u));
            float learningRateStep = float(params.trainingIteration + 1u);
            updateGaussianParam(gaussian,
                                adamState,
                                gaussianGrads[index],
                                optimizerStep,
                                learningRateStep);
            gaussianParams[index] = gaussian;
            adamStates[index] = adamState;
        }

        if (params.validationEnabled != 0u) {
            bool finitePosition = all(isfinite(gaussian.positionOpacity.xyz));
            bool finiteOpacity = isfinite(gaussian.positionOpacity.w);
            bool finiteRawScale = all(isfinite(gaussian.scale.xyz));
            bool finiteActivatedScale = all(isfinite(exp(gaussian.scale.xyz)));
            bool finiteRotation = all(isfinite(gaussian.rotation));
            bool finiteSH = true;
            for (uint coefficient = 0u; coefficient < 16u; ++coefficient) {
                finiteSH = finiteSH && all(isfinite(gaussian.sh[coefficient]));
            }
            bool finite = finitePosition && finiteOpacity && finiteRawScale &&
                          finiteActivatedScale && finiteRotation && finiteSH;
            nonFiniteCount = finite ? 0u : 1u;
            nonFinitePositionCount = finitePosition ? 0u : 1u;
            nonFiniteOpacityCount = finiteOpacity ? 0u : 1u;
            nonFiniteRawScaleCount = finiteRawScale ? 0u : 1u;
            nonFiniteActivatedScaleCount = finiteActivatedScale ? 0u : 1u;
            nonFiniteRotationCount = finiteRotation ? 0u : 1u;
            nonFiniteSHCount = finiteSH ? 0u : 1u;
            firstNonFiniteGaussianIndex = finite ? 0xffffffffu : index;
        }
    }

    if (params.validationEnabled == 0u) {
        return;
    }

    uint localIndex = groupThreadID.x;
    validationNonFiniteGaussianCount[localIndex] = nonFiniteCount;
    validationNonFinitePositionCount[localIndex] = nonFinitePositionCount;
    validationNonFiniteOpacityCount[localIndex] = nonFiniteOpacityCount;
    validationNonFiniteRawScaleCount[localIndex] = nonFiniteRawScaleCount;
    validationNonFiniteActivatedScaleCount[localIndex] = nonFiniteActivatedScaleCount;
    validationNonFiniteRotationCount[localIndex] = nonFiniteRotationCount;
    validationNonFiniteSHCount[localIndex] = nonFiniteSHCount;
    validationFirstNonFiniteGaussianIndex[localIndex] = firstNonFiniteGaussianIndex;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (localIndex < stride) {
            validationNonFiniteGaussianCount[localIndex] +=
                validationNonFiniteGaussianCount[localIndex + stride];
            validationNonFinitePositionCount[localIndex] +=
                validationNonFinitePositionCount[localIndex + stride];
            validationNonFiniteOpacityCount[localIndex] +=
                validationNonFiniteOpacityCount[localIndex + stride];
            validationNonFiniteRawScaleCount[localIndex] +=
                validationNonFiniteRawScaleCount[localIndex + stride];
            validationNonFiniteActivatedScaleCount[localIndex] +=
                validationNonFiniteActivatedScaleCount[localIndex + stride];
            validationNonFiniteRotationCount[localIndex] +=
                validationNonFiniteRotationCount[localIndex + stride];
            validationNonFiniteSHCount[localIndex] +=
                validationNonFiniteSHCount[localIndex + stride];
            validationFirstNonFiniteGaussianIndex[localIndex] = min(
                validationFirstNonFiniteGaussianIndex[localIndex],
                validationFirstNonFiniteGaussianIndex[localIndex + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (localIndex == 0u) {
        TrainingGaussianValidationPartial partial;
        partial.nonFiniteGaussianCount = validationNonFiniteGaussianCount[0];
        partial.nonFinitePositionCount = validationNonFinitePositionCount[0];
        partial.nonFiniteOpacityCount = validationNonFiniteOpacityCount[0];
        partial.nonFiniteRawScaleCount = validationNonFiniteRawScaleCount[0];
        partial.nonFiniteActivatedScaleCount = validationNonFiniteActivatedScaleCount[0];
        partial.nonFiniteRotationCount = validationNonFiniteRotationCount[0];
        partial.nonFiniteSHCount = validationNonFiniteSHCount[0];
        partial.firstNonFiniteGaussianIndex = validationFirstNonFiniteGaussianIndex[0];
        gaussianValidationPartials[groupID.x] = partial;
    }
}
