#include "train_common.slang"

[[vk::binding(0, 0)]]
RWStructuredBuffer<GaussianTrainParam> gaussianParams;

[[vk::binding(1, 0)]]
StructuredBuffer<GaussianGrad> gaussianGrads;

[[vk::binding(2, 0)]]
RWStructuredBuffer<AdamState> adamStates;

[[vk::binding(9, 0)]]
RWStructuredBuffer<uint4> counters;

static const float OPTIMIZER_LEARNING_RATE = 1e-3;
static const float OPTIMIZER_BETA1 = 0.9;
static const float OPTIMIZER_BETA2 = 0.999;
static const float OPTIMIZER_EPSILON = 1e-8;
static const float OPTIMIZER_GRAD_CLIP = 1e3;

float4 clipGrad(float4 grad) {
    return clamp(grad, float4(-OPTIMIZER_GRAD_CLIP), float4(OPTIMIZER_GRAD_CLIP));
}

float4 safeNormalizeQuaternion(float4 rotation) {
    float lengthSquared = dot(rotation, rotation);
    if (lengthSquared <= 1e-16) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    return rotation * rsqrt(lengthSquared);
}

float4 adamUpdate(inout float4 firstMoment, inout float4 secondMoment, float4 grad, float step) {
    grad = clipGrad(grad);
    firstMoment = OPTIMIZER_BETA1 * firstMoment + (1.0 - OPTIMIZER_BETA1) * grad;
    secondMoment = OPTIMIZER_BETA2 * secondMoment + (1.0 - OPTIMIZER_BETA2) * grad * grad;

    float firstMomentCorrection = 1.0 - pow(OPTIMIZER_BETA1, step);
    float secondMomentCorrection = 1.0 - pow(OPTIMIZER_BETA2, step);
    float4 correctedFirstMoment = firstMoment / max(firstMomentCorrection, 1e-8);
    float4 correctedSecondMoment = secondMoment / max(secondMomentCorrection, 1e-8);
    return OPTIMIZER_LEARNING_RATE * correctedFirstMoment / (sqrt(correctedSecondMoment) + float4(OPTIMIZER_EPSILON));
}

void updateGaussianParam(inout GaussianTrainParam gaussian,
                         inout AdamState adamState,
                         GaussianGrad grad,
                         float step) {
    gaussian.positionOpacity -= adamUpdate(adamState.firstMoment.positionOpacity,
                                           adamState.secondMoment.positionOpacity,
                                           grad.positionOpacity,
                                           step);
    gaussian.scale -= adamUpdate(adamState.firstMoment.scale,
                                 adamState.secondMoment.scale,
                                 grad.scale,
                                 step);
    gaussian.rotation -= adamUpdate(adamState.firstMoment.rotation,
                                    adamState.secondMoment.rotation,
                                    grad.rotation,
                                    step);

    gaussian.positionOpacity.w = clamp(gaussian.positionOpacity.w, 0.0, 1.0);
    gaussian.scale.xyz = max(abs(gaussian.scale.xyz), float3(1e-6));
    gaussian.rotation = safeNormalizeQuaternion(gaussian.rotation);

    for (uint i = 0; i < 16; ++i) {
        gaussian.sh[i] -= adamUpdate(adamState.firstMoment.sh[i],
                                     adamState.secondMoment.sh[i],
                                     grad.sh[i],
                                     step);
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
