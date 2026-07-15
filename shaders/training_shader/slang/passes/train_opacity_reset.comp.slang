#include "../common/train_common.slang"

struct TrainingDensificationPushConstants {
    uint gaussianCount;
    uint maxGaussianCount;
    uint splitChildren;
    uint pruneByScreenSize;
    float sceneExtent;
    float gradThreshold;
    float minOpacity;
    float percentDense;
    float screenSizePruneThreshold;
    uint randomSeed;
    uint resetOpacity;
    float worldSizePruneThreshold;
    uint padding1;
};

[[vk::binding(17, 0)]]
RWStructuredBuffer<GaussianTrainParam> densifiedParams;

[[vk::binding(18, 0)]]
RWStructuredBuffer<AdamState> densifiedAdamStates;

[[vk::binding(19, 0)]]
StructuredBuffer<uint> densificationCounters;

[[vk::push_constant]]
ConstantBuffer<TrainingDensificationPushConstants> densifyParams;

float trainOpacityActivationForReset(float rawOpacity) {
    return 1.0 / (1.0 + exp(-rawOpacity));
}

float trainOpacityInverseActivationForReset(float opacity) {
    float clampedOpacity = clamp(opacity, 1e-6, 1.0 - 1e-6);
    return log(clampedOpacity / (1.0 - clampedOpacity));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    if (densifyParams.resetOpacity == 0u) {
        return;
    }

    uint index = dispatchThreadID.x;
    uint finalCount = min(densificationCounters[1], densifyParams.maxGaussianCount);
    if (index >= finalCount) {
        return;
    }

    GaussianTrainParam gaussian = densifiedParams[index];
    float opacity = min(trainOpacityActivationForReset(gaussian.positionOpacity.w), 0.01);
    gaussian.positionOpacity.w = trainOpacityInverseActivationForReset(opacity);
    densifiedParams[index] = gaussian;

    AdamState adamState = densifiedAdamStates[index];
    adamState.firstMoment.positionOpacity.w = 0.0;
    adamState.secondMoment.positionOpacity.w = 0.0;
    densifiedAdamStates[index] = adamState;
}
