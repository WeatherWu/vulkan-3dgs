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

[[vk::binding(0, 0)]]
StructuredBuffer<GaussianTrainParam> gaussianParams;

[[vk::binding(2, 0)]]
StructuredBuffer<AdamState> adamStates;

[[vk::binding(16, 0)]]
RWStructuredBuffer<GaussianDensificationState> densificationStates;

[[vk::binding(17, 0)]]
RWStructuredBuffer<GaussianTrainParam> densifiedParams;

[[vk::binding(18, 0)]]
RWStructuredBuffer<AdamState> densifiedAdamStates;

[[vk::binding(19, 0)]]
RWStructuredBuffer<uint4> densificationCounters;

[[vk::push_constant]]
ConstantBuffer<TrainingDensificationPushConstants> densifyParams;

float trainOpacityActivationForDensify(float rawOpacity) {
    return 1.0 / (1.0 + exp(-rawOpacity));
}

float trainOpacityInverseActivationForDensify(float opacity) {
    float clampedOpacity = clamp(opacity, 1e-6, 1.0 - 1e-6);
    return log(clampedOpacity / (1.0 - clampedOpacity));
}

float3 trainScaleActivationForDensify(float3 rawScale) {
    return exp(rawScale);
}

float3x3 trainQuatToMat3ForDensify(float4 q) {
    return float3x3(
        1.0 - 2.0 * q.y * q.y - 2.0 * q.z * q.z,
        2.0 * q.x * q.y - 2.0 * q.z * q.w,
        2.0 * q.x * q.z + 2.0 * q.y * q.w,

        2.0 * q.x * q.y + 2.0 * q.z * q.w,
        1.0 - 2.0 * q.x * q.x - 2.0 * q.z * q.z,
        2.0 * q.y * q.z - 2.0 * q.x * q.w,

        2.0 * q.x * q.z - 2.0 * q.y * q.w,
        2.0 * q.y * q.z + 2.0 * q.x * q.w,
        1.0 - 2.0 * q.x * q.x - 2.0 * q.y * q.y
    );
}

uint hashUint(uint value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

float randomSigned(uint seed, uint axis) {
    uint value = hashUint(seed ^ (axis * 0x9e3779b9u) ^ densifyParams.randomSeed);
    return (float(value & 0x00ffffffu) / float(0x00ffffffu)) * 2.0 - 1.0;
}

float randomUnit(uint seed, uint axis) {
    uint value = hashUint(seed ^ (axis * 0x85ebca6bu) ^ densifyParams.randomSeed);
    return max(float(value & 0x00ffffffu) / float(0x01000000u), 1e-6);
}

float randomNormal(uint seed, uint axis) {
    float u1 = randomUnit(seed, axis * 2u + 0u);
    float u2 = randomUnit(seed, axis * 2u + 1u);
    return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
}

GaussianGrad zeroGaussianGrad() {
    GaussianGrad grad;
    grad.positionOpacity = 0.0;
    grad.scale = 0.0;
    grad.rotation = 0.0;
    for (uint i = 0; i < 16; ++i) {
        grad.sh[i] = 0.0;
    }
    return grad;
}

AdamState makeZeroAdamState() {
    AdamState state;
    state.firstMoment = zeroGaussianGrad();
    state.secondMoment = zeroGaussianGrad();
    return state;
}

GaussianGrad scaleGaussianGrad(GaussianGrad source, float factor) {
    source.positionOpacity *= factor;
    source.scale *= factor;
    source.rotation *= factor;
    for (uint i = 0; i < 16; ++i) {
        source.sh[i] *= factor;
    }
    return source;
}

AdamState scaleAdamState(AdamState state, float factor) {
    state.firstMoment = scaleGaussianGrad(state.firstMoment, factor);
    state.secondMoment = scaleGaussianGrad(state.secondMoment, factor * factor);
    return state;
}

float activatedOpacity(GaussianTrainParam gaussian) {
    return trainOpacityActivationForDensify(gaussian.positionOpacity.w);
}

float maxActivatedScale(GaussianTrainParam gaussian) {
    float3 scale = trainScaleActivationForDensify(gaussian.scale.xyz);
    return max(max(scale.x, scale.y), scale.z);
}

bool shouldPrune(GaussianTrainParam gaussian,
                 GaussianDensificationState densification) {
    if (activatedOpacity(gaussian) < densifyParams.minOpacity) {
        return true;
    }

    if (densifyParams.pruneByScreenSize != 0u &&
        densification.maxScreenRadius > densifyParams.screenSizePruneThreshold) {
        return true;
    }

    if (densifyParams.pruneByScreenSize != 0u &&
        maxActivatedScale(gaussian) > densifyParams.worldSizePruneThreshold * densifyParams.sceneExtent) {
        return true;
    }

    return false;
}

float averageScreenGrad(GaussianDensificationState densification) {
    return densification.screenGradCount > 0
        ? densification.screenGradSum / float(densification.screenGradCount)
        : 0.0;
}

bool shouldDensify(GaussianTrainParam gaussian,
                   GaussianDensificationState densification) {
    float grad = averageScreenGrad(densification);
    if (grad < densifyParams.gradThreshold) {
        return false;
    }

    return maxActivatedScale(gaussian) > 0.0;
}

bool shouldClone(GaussianTrainParam gaussian,
                 GaussianDensificationState densification) {
    if (!shouldDensify(gaussian, densification)) {
        return false;
    }

    return maxActivatedScale(gaussian) <= densifyParams.percentDense * densifyParams.sceneExtent;
}

bool shouldSplit(GaussianTrainParam gaussian,
                 GaussianDensificationState densification) {
    if (!shouldDensify(gaussian, densification)) {
        return false;
    }

    return maxActivatedScale(gaussian) > densifyParams.percentDense * densifyParams.sceneExtent;
}

GaussianTrainParam makeClone(GaussianTrainParam source) {
    return source;
}

GaussianTrainParam applyOpacityReset(GaussianTrainParam source) {
    if (densifyParams.resetOpacity == 0u) {
        return source;
    }

    float opacity = min(activatedOpacity(source), 0.01);
    source.positionOpacity.w = trainOpacityInverseActivationForDensify(opacity);
    return source;
}

GaussianTrainParam makeSplitChild(GaussianTrainParam source, uint sourceIndex, uint childIndex) {
    GaussianTrainParam child = source;
    float3 scale = trainScaleActivationForDensify(source.scale.xyz);
    float3 randomOffset = float3(
        randomNormal(sourceIndex * 17u + childIndex, 0u),
        randomNormal(sourceIndex * 17u + childIndex, 1u),
        randomNormal(sourceIndex * 17u + childIndex, 2u)
    );

    float4 normalizedRotation = normalize(source.rotation);
    float3 offset = mul(trainQuatToMat3ForDensify(normalizedRotation), randomOffset * scale);
    child.positionOpacity.xyz = source.positionOpacity.xyz + offset;

    float childScaleDivisor = 0.8 * max(float(densifyParams.splitChildren), 1.0);
    child.scale.xyz = log(max(scale / childScaleDivisor, float3(1e-6)));
    return applyOpacityReset(child);
}

void appendGaussian(GaussianTrainParam gaussian, AdamState adamState) {
    uint outputIndex = 0;
    InterlockedAdd(densificationCounters[0].x, 1u, outputIndex);
    if (outputIndex >= densifyParams.maxGaussianCount) {
        return;
    }

    densifiedParams[outputIndex] = gaussian;
    densifiedAdamStates[outputIndex] = adamState;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint gaussianIndex = dispatchThreadID.x;
    if (gaussianIndex >= densifyParams.gaussianCount) {
        return;
    }

    GaussianTrainParam gaussian = gaussianParams[gaussianIndex];
    AdamState adamState = adamStates[gaussianIndex];
    GaussianDensificationState densification = densificationStates[gaussianIndex];

    bool prune = shouldPrune(gaussian, densification);
    if (!prune) {
        bool split = shouldSplit(gaussian, densification);
        bool clone = shouldClone(gaussian, densification);
        if (split) {
            uint children = max(densifyParams.splitChildren, 2u);
            AdamState childAdamState = scaleAdamState(adamState, 1.0 / float(children));
            for (uint childIndex = 0; childIndex < children; ++childIndex) {
                appendGaussian(makeSplitChild(gaussian, gaussianIndex, childIndex), childAdamState);
            }
        } else if (clone) {
            AdamState clonedAdamState = scaleAdamState(adamState, 0.5);
            appendGaussian(applyOpacityReset(makeClone(gaussian)), clonedAdamState);
            appendGaussian(applyOpacityReset(makeClone(gaussian)), clonedAdamState);
        } else {
            appendGaussian(applyOpacityReset(makeClone(gaussian)), adamState);
        }
    }

    GaussianDensificationState zeroState;
    zeroState.screenGradSum = 0.0;
    zeroState.screenGradCount = 0;
    zeroState.maxScreenRadius = 0.0;
    zeroState.padding0 = 0;
    densificationStates[gaussianIndex] = zeroState;
}
