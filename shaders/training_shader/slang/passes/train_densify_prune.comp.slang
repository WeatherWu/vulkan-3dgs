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

[[vk::binding(19, 0)]]
RWStructuredBuffer<uint> densificationCounters;

[[vk::binding(17, 0)]]
RWStructuredBuffer<GaussianTrainParam> densifiedParams;

[[vk::binding(18, 0)]]
RWStructuredBuffer<AdamState> densifiedAdamStates;

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

bool finiteGaussianForDensify(GaussianTrainParam gaussian) {
    bool finite = all(isfinite(gaussian.positionOpacity)) &&
                  all(isfinite(gaussian.scale)) &&
                  all(isfinite(gaussian.rotation));
    for (uint i = 0; i < 16; ++i) {
        finite = finite && all(isfinite(gaussian.sh[i]));
    }
    return finite;
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

float averageScreenGrad(GaussianDensificationState densification) {
    float grad = densification.screenGradCount > 0
        ? densification.screenGradSum / float(densification.screenGradCount)
        : 0.0;
    return isnan(grad) ? 0.0 : grad;
}

bool shouldDensify(GaussianTrainParam gaussian,
                   GaussianDensificationState densification) {
    if (!finiteGaussianForDensify(gaussian)) {
        return false;
    }

    float grad = averageScreenGrad(densification);
    if (grad < densifyParams.gradThreshold) {
        return false;
    }

    float maxScale = maxActivatedScale(gaussian);
    return isfinite(maxScale) && maxScale > 0.0;
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

GaussianTrainParam makeSplitChild(GaussianTrainParam source, uint sourceIndex, uint childIndex) {
    GaussianTrainParam child = source;
    float3 scale = trainScaleActivationForDensify(source.scale.xyz);
    float3 randomOffset = float3(
        randomNormal(sourceIndex * 17u + childIndex, 0u),
        randomNormal(sourceIndex * 17u + childIndex, 1u),
        randomNormal(sourceIndex * 17u + childIndex, 2u)
    );

    float4 normalizedRotation = source.rotation / max(length(source.rotation), 1e-12);
    float3 offset = mul(trainQuatToMat3ForDensify(normalizedRotation), randomOffset * scale);
    child.positionOpacity.xyz = source.positionOpacity.xyz + offset;

    float childScaleDivisor = 0.8 * max(float(densifyParams.splitChildren), 1.0);
    child.scale.xyz = log(scale / childScaleDivisor);
    return child;
}

GaussianDensificationState zeroDensificationState() {
    GaussianDensificationState state;
    state.screenGradSum = 0.0;
    state.screenGradCount = 0;
    clearDensificationMaxScreenRadius(state);
    state.padding0 = 0;
    return state;
}

void incrementCounter(inout uint counter) {
    uint ignored = 0;
    InterlockedAdd(counter, 1u, ignored);
}

void appendCandidate(GaussianTrainParam gaussian,
                     AdamState adamState,
                     GaussianDensificationState densificationState) {
    uint candidateIndex = 0;
    InterlockedAdd(densificationCounters[0], 1u, candidateIndex);
    if (candidateIndex >= densifyParams.maxGaussianCount) {
        return;
    }

    bool pruneByNonFinite = !finiteGaussianForDensify(gaussian);
    bool pruneByOpacity = activatedOpacity(gaussian) < densifyParams.minOpacity;
    bool pruneByScreenSize = densifyParams.pruneByScreenSize != 0u &&
        loadDensificationMaxScreenRadius(densificationState) > densifyParams.screenSizePruneThreshold;
    bool pruneByWorldSize = densifyParams.pruneByScreenSize != 0u &&
        maxActivatedScale(gaussian) > densifyParams.worldSizePruneThreshold * densifyParams.sceneExtent;

    if (pruneByOpacity) {
        incrementCounter(densificationCounters[2]);
    }
    if (pruneByScreenSize) {
        incrementCounter(densificationCounters[3]);
    }
    if (pruneByWorldSize) {
        incrementCounter(densificationCounters[4]);
    }
    if (pruneByNonFinite || pruneByOpacity || pruneByScreenSize || pruneByWorldSize) {
        incrementCounter(densificationCounters[8]);
        return;
    }

    uint outputIndex = 0;
    InterlockedAdd(densificationCounters[1], 1u, outputIndex);
    if (outputIndex < densifyParams.maxGaussianCount) {
        densifiedParams[outputIndex] = gaussian;
        densifiedAdamStates[outputIndex] = adamState;
    }
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

    bool split = shouldSplit(gaussian, densification);
    bool clone = shouldClone(gaussian, densification);
    if (split) {
        incrementCounter(densificationCounters[7]);
        uint children = max(densifyParams.splitChildren, 2u);
        AdamState childAdamState = makeZeroAdamState();
        GaussianDensificationState childDensificationState = zeroDensificationState();
        for (uint childIndex = 0; childIndex < children; ++childIndex) {
            appendCandidate(makeSplitChild(gaussian, gaussianIndex, childIndex),
                            childAdamState,
                            childDensificationState);
        }
    } else if (clone) {
        incrementCounter(densificationCounters[6]);
        appendCandidate(makeClone(gaussian), adamState, zeroDensificationState());
        appendCandidate(makeClone(gaussian), makeZeroAdamState(), zeroDensificationState());
    } else {
        incrementCounter(densificationCounters[5]);
        appendCandidate(makeClone(gaussian), adamState, zeroDensificationState());
    }

}
