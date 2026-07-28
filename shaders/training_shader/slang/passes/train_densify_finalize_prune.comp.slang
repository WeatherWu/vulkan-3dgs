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
RWStructuredBuffer<uint> densificationCounters;

[[vk::binding(25, 0)]]
StructuredBuffer<GaussianTrainParam> candidateParams;

[[vk::binding(26, 0)]]
StructuredBuffer<AdamState> candidateAdamStates;

[[vk::binding(27, 0)]]
StructuredBuffer<GaussianDensificationState> candidateDensificationStates;

[[vk::push_constant]]
ConstantBuffer<TrainingDensificationPushConstants> densifyParams;

float trainOpacityActivationForDensify(float rawOpacity) {
    return 1.0 / (1.0 + exp(-rawOpacity));
}

float3 trainScaleActivationForDensify(float3 rawScale) {
    return exp(rawScale);
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

float activatedOpacity(GaussianTrainParam gaussian) {
    return trainOpacityActivationForDensify(gaussian.positionOpacity.w);
}

float maxActivatedScale(GaussianTrainParam gaussian) {
    float3 scale = trainScaleActivationForDensify(gaussian.scale.xyz);
    return max(max(scale.x, scale.y), scale.z);
}

void incrementCounter(inout uint counter) {
    uint ignored = 0;
    InterlockedAdd(counter, 1u, ignored);
}

void appendFinal(GaussianTrainParam gaussian, AdamState adamState) {
    uint outputIndex = 0;
    InterlockedAdd(densificationCounters[1], 1u, outputIndex);
    if (outputIndex >= densifyParams.maxGaussianCount) {
        return;
    }

    densifiedParams[outputIndex] = gaussian;
    densifiedAdamStates[outputIndex] = adamState;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint candidateIndex = dispatchThreadID.x;
    uint candidateCount = min(densificationCounters[0], densifyParams.maxGaussianCount);
    if (candidateIndex >= candidateCount) {
        return;
    }

    GaussianTrainParam gaussian = candidateParams[candidateIndex];
    AdamState adamState = candidateAdamStates[candidateIndex];
    GaussianDensificationState densification = candidateDensificationStates[candidateIndex];

    bool pruneByNonFinite = !finiteGaussianForDensify(gaussian);
    bool pruneByOpacity = activatedOpacity(gaussian) < densifyParams.minOpacity;
    bool pruneByScreenSize = densifyParams.pruneByScreenSize != 0u &&
        loadDensificationMaxScreenRadius(densification) > densifyParams.screenSizePruneThreshold;
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

    appendFinal(gaussian, adamState);
}
