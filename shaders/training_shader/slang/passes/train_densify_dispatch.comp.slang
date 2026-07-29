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

[[vk::binding(19, 0)]]
RWStructuredBuffer<uint> densificationCounters;

[[vk::push_constant]]
ConstantBuffer<TrainingDensificationPushConstants> densifyParams;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    if (dispatchThreadID.x != 0u) {
        return;
    }

    uint outputCount = min(densificationCounters[1], densifyParams.maxGaussianCount);
    densificationCounters[12] = (outputCount + 255u) / 256u;
    densificationCounters[13] = 1u;
    densificationCounters[14] = 1u;
}
