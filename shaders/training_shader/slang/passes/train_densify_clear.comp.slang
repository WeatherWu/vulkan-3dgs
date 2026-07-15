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
    if (dispatchThreadID.x > 0) {
        return;
    }
    for (uint i = 0; i < 16; ++i) {
        densificationCounters[i] = 0u;
    }
}
