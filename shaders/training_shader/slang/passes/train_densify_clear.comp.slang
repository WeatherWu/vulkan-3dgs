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
RWStructuredBuffer<uint4> densificationCounters;

[[vk::push_constant]]
ConstantBuffer<TrainingDensificationPushConstants> densifyParams;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    if (dispatchThreadID.x > 0) {
        return;
    }
    densificationCounters[0] = uint4(0, 0, 0, 0);
}
