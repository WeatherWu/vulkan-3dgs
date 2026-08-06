#include "../common/train_prefix_common.slang"

[[vk::binding(32, 0)]]
RWStructuredBuffer<uint2> gaussianTileRanges;

[[vk::binding(33, 0)]]
StructuredBuffer<uint> prefixScratch;

[[vk::push_constant]]
ConstantBuffer<TrainingPrefixPushConstants> prefix;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;
    if (index >= prefix.elementCount) {
        return;
    }
    gaussianTileRanges[index].x += prefixScratch[prefix.inputOffset + index / 256u];
}
