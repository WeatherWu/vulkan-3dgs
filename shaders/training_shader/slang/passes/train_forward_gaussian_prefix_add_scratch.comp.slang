#include "../common/train_prefix_common.slang"

[[vk::binding(33, 0)]]
RWStructuredBuffer<uint> prefixScratch;

[[vk::push_constant]]
ConstantBuffer<TrainingPrefixPushConstants> prefix;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;
    if (index >= prefix.elementCount) {
        return;
    }
    prefixScratch[prefix.inputOffset + index] +=
        prefixScratch[prefix.parentOffset + index / 256u];
}
