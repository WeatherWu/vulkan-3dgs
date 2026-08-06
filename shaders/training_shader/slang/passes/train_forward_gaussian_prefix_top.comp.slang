#include "../common/train_prefix_common.slang"

[[vk::binding(9, 0)]]
RWStructuredBuffer<uint4> counters;

[[vk::binding(33, 0)]]
RWStructuredBuffer<uint> prefixScratch;

[[vk::push_constant]]
ConstantBuffer<TrainingPrefixPushConstants> prefix;

groupshared uint scanValues[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 groupThreadID : SV_GroupThreadID) {
    uint localIndex = groupThreadID.x;
    scanValues[localIndex] = localIndex < prefix.elementCount
        ? prefixScratch[prefix.inputOffset + localIndex]
        : 0u;
    GroupMemoryBarrierWithGroupSync();

    for (uint offset = 1u; offset < 256u; offset <<= 1u) {
        uint addend = localIndex >= offset ? scanValues[localIndex - offset] : 0u;
        GroupMemoryBarrierWithGroupSync();
        scanValues[localIndex] += addend;
        GroupMemoryBarrierWithGroupSync();
    }

    if (localIndex < prefix.elementCount) {
        prefixScratch[prefix.inputOffset + localIndex] = localIndex == 0u
            ? 0u
            : scanValues[localIndex - 1u];
    }
    if (localIndex == 0u) {
        uint total = 0u;
        if (prefix.elementCount > 0u) {
            total = scanValues[prefix.elementCount - 1u];
        }
        counters[0].x = total;
    }
}
