#include "../common/train_prefix_common.slang"

[[vk::binding(33, 0)]]
RWStructuredBuffer<uint> prefixScratch;

[[vk::push_constant]]
ConstantBuffer<TrainingPrefixPushConstants> prefix;

groupshared uint scanValues[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID,
          uint3 groupThreadID : SV_GroupThreadID,
          uint3 groupID : SV_GroupID) {
    uint localIndex = groupThreadID.x;
    uint index = dispatchThreadID.x;
    scanValues[localIndex] = index < prefix.elementCount
        ? prefixScratch[prefix.inputOffset + index]
        : 0u;
    GroupMemoryBarrierWithGroupSync();

    for (uint offset = 1u; offset < 256u; offset <<= 1u) {
        uint addend = localIndex >= offset ? scanValues[localIndex - offset] : 0u;
        GroupMemoryBarrierWithGroupSync();
        scanValues[localIndex] += addend;
        GroupMemoryBarrierWithGroupSync();
    }

    if (index < prefix.elementCount) {
        prefixScratch[prefix.inputOffset + index] = localIndex == 0u
            ? 0u
            : scanValues[localIndex - 1u];
    }
    if (localIndex == 255u) {
        prefixScratch[prefix.outputOffset + groupID.x] = scanValues[255];
    }
}
