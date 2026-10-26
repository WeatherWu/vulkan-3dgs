// Full-sort helper. validationIteration selects flag/scatter/ordered-count operations.
#include "../common/train_common.slang"
[[vk::binding(3, 0)]] StructuredBuffer<ProjectedGaussian> projected;
[[vk::binding(32, 0)]] StructuredBuffer<uint2> originalRanges;
[[vk::binding(60, 0)]] RWStructuredBuffer<uint2> ranks;
[[vk::binding(61, 0)]] RWStructuredBuffer<uint> depthKeys;
[[vk::binding(62, 0)]] RWStructuredBuffer<uint> visibleIDs;
[[vk::binding(63, 0)]] RWStructuredBuffer<uint2> orderedRanges;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread : SV_DispatchThreadID) {
    uint i = thread.x;
    if (i >= params.gaussianCount) return;
    if (params.validationIteration == 0u) {
        ranks[i] = uint2(0u, originalRanges[i].y > 0u ? 1u : 0u);
    } else if (params.validationIteration == 1u) {
        if (originalRanges[i].y == 0u) return;
        uint rank = ranks[i].x;
        uint bits = asuint(projected[i].centerRadius.z);
        depthKeys[rank] = (bits & 0x80000000u) != 0u ? ~bits : bits ^ 0x80000000u;
        visibleIDs[rank] = i;
    } else {
        orderedRanges[i] = uint2(0u, originalRanges[visibleIDs[i]].y);
    }
}
