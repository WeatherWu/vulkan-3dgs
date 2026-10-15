#include "../common/train_forward_tile_common.slang"

[[vk::binding(3, 0)]]
StructuredBuffer<ProjectedGaussian> projectedGaussians;

[[vk::binding(5, 0)]]
StructuredBuffer<uint4> parentRanges;

[[vk::binding(24, 0)]]
StructuredBuffer<uint> sortedParentItems;

[[vk::binding(35, 0)]]
RWStructuredBuffer<uint4> childCounts;

[[vk::binding(36, 0)]]
RWStructuredBuffer<uint> childMasks;

groupshared uint4 sharedChildCounts[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 groupThreadID : SV_GroupThreadID,
          uint3 groupID : SV_GroupID) {
    uint parentIndex = groupID.y * tileCountX() + groupID.x;
    uint parentCount = tileCountX() * tileCountY();
    if (parentIndex >= parentCount) return;

    uint4 range = parentRanges[parentIndex * 4u];
    // This dispatch temporarily receives the configured coarse threshold in
    // tileItemCount. The parent range supplies its own item count and offset.
    if (range.y < max(params.tileItemCount, 1u)) {
        if (groupThreadID.x == 0u) childCounts[parentIndex] = uint4(0u);
        return;
    }

    uint2 parentOrigin = uint2(parentIndex % tileCountX(),
                               parentIndex / tileCountX()) * 16u;
    uint4 localCounts = uint4(0u);
    for (uint item = groupThreadID.x; item < range.y; item += 256u) {
        uint itemIndex = range.x + item;
        uint gaussianIndex = sortedParentItems[itemIndex];
        uint mask = 0u;
        if (gaussianIndex < params.gaussianCount) {
            ProjectedGaussian gaussian = projectedGaussians[gaussianIndex];
            for (uint child = 0u; child < 4u; ++child) {
                uint2 origin = parentOrigin +
                    uint2(child & 1u, child >> 1u) * 8u;
                if (trainGaussianIntersectsPixelRect(gaussian, origin, 8u)) {
                    mask |= 1u << child;
                    localCounts[child] += 1u;
                }
            }
        }
        childMasks[itemIndex] = mask;
    }

    sharedChildCounts[groupThreadID.x] = localCounts;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (groupThreadID.x < stride) {
            sharedChildCounts[groupThreadID.x] +=
                sharedChildCounts[groupThreadID.x + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (groupThreadID.x == 0u) childCounts[parentIndex] = sharedChildCounts[0];
}
