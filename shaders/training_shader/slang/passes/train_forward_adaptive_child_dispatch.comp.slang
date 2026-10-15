#include "../common/train_forward_tile_common.slang"

[[vk::binding(9, 0)]]
StructuredBuffer<uint4> counters;

[[vk::binding(41, 0)]]
RWStructuredBuffer<uint4> childDispatch;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    uint parentCount = tileCountX() * tileCountY();
    uint selectedCount = min(counters[0].z, parentCount);
    uint childGroups = selectedCount * 4u;
    // 65535 is guaranteed by Vulkan for both X and Y workgroup-count limits.
    // Padded groups in the final row are rejected before reading activeParents.
    childDispatch[0] = uint4(min(childGroups, TRAIN_ADAPTIVE_CHILD_DISPATCH_WIDTH),
                            max(childGroups / TRAIN_ADAPTIVE_CHILD_DISPATCH_WIDTH +
                                uint(childGroups % TRAIN_ADAPTIVE_CHILD_DISPATCH_WIDTH != 0u), 1u),
                            1u, selectedCount);
}
