#include "../common/train_forward_tile_common.slang"

[[vk::binding(9, 0)]]
RWStructuredBuffer<uint4> counters;

[[vk::binding(39, 0)]]
RWStructuredBuffer<uint> childBlockOffsets;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    uint parentCount = tileCountX() * tileCountY();
    uint blockCount = (parentCount + 255u) / 256u;
    uint childBudget = params.tileItemCount / 4u;
    uint running = 0u;
    for (uint blockIndex = 0u; blockIndex < blockCount; ++blockIndex) {
        uint blockTotal = childBlockOffsets[blockIndex];
        childBlockOffsets[blockIndex] = running;
        if (running > childBudget) {
            running = childBudget + 1u;
        } else if (blockTotal > childBudget - running) {
            running = childBudget + 1u;
        } else {
            running += blockTotal;
        }
    }
    counters[0].y = 0u;
    counters[0].z = 0u;
}
