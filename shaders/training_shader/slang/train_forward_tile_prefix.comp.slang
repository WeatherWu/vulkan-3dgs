#include "train_forward_tile_common.slang"

[[vk::binding(5, 0)]]
RWStructuredBuffer<uint4> tileRanges;

[[vk::binding(9, 0)]]
RWStructuredBuffer<uint4> counters;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint runningStart = 0;
    uint tileCount = trainTileCount();

    for (uint tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
        uint4 range = tileRanges[tileIndex];
        uint count = range.y;
        tileRanges[tileIndex] = uint4(runningStart, count, 0, 0);
        runningStart += count;
    }

    counters[0].x = runningStart;
}
