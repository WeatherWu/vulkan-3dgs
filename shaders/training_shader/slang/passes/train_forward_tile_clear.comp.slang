#include "../common/train_forward_tile_common.slang"

[[vk::binding(5, 0)]]
RWStructuredBuffer<uint4> tileRanges;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint tileIndex = dispatchThreadID.x;
    if (tileIndex >= trainTileCount()) {
        return;
    }

    tileRanges[tileIndex] = uint4(0, 0, 0, 0);
}
