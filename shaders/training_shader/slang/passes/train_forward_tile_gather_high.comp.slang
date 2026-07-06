#include "../common/train_common.slang"

[[vk::binding(21, 0)]]
StructuredBuffer<uint> tileKeyHigh;

[[vk::binding(22, 0)]]
StructuredBuffer<uint> tileSortIndices;

[[vk::binding(23, 0)]]
RWStructuredBuffer<uint> tileSortScratch;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;
    if (index >= params.tileItemCount) {
        return;
    }

    uint sourceIndex = tileSortIndices[index];
    tileSortScratch[index] = tileKeyHigh[sourceIndex];
}
