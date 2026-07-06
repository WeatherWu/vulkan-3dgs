#include "../common/train_common.slang"

[[vk::binding(4, 0)]]
StructuredBuffer<uint> tileItems;

[[vk::binding(22, 0)]]
StructuredBuffer<uint> tileSortIndices;

[[vk::binding(24, 0)]]
RWStructuredBuffer<uint> tileItemsSorted;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;
    if (index >= params.tileItemCount) {
        return;
    }

    uint lowSortedIndex = tileSortIndices[index];
    tileItemsSorted[index] = tileItems[lowSortedIndex];
}
