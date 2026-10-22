// S4-3 private synthetic diagnostic only. Never selected by a training renderer.
[[vk::binding(0, 0)]] StructuredBuffer<uint> originalHigh;
[[vk::binding(1, 0)]] StructuredBuffer<uint> sortedIndices;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> highKeys;
[[vk::binding(3, 0)]] StructuredBuffer<uint> control; // GPU-written [count,tileCount]
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> ranges;
struct DiagnosticPush { uint capacity; uint mode; };
[[vk::push_constant]] DiagnosticPush diagnostic;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;
    uint count = control[0];
    if (count > diagnostic.capacity) return;
    uint tileCount = control[1];
    if (diagnostic.mode == 0u) {
        if (index < count) highKeys[index] = originalHigh[sortedIndices[index]];
    } else if (diagnostic.mode == 1u) {
        if (index > count) return;
        uint currentTile = index == count ? tileCount : originalHigh[sortedIndices[index]];
        uint previousTile = index == 0u ? tileCount : originalHigh[sortedIndices[index - 1u]];
        // Each scalar address has one writer. Empty ranges were cleared in restore.
        if (index < count && currentTile < tileCount && currentTile != previousTile)
            ranges[2u * currentTile] = index;
        if (index > 0u && previousTile < tileCount && previousTile != currentTile)
            ranges[2u * previousTile + 1u] = index;
    } else if (diagnostic.mode == 2u && index < tileCount) {
        ranges[2u * index + 1u] -= ranges[2u * index];
    }
}
