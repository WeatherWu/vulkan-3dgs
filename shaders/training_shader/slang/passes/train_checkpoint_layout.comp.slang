#include "../common/train_forward_common.slang"
#include "../common/train_checkpoint_layout.slang"
[[vk::binding(5, 0)]] StructuredBuffer<uint4> tileRanges;
[[vk::binding(14, 0)]] StructuredBuffer<PixelBlendState> pixelBlendStates;
groupshared uint maxima[256];
groupshared uint stateCounts[256];
[shader("compute")]
[numthreads(16,16,1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint tile = gid.y * tileCountX() + gid.x;
    uint lane = tid.y * 16u + tid.x;
    uint2 pixel = gid.xy * 16u + tid.xy;
    uint4 range = tileRanges[tile];
    bool safeRange = range.x <= params.tileItemCount &&
                     range.y <= params.tileItemCount - min(range.x, params.tileItemCount);
    if (!safeRange && lane == 0u) checkpointError(1u);
    uint count = safeRange ? range.y : 0u;
    uint p = 0u;
    if (pixel.x < params.width && pixel.y < params.height)
        p = min(pixelBlendStates[pixelIndex(pixel)].processedCount, count);
    maxima[lane] = p;
    stateCounts[lane] = p == 0u ? 0u : (p - 1u) / 32u;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            maxima[lane] = max(maxima[lane], maxima[lane + stride]);
            stateCounts[lane] += stateCounts[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u) {
        CheckpointTile meta;
        meta.range = uint4(range.x, count, 0u, count == 0u ? 0u : (count - 1u) / 32u);
        meta.activity = uint4(maxima[0], maxima[0] / 32u + uint(maxima[0] % 32u != 0u), 0u, 0u);
        checkpointTiles[tile] = meta;
        uint ignored;
        InterlockedAdd(checkpointControl[1].w, stateCounts[0], ignored);
    }
}
