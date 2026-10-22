#include "../common/train_checkpoint_layout.slang"
struct PrefixParams {
    uint mode; uint count; uint input; uint output;
    uint parent; uint maxGroupsX; uint maxGroupsY; uint padding;
};
[[vk::push_constant]] ConstantBuffer<PrefixParams> prefix;
groupshared uint2 values[256];
[shader("compute")]
[numthreads(256,1,1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint i = gid.x * 256u + tid.x;
    if (prefix.mode == 5u) {
        if (i == 0u) {
            if (checkpointControl[1].w != checkpointControl[3].x) checkpointError(32u);
            uint flags = checkpointControl[1].z;
            if (flags != 0u) {
                checkpointControl[2] = uint4(0u, 1u, 1u, checkpointControl[1].y);
                checkpointControl[4].w = 1u;
            } else {
                checkpointControl[4].x = 0u;
                checkpointControl[4].w = 0u;
            }
            checkpointControl[5].x += uint(flags != 0u);
            checkpointControl[5].y |= flags;
            checkpointControl[5].z += 1u;
        }
        return;
    }
    if (prefix.mode == 3u) {
        if (i < prefix.count)
            checkpointScratch[prefix.input + i] += checkpointScratch[prefix.parent + i / 256u];
        return;
    }
    if (prefix.mode == 4u) {
        if (i < prefix.count) {
            uint2 add = checkpointScratch[prefix.parent + i / 256u];
            checkpointTiles[i].range.z += add.x;
            checkpointTiles[i].activity.z += add.y;
        }
        return;
    }
    uint2 value = uint2(0u);
    if (i < prefix.count) {
        if (prefix.mode == 0u)
            value = uint2(checkpointTiles[i].range.w, checkpointTiles[i].activity.y);
        else
            value = checkpointScratch[prefix.input + i];
    }
    values[tid.x] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 1u; offset < 256u; offset <<= 1u) {
        uint2 add = uint2(0u);
        if (tid.x >= offset) add = values[tid.x - offset];
        GroupMemoryBarrierWithGroupSync();
        values[tid.x] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    if (i < prefix.count) {
        uint2 exclusive = uint2(0u);
        if (tid.x > 0u) exclusive = values[tid.x - 1u];
        if (prefix.mode == 0u) {
            checkpointTiles[i].range.z = exclusive.x;
            checkpointTiles[i].activity.z = exclusive.y;
        } else {
            checkpointScratch[prefix.input + i] = exclusive;
        }
    }
    if (tid.x == 255u) {
        if (prefix.mode != 2u) {
            checkpointScratch[prefix.output + gid.x] = values[255];
        } else {
            uint2 total = values[255];
            checkpointControl[1].xy = total;
            if (total.x > checkpointControl[0].x) checkpointError(2u);
            if (total.y > checkpointControl[0].y) checkpointError(4u);
            uint groups = total.y / 4u + uint(total.y % 4u != 0u);
            uint x = min(groups, prefix.maxGroupsX);
            uint y = x == 0u ? 1u : groups / x + uint(groups % x != 0u);
            if (y > prefix.maxGroupsY) checkpointError(16u);
            checkpointControl[2] = uint4(x, y, 1u, total.y);
        }
    }
}
