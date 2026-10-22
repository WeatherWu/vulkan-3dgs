#include "../common/train_forward_common.slang"
#include "../common/train_checkpoint_layout.slang"
[[vk::binding(49, 0)]] RWStructuredBuffer<CheckpointTask> checkpointTasks;
groupshared uint initialErrors;
[shader("compute")]
[numthreads(256,1,1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    if (tid.x == 0u) InterlockedOr(checkpointControl[1].z, 0u, initialErrors);
    GroupMemoryBarrierWithGroupSync();
    if (initialErrors != 0u) return;
    uint tile = gid.y * tileCountX() + gid.x;
    CheckpointTile meta = checkpointTiles[tile];
    uint rawBuckets = meta.range.y / 32u + uint(meta.range.y % 32u != 0u);
    for (uint bucket = tid.x; bucket < meta.activity.y; bucket += 256u) {
        uint output = meta.activity.z + bucket;
        uint slot = bucket + 1u == rawBuckets ? 0xffffffffu : meta.range.z + bucket;
        if (output >= checkpointControl[0].y ||
            (slot != 0xffffffffu && slot >= checkpointControl[0].x)) {
            checkpointError(4u);
            continue;
        }
        CheckpointTask task;
        task.tile = tile; task.bucket = bucket; task.slot = slot;
        task.itemCount = min(32u, meta.range.y - bucket * 32u);
        checkpointTasks[output] = task;
    }
}
