#ifndef TRAIN_CHECKPOINT_LAYOUT_SLANG
#define TRAIN_CHECKPOINT_LAYOUT_SLANG
struct CheckpointTile {
    uint4 range;    // raw start/count, boundary base/count
    uint4 activity; // max processed, active buckets, task base, flags
};
struct CheckpointTask {
    uint tile;
    uint bucket;
    uint slot;
    uint itemCount;
};
[[vk::binding(43, 0)]] RWStructuredBuffer<CheckpointTile> checkpointTiles;
[[vk::binding(44, 0)]] RWStructuredBuffer<uint2> checkpointScratch;
[[vk::binding(50, 0)]] RWStructuredBuffer<uint4> checkpointControl;
void checkpointError(uint flag) {
    uint ignored;
    InterlockedOr(checkpointControl[1].z, flag, ignored);
}
#endif
