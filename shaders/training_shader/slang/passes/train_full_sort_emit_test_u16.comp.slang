// Depth-ordered emit. gaussianCount is the GPU-produced participating count.
#include "../common/train_forward_tile_common.slang"
[[vk::binding(3, 0)]] StructuredBuffer<ProjectedGaussian> projected;
[[vk::binding(21, 0)]] RWStructuredBuffer<uint16_t> tileKeys;
[[vk::binding(24, 0)]] RWStructuredBuffer<uint> gaussianIDs;
[[vk::binding(32, 0)]] StructuredBuffer<uint2> orderedRanges;
[[vk::binding(62, 0)]] StructuredBuffer<uint> visibleIDs;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread : SV_DispatchThreadID) {
    uint rank = thread.x;
    if (rank >= params.gaussianCount) return;
    uint id = visibleIDs[rank];
    ProjectedGaussian gaussian = projected[id];
    GaussianTileCullingData culling = trainPrepareGaussianTileCulling(gaussian);
    if (!culling.active || !culling.bounds.valid) return;
    uint2 itemRange = orderedRanges[rank];
    uint output = itemRange.x;
    uint end = min(itemRange.x + itemRange.y, params.tileItemCount);
    bool rows = trainGaussianScansRows(culling);
    uint first = rows ? culling.bounds.minTile.y : culling.bounds.minTile.x;
    uint last = rows ? culling.bounds.maxTileExclusive.y : culling.bounds.maxTileExclusive.x;
    for (uint strip = first; strip < last; ++strip) {
        TileScanRange scan = trainTileRangeForStrip(culling, strip, rows);
        for (uint item = scan.minimum; item < scan.maximumExclusive; ++item) {
            if (output < end) {
                tileKeys[output] = uint16_t(trainTileIndex(trainTileFromScan(strip, item, rows)));
                gaussianIDs[output] = id;
            }
            ++output;
        }
    }
}
