#include "../common/train_backward_common.slang"
#include "../common/train_checkpoint_layout.slang"

[[vk::binding(3,0)]] StructuredBuffer<ProjectedGaussian> projectedGaussians;
[[vk::binding(4,0)]] StructuredBuffer<uint> tileItems;
[[vk::binding(12,0)]] StructuredBuffer<PixelGrad> pixelGrads;
[[vk::binding(13,0)]] RWStructuredBuffer<ProjectedGaussianGrad> projectedGrads;
[[vk::binding(14,0)]] StructuredBuffer<PixelBlendState> pixelBlendStates;
[[vk::binding(45,0)]] StructuredBuffer<float4> bank0;
[[vk::binding(46,0)]] StructuredBuffer<float4> bank1;
[[vk::binding(47,0)]] StructuredBuffer<float4> bank2;
[[vk::binding(48,0)]] StructuredBuffer<float4> bank3;
[[vk::binding(49,0)]] StructuredBuffer<CheckpointTask> checkpointTasks;

groupshared uint subgroupCounter;

void consumerError(uint flag) {
    if (WaveIsFirstLane()) {
        checkpointError(flag);
        uint ignored;
        InterlockedOr(checkpointControl[5].y, flag, ignored);
    }
}

float4 readBoundary(uint slot, uint rank) {
    uint element = slot * 64u + (rank & 63u);
    switch (rank >> 6u) {
        case 0u: return bank0[element];
        case 1u: return bank1[element];
        case 2u: return bank2[element];
        default: return bank3[element];
    }
}

[shader("compute")]
[numthreads(128,1,1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    if (WaveGetLaneCount() != 32u) {
        if (gid.x == 0u && gid.y == 0u) consumerError(64u);
        return;
    }
    if (tid.x == 0u) subgroupCounter = 0u;
    GroupMemoryBarrierWithGroupSync();
    uint subgroup = 0u;
    if (WaveIsFirstLane()) InterlockedAdd(subgroupCounter, 1u, subgroup);
    subgroup = WaveReadLaneAt(subgroup, 0u);
    uint lane = WaveGetLaneIndex();
    uint group = gid.y * checkpointControl[2].x + gid.x;
    uint taskIndex = group * 4u + subgroup;
    if (gid.x == 0u && gid.y == 0u && subgroup == 0u && lane == 0u)
        checkpointControl[5].w = 32u;
    if (taskIndex >= checkpointControl[1].y) return;
    if (taskIndex >= checkpointControl[0].y) { consumerError(128u); return; }
    CheckpointTask task = checkpointTasks[taskIndex];
    if (task.tile >= checkpointControl[0].z) { consumerError(128u); return; }
    CheckpointTile meta = checkpointTiles[task.tile];
    uint rawBuckets = meta.range.y / 32u + uint(meta.range.y % 32u != 0u);
    if (task.bucket >= rawBuckets) { consumerError(128u); return; }
    uint begin = task.bucket * 32u;
    if (task.itemCount == 0u || task.itemCount > 32u || begin >= meta.range.y ||
        task.itemCount > meta.range.y - begin) { consumerError(128u); return; }
    uint end = begin + task.itemCount;
    uint tilesX = (params.width + 15u) / 16u;
    uint2 origin = uint2(task.tile % tilesX, task.tile / tilesX) * 16u;
    uint id = 0xffffffffu;
    uint itemOffset = 0xffffffffu;
    ProjectedGaussian gaussian;
    gaussian.centerRadius = float4(0.0);
    gaussian.conicOpacity = float4(0.0);
    gaussian.color = float4(0.0);
    if (lane < task.itemCount) {
        itemOffset = end - 1u - lane;
        id = tileItems[meta.range.x + itemOffset];
        if (id < params.gaussianCount) gaussian = projectedGaussians[id];
    }
    float2 gradCenter = float2(0.0);
    float4 gradConicOpacity = float4(0.0);
    float3 gradColor = float3(0.0);
    float4 state = float4(0.0, 0.0, 0.0, 1.0);
    float3 pixelGrad = float3(0.0);
    uint processed = 0u;
    float4 prefetchedState = state;
    float3 prefetchedGrad = pixelGrad;
    uint prefetchedProcessed = 0u;
    // Every subgroup lane participates in all shuffles, including bubbles,
    // invalid Gaussians and image-edge pixels. No workgroup barrier in this loop.
    for (uint step = 0u; step < 256u + 31u; ++step) {
        if ((step & 31u) == 0u) {
            prefetchedState = float4(0.0, 0.0, 0.0, 1.0);
            prefetchedGrad = float3(0.0);
            prefetchedProcessed = 0u;
            if (step < 256u) {
                uint rank = step + lane;
                uint2 pixel = origin + uint2(rank & 15u, rank >> 4u);
                if (pixel.x < params.width && pixel.y < params.height) {
                    uint index = pixelIndex(pixel);
                    PixelBlendState blend = pixelBlendStates[index];
                    uint p = min(blend.processedCount, meta.range.y);
                    if (p > begin) {
                        prefetchedProcessed = p;
                        prefetchedGrad = pixelGrads[index].color.xyz;
                        prefetchedState.w = blend.finalTransmittance;
                        if (p > end) {
                            if (task.slot < checkpointControl[0].x)
                                prefetchedState = readBoundary(task.slot, rank);
                            else {
                                prefetchedProcessed = 0u;
                                uint ignored;
                                checkpointError(128u);
                                InterlockedOr(checkpointControl[5].y, 128u, ignored);
                            }
                        }
                    }
                }
            }
        }
        uint peer = lane > 0u ? lane - 1u : 0u;
        float4 shiftedState = WaveReadLaneAt(state, peer);
        float3 shiftedGrad = WaveReadLaneAt(pixelGrad, peer);
        uint shiftedProcessed = WaveReadLaneAt(processed, peer);
        // Broadcast outside the lane-zero branch: source lanes must participate.
        float4 seedState = WaveReadLaneAt(prefetchedState, step & 31u);
        float3 seedGrad = WaveReadLaneAt(prefetchedGrad, step & 31u);
        uint seedProcessed = WaveReadLaneAt(prefetchedProcessed, step & 31u);
        state = lane == 0u ? seedState : shiftedState;
        pixelGrad = lane == 0u ? seedGrad : shiftedGrad;
        processed = lane == 0u ? seedProcessed : shiftedProcessed;
        int rank = int(step) - int(lane);
        if (rank >= 0 && rank < 256 && id < params.gaussianCount && itemOffset < processed) {
            uint2 pixel = origin + uint2(uint(rank) & 15u, uint(rank) >> 4u);
            GaussianSample sample = evaluateGaussianSample(gaussian, pixelCenter(pixel));
            if (contributesToPixel(sample)) {
                float u = max(1.0 - sample.alpha, 1e-6);
                state.w /= u;
                float dAlpha = dot(pixelGrad, state.w * (sample.color - state.xyz));
                float dPower = dAlpha * gaussian.conicOpacity.w * sample.coverage;
                float dx = sample.delta.x;
                float dy = sample.delta.y;
                // Match backwardTensorPair, including its conic.y and opacity conventions.
                gradCenter += dPower * float2(gaussian.conicOpacity.x * dx + gaussian.conicOpacity.y * dy,
                                              gaussian.conicOpacity.y * dx + gaussian.conicOpacity.z * dy);
                gradConicOpacity += float4(dPower * (-0.5 * dx * dx),
                                           dPower * (-0.5 * dx * dy),
                                           dPower * (-0.5 * dy * dy),
                                           dPower / max(gaussian.conicOpacity.w, 1e-20));
                gradColor += pixelGrad * (state.w * sample.alpha);
                state.xyz = sample.alpha * sample.color + u * state.xyz;
            }
        }
    }
    if (id < params.gaussianCount &&
        (any(gradCenter != float2(0.0)) || any(gradConicOpacity != float4(0.0)) || any(gradColor != float3(0.0)))) {
        atomicAddFloat(projectedGrads[id].centerRadius.x, gradCenter.x);
        atomicAddFloat(projectedGrads[id].centerRadius.y, gradCenter.y);
        atomicAddFloat(projectedGrads[id].conicOpacity.x, gradConicOpacity.x);
        atomicAddFloat(projectedGrads[id].conicOpacity.y, gradConicOpacity.y);
        atomicAddFloat(projectedGrads[id].conicOpacity.z, gradConicOpacity.z);
        atomicAddFloat(projectedGrads[id].conicOpacity.w, gradConicOpacity.w);
        atomicAddFloat(projectedGrads[id].color.x, gradColor.x);
        atomicAddFloat(projectedGrads[id].color.y, gradColor.y);
        atomicAddFloat(projectedGrads[id].color.z, gradColor.z);
    }
}
